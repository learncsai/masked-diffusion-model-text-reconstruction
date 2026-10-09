"""Known-population missing-row decomposition at the lag-contribution level.

An input with one unseen context may also have well-observed contexts. This
diagnostic separates the output vectors, retaining their cross term, rather
than attributing the whole input's error to its least-observed row.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.run_rare_context_development import write,utc,digest
from diffusion_lm_rmt.population_calibration import SparsePopulation
from diffusion_lm_rmt.reference_correction import sample_source_chunks
from diffusion_lm_rmt import sparse_categorical_lag as lag
from diffusion_lm_rmt.categorical_lag import offset_list,pattern_weights
from diffusion_lm_rmt.calibration_diagnostics import Linearization,score_weights


def components(evaluation,mask,tables_a,tables_b,prior,offsets,gram,cross,ridge):
    rows,_,contexts,visible,patterns=lag.context_design(evaluation,mask,offsets)
    weights=np.zeros((len(rows),len(offsets)))
    for code in np.unique(patterns):
        if code:
            active=tuple(j for j in range(len(offsets)) if code&(1<<j))
            weights[np.ix_(np.flatnonzero(patterns==code),active)]=pattern_weights(gram,cross,active,ridge)
    pieces=[]
    for j,d in enumerate(offsets):
        selected=np.flatnonzero(visible[j])
        select=sparse.csr_matrix((weights[selected,j],(selected,contexts[j][selected])),shape=(len(rows),len(prior)))
        pieces.append(lag.Prediction(select@(tables_a[d][0]-tables_b[d][0]),
            select@(tables_a[d][1]-tables_b[d][1]),prior))
    covariance=np.zeros((len(rows),len(offsets),len(offsets)))
    for j in range(len(offsets)):
        for k in range(j+1):
            covariance[:,j,k]=lag.row_inner(pieces[j],pieces[k])
            covariance[:,k,j]=covariance[:,j,k]
    return covariance


def run():
    config_path=ROOT/'configs/rare_context_development_v1.json'
    config=json.loads(config_path.read_text())
    root=ROOT/config['output']
    output=root/'missing_row_decomposition'
    output.mkdir(exist_ok=True)
    records=[]
    for ci,case in enumerate(config['cases'][:3]):
        def rng(*tags):
            return np.random.default_rng(np.random.SeedSequence([config['seed'],ci,*tags]))
        spec={**config['population'],**{k:v for k,v in case.items() if k in config['population']}}
        pop=SparsePopulation(spec,rng(0))
        def sample(n,*tags):
            return sample_source_chunks(pop,n,case['chunks_per_source'],rng(*tags))[0]
        prior=np.bincount(sample(config['prior_chunks'],1).ravel(),minlength=spec['classes']).astype(float)
        prior/=prior.sum()
        offsets=offset_list(config['radius'])
        gram,cross=lag.fitting_equations(sample(config['auxiliary_chunks'],2),sample(config['auxiliary_chunks'],3),
            prior,offsets,case['smoothing'],config['mask_rate'],rng(4))
        evaluation=sample(config['evaluation_chunks'],5)
        mask=rng(6).random(evaluation.shape)<config['mask_rate']
        _,_,contexts,visible,_=lag.context_design(evaluation,mask,offsets)
        weight=score_weights(mask)
        means=pop.mean_lag_counts(offsets)
        for n in config['sizes']:
            path=output/f'{case["name"]}_n{n}_lag_covariance.npz'
            if path.exists():
                with np.load(path) as z:
                    covariance=z['covariance']
            else:
                expansion=Linearization({d:n*c for d,c in means.items()},prior,case['smoothing'])
                covariance=[]
                for pair in range(config['pairs']):
                    tables=[expansion.tables_at(lag.pair_counts(sample(n,9,n,pair,arm),offsets,len(prior))) for arm in range(2)]
                    covariance.append(components(evaluation,mask,*tables,prior,offsets,gram,cross,config['ridge']))
                covariance=np.asarray(covariance)
                np.savez_compressed(path,covariance=covariance)
            with np.load(root/case['name']/f'observed_n{n}.npz') as z:
                expected=z['linear']
                actual=z['exact']@weight
            np.testing.assert_allclose(covariance.sum(axis=(2,3)),expected,rtol=1e-9,atol=1e-12)
            for m in config['reference_sizes']:
                banks=[]
                for bank in range(config['reference_banks']):
                    counts=lag.pair_counts(sample(m,7,m,bank),offsets,len(prior))
                    missing=np.zeros((len(weight),len(offsets)),dtype=bool)
                    for j,d in enumerate(offsets):
                        totals=np.asarray(counts[d].sum(axis=1)).ravel()
                        missing[:,j]=visible[j]&(totals[contexts[j]]==0)
                    seen=~missing
                    unseen_energy=np.einsum('pqjk,qj,qk,q->p',covariance,missing,missing,weight)
                    seen_energy=np.einsum('pqjk,qj,qk,q->p',covariance,seen,seen,weight)
                    twice_cross=expected@weight-unseen_energy-seen_energy
                    with np.load(root/case['name']/f'forecast_m{m}_bank{bank}.npz') as z:
                        forecast=float(z[f'n{n}_point_forecasts'][2]@weight)
                    # The empirical source law has exactly zero columns for
                    # unseen contexts, including their cross covariances.
                    coverage_gap=-unseen_energy-twice_cross
                    seen_error=forecast-seen_energy
                    propagation=expected@weight-actual
                    np.testing.assert_allclose(coverage_gap+seen_error+propagation,forecast-actual,rtol=1e-10,atol=1e-12)
                    banks.append(np.column_stack([unseen_energy,twice_cross,coverage_gap,seen_error,propagation]))
                banks=np.asarray(banks)
                generator=rng(500,n,m)
                bi=generator.integers(config['reference_banks'],size=(5000,config['reference_banks']))
                pi=generator.integers(config['pairs'],size=(5000,config['pairs']))
                values=np.zeros((5000,banks.shape[-1]))
                for draw in range(5000):
                    values[draw]=banks[bi[draw]][:,pi[draw]].mean(axis=(0,1))/actual[pi[draw]].mean()
                for j,name in enumerate(['unseen_energy','unseen_twice_cross','coverage_gap','seen_reference_error','propagation_error']):
                    low,high=np.quantile(values[:,j],[.025,.975])
                    records.append(dict(case=case['name'],n=n,m=m,component=name,
                        fraction=float(banks[:,:,j].mean()/actual.mean()),low=float(low),high=float(high)))
            print(f'{case["name"]} n={n}: exact missing/seen/cross decomposition verified',flush=True)
    write(output/'summary.json',records)
    write(output/'validation.json',dict(status='passed',saved_utc=utc(),code_sha256=digest(__file__),
        lag_covariances_sum_to_saved_population_linear_disagreement=True,
        missing_seen_propagation_components_sum_to_total_error=True,
        synthetic_only=True,real_text_outcomes_not_read=True))


if __name__=='__main__':
    run()
