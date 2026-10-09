"""Separate finite-reference and propagation errors with a known population."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from diffusion_lm_rmt.population_calibration import SparsePopulation
from diffusion_lm_rmt.reference_correction import sample_source_chunks, source_split_jackknife, crossed_interval
from diffusion_lm_rmt import sparse_categorical_lag as lag
from diffusion_lm_rmt.categorical_lag import offset_list
from diffusion_lm_rmt.calibration_diagnostics import Linearization, score_weights


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def utc():
    return datetime.now(timezone.utc).isoformat()


def run(config_path, case_names=None, pilot=False):
    started = time.perf_counter()
    config = json.loads(config_path.read_text())
    output = ROOT/config['output']
    if pilot:
        output = output/'pilot'
    output.mkdir(parents=True, exist_ok=True)
    code = [Path(__file__), ROOT/'src/diffusion_lm_rmt/reference_correction.py',
            ROOT/'src/diffusion_lm_rmt/reference_ratio.py',
            ROOT/'src/diffusion_lm_rmt/population_calibration.py',
            ROOT/'src/diffusion_lm_rmt/sparse_categorical_lag.py',
            ROOT/'src/diffusion_lm_rmt/calibration_diagnostics.py']
    contract = dict(config=config, config_sha256=digest(config_path), pilot=pilot,
                    code_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in code})
    signature = hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    plan = output/'design.json'
    if plan.exists():
        assert json.loads(plan.read_text())['signature'] == signature, 'Design changed. Use a new version.'
    else:
        write(plan, dict(saved_utc=utc(), signature=signature, **contract))
    for ci, case in enumerate(config['cases']):
        if case_names and case['name'] not in case_names:
            continue
        dest = output/case['name']
        dest.mkdir(exist_ok=True)
        if (dest/'complete.json').exists():
            print(f"Already complete: {case['name']}", flush=True)
            continue
        def stream(*tags):
            return np.random.default_rng(np.random.SeedSequence([config['seed'], ci, *tags]))
        spec = {**config['population'], **{k:v for k,v in case.items() if k in config['population']}}
        population = SparsePopulation(spec, stream(0))
        cps = case['chunks_per_source']
        def sample(chunks, *tags):
            return sample_source_chunks(population, chunks, cps, stream(*tags))
        prior_ids,_ = sample(config['prior_chunks'],1)
        prior = np.bincount(prior_ids.ravel(), minlength=spec['classes']).astype(float)
        prior /= prior.sum()
        offsets = offset_list(config['radius'])
        auxiliary_a,_ = sample(config['auxiliary_chunks'],2)
        auxiliary_b,_ = sample(config['auxiliary_chunks'],3)
        gram,cross = lag.fitting_equations(auxiliary_a,auxiliary_b,prior,offsets,
            case['smoothing'],config['mask_rate'],stream(4))
        evaluation,_ = sample(config['evaluation_chunks'],5)
        mask = stream(6).random(evaluation.shape) < config['mask_rate']
        weights = score_weights(mask)
        mean = population.mean_lag_counts(offsets)
        sizes = config['sizes']
        def predict(tables):
            return lag.masked_prediction(evaluation,mask,tables,prior,offsets,
                gram,cross,config['ridge'])[2]
        banks = 2 if pilot else config['reference_banks']
        ms = config['reference_sizes'][:2] if pilot else config['reference_sizes']
        forecast_files = []
        for m in ms:
            for bank in range(banks):
                path = dest/f'forecast_m{m}_bank{bank}.npz'
                forecast_files.append(path)
                if path.exists():
                    continue
                reference,groups = sample(m,7,m,bank)
                values = source_split_jackknife(reference,groups,evaluation,mask,prior,
                    offsets,gram,cross,sizes,smoothing=case['smoothing'],
                    ridge=config['ridge'],seed=int(stream(8,m,bank).integers(2**31)))
                data = {}
                for n,v in values.items():
                    for k in ['point_forecasts','point_weight','strata','minimum_row_count',
                              'jackknife_raw','jackknife_mean','jackknife_mean_raw','negative_point_fraction']:
                        data[f'n{n}_{k}'] = v[k]
                np.savez_compressed(path, **data)
                print(f"{case['name']} m={m} bank={bank+1}/{banks}: forecasts ({time.perf_counter()-started:.1f}s)",flush=True)
                write(output/'progress.json',dict(stage='development forecasts',case=case['name'],m=m,bank=bank+1,utc=utc()))
        freeze_path = dest/'forecasts_saved.json'
        if not freeze_path.exists():
            write(freeze_path,dict(saved_utc=utc(),signature=signature,
                files={p.name:digest(p) for p in forecast_files},
                note='All forecasts saved before this case A/B outcomes. Development experiment.'))
        records = json.loads(freeze_path.read_text())
        assert all(digest(dest/k)==v for k,v in records['files'].items())
        pairs = 24 if pilot else config['pairs']
        for n in sizes:
            path = dest/f'observed_n{n}.npz'
            if path.exists():
                continue
            expansion = Linearization({d:n*c for d,c in mean.items()},prior,case['smoothing'])
            exact,linear = [],[]
            for pair in range(pairs):
                pe,pl = [],[]
                for arm in range(2):
                    ids,_ = sample(n,9,n,pair,arm)
                    counts = lag.pair_counts(ids,offsets,len(prior))
                    pe.append(predict(lag.conditional_tables(counts,prior,case['smoothing'])))
                    pl.append(predict(expansion.tables_at(counts)))
                exact.append(lag.squared_difference(*pe))
                linear.append(lag.squared_difference(*pl))
            np.savez_compressed(path,exact=np.asarray(exact),linear=np.asarray(linear),point_weight=weights)
            print(f"{case['name']} n={n}: {pairs} paired nonlinear/oracle-linear outcomes ({time.perf_counter()-started:.1f}s)",flush=True)
        rows, strata = [],[]
        for n in sizes:
            with np.load(dest/f'observed_n{n}.npz') as z:
                exact,linear = z['exact'],z['linear']
            obs, lin = exact@weights, linear@weights
            for m in ms:
                forecasts, raw, labels = [],[],[]
                for bank in range(banks):
                    with np.load(dest/f'forecast_m{m}_bank{bank}.npz') as z:
                        point = z[f'n{n}_point_forecasts']
                        forecasts.append([point[2]@weights, float(z[f'n{n}_jackknife_mean']),point[0]@weights])
                        raw.append(point[2])
                        labels.append(z[f'n{n}_strata'])
                forecasts = np.asarray(forecasts)
                infer = crossed_interval(forecasts,obs,seed=5000+ci*100+n+m)
                for j,method in enumerate(['source_full','source_jackknife','multinomial']):
                    rows.append(dict(case=case['name'],n=n,m=m,method=method,pairs=pairs,banks=banks,
                        forecast=float(forecasts[:,j].mean()),observed=float(obs.mean()),
                        population_linear=float(lin.mean()),
                        ratio=float(infer['ratio'][j]),ratio_ci=infer['ratio_ci'][j].tolist(),
                        absolute_log_error=float(infer['mean_bank_absolute_log_error'][j]),
                        paired_error_difference=float(infer['paired_error_difference'][j]),
                        paired_error_difference_ci=infer['paired_error_difference_ci'][j].tolist(),
                        reference_error_fraction=float((forecasts[:,j].mean()-lin.mean())/obs.mean()),
                        propagation_error_fraction=float((lin.mean()-obs.mean())/obs.mean())))
                for category,name in enumerate(['0','1-4','5-19','20-99','100+','no_visible_offset']):
                    indicator = np.asarray(labels)==category
                    restricted = indicator*weights
                    oracle = restricted@linear.mean(axis=0)
                    observed = restricted@exact.mean(axis=0)
                    predicted = np.sum(np.asarray(raw)*restricted,axis=1)
                    strata.append(dict(case=case['name'],n=n,m=m,stratum=name,
                        evaluation_mass=float(restricted.sum(axis=1).mean()),
                        source_prediction=float(predicted.mean()),oracle_linear=float(oracle.mean()),
                        observed=float(observed.mean()),
                        reference_error_fraction=float((predicted-oracle).mean()/obs.mean()),
                        propagation_error_fraction=float((oracle-observed).mean()/obs.mean())))
        write(dest/'summary.json',rows)
        write(dest/'coverage_decomposition.json',strata)
        write(dest/'complete.json',dict(complete=True,signature=signature,finished_utc=utc(),
            duration_seconds=time.perf_counter()-started,code_sha256=contract['code_sha256']))
    print('Development stage finished.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'configs/rare_context_development_v1.json')
    parser.add_argument('--cases',nargs='+')
    parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args()
    run(args.config,args.cases,args.pilot)
