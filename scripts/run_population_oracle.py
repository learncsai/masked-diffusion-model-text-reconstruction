"""Evaluate both population laws on the existing matched synthetic controls."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from diffusion_lm_rmt.population_calibration import SparsePopulation
from diffusion_lm_rmt.population_oracle import exact_forecasts
from diffusion_lm_rmt.reference_correction import sample_source_chunks
from diffusion_lm_rmt.categorical_lag import offset_list, pattern_weights
from diffusion_lm_rmt.calibration_diagnostics import score_weights
from diffusion_lm_rmt import sparse_categorical_lag as lag


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def context(config, ci):
    case = config['cases'][ci]
    def stream(*tags):
        return np.random.default_rng(np.random.SeedSequence([config['seed'], ci, *tags]))
    spec = {**config['population'], **{k:v for k,v in case.items() if k in config['population']}}
    population = SparsePopulation(spec, stream(0))
    def sample(chunks, tag):
        return sample_source_chunks(population, chunks, case['chunks_per_source'], stream(tag))[0]
    prior = np.bincount(sample(config['prior_chunks'], 1).ravel(), minlength=spec['classes']).astype(float)
    prior /= prior.sum()
    offsets = offset_list(config['radius'])
    gram, cross = lag.fitting_equations(sample(config['auxiliary_chunks'], 2),
        sample(config['auxiliary_chunks'], 3), prior, offsets, case['smoothing'],
        config['mask_rate'], stream(4))
    evaluation = sample(config['evaluation_chunks'], 5)
    mask = stream(6).random(evaluation.shape) < config['mask_rate']
    rows, _, contexts, _, patterns = lag.context_design(evaluation, mask, offsets)
    weights = np.zeros((len(rows), len(offsets)))
    for code in np.unique(patterns):
        if code:
            active = tuple(j for j in range(len(offsets)) if code & (1 << j))
            weights[np.ix_(np.flatnonzero(patterns == code), active)] = pattern_weights(
                gram, cross, active, config['ridge'])
    return population, np.asarray(contexts).T, weights, prior, offsets, score_weights(mask)


def summarize(root, out, config):
    rows = []
    for config_path, ci in config['oracle_cases']:
        setup = read(root/config_path)
        case = setup['cases'][ci]
        name = case['name']
        with np.load(out/f'{name}_moments.npz') as z:
            moments = {k:z[k] for k in z.files}
        old = root/setup['output']/name
        finite = read(old/'summary.json')
        for ni, n in enumerate(setup['sizes']):
            with np.load(old/f'observed_n{n}.npz') as z:
                obs = z['exact'] @ z['point_weight']
                linear = z['linear'] @ z['point_weight']
                np.testing.assert_allclose(moments['point_weight'], z['point_weight'])
            rng = np.random.default_rng(config['seed']+ci+n+int(case['smoothing']))
            indices = rng.integers(len(obs), size=(config['bootstrap_draws'],len(obs)))
            ob = obs[indices].mean(axis=1)
            lb = linear[indices].mean(axis=1)
            predictions = np.array([moments[k][ni] @ moments['point_weight'] for k in ('source','multinomial')])
            row = dict(case=name, n=n, alpha=case['smoothing'], pairs=len(obs),
                       observed=float(obs.mean()), population_linear_mc=float(linear.mean()),
                       population_linear_mc_ci=np.quantile(lb,[.025,.975]).tolist(),
                       source_oracle=float(predictions[0]), multinomial_oracle=float(predictions[1]),
                       source_ratio=float(predictions[0]/obs.mean()),
                       multinomial_ratio=float(predictions[1]/obs.mean()),
                       ratio_ci=np.quantile(predictions[None]/ob[:,None],[.025,.975],axis=0).T.tolist(),
                       source_minus_multinomial_error=float(abs(np.log(predictions[0]/obs.mean()))-abs(np.log(predictions[1]/obs.mean()))),
                       paired_error_ci=np.quantile(abs(np.log(predictions[0]/ob))-abs(np.log(predictions[1]/ob)),[.025,.975]).tolist(),
                       finite=[r for r in finite if r['n']==n])
            rows.append(row)
    (out/'oracle_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    for r in rows:
        print(r['case'],r['n'],'oracle ratios',round(r['source_ratio'],4),round(r['multinomial_ratio'],4),flush=True)
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--summarize-only',action='store_true')
    ap.add_argument('--root',type=Path,default=ROOT)
    a=ap.parse_args()
    config=read(a.root/'configs/oracle_multiplier_v1.json')
    out=a.root/config['output'];out.mkdir(parents=True,exist_ok=True)
    if not a.summarize_only:
        start=time.perf_counter()
        for config_path,ci in config['oracle_cases']:
            setup=read(a.root/config_path);case=setup['cases'][ci];name=case['name']
            dest=out/f'{name}_moments.npz'
            if dest.exists():
                print('Saved oracle:',name,flush=True)
                continue
            population,contexts,weights,prior,offsets,point_weight=context(setup,ci)
            def progress(i,total):
                if i%4==0:print(name,f'topic {i}/{total}',f'{time.perf_counter()-start:.1f}s',flush=True)
            result=exact_forecasts(population,contexts,weights,prior,offsets,setup['sizes'],
                case['smoothing'],case['chunks_per_source'],progress)
            np.savez_compressed(dest,**result,point_weight=point_weight)
        (out/'oracle_provenance.json').write_text(json.dumps(dict(
            seconds=time.perf_counter()-start,config_sha256=hashlib.sha256((a.root/'configs/oracle_multiplier_v1.json').read_bytes()).hexdigest(),
            source_sha256={p:hashlib.sha256((a.root/p).read_bytes()).hexdigest() for p in
                ['src/diffusion_lm_rmt/population_oracle.py','scripts/run_population_oracle.py']},
            exact_population_means=True,exact_covariance_contraction=True,
            no_reference_or_outcomes_in_oracle=True),indent=2)+'\n')
    summarize(a.root,out,config)


if __name__=='__main__':
    main()
