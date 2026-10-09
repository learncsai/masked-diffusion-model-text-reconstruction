"""Freeze, forecast, then score a six-bank, twenty-pair text experiment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.evaluate_lag_disagreement import read_json,write_json,sha,utc,source_groups,validate_records
from scripts.evaluate_lag_disagreement_extension import load_context
from diffusion_lm_rmt.source_moment_forecast import reference_forecasts,source_split_jackknife
from diffusion_lm_rmt.reference_correction import crossed_interval
from diffusion_lm_rmt import sparse_categorical_lag as lag
from diffusion_lm_rmt.categorical_lag import offset_list,per_sequence_squared_error

CONFIG=ROOT/'configs/rare_context_prospective_v1.json'
PROTOCOL=ROOT/'docs/rare_context_prospective_protocol.txt'
CODE=['scripts/run_rare_context_study.py','scripts/prepare_rare_context_study.py',
      'src/diffusion_lm_rmt/source_moment_forecast.py','src/diffusion_lm_rmt/reference_correction.py',
      'src/diffusion_lm_rmt/reference_ratio.py','src/diffusion_lm_rmt/sparse_categorical_lag.py',
      'src/diffusion_lm_rmt/categorical_lag.py','src/diffusion_lm_rmt/neural_data.py',
      'tests/test_source_moment_forecast.py','tests/test_reference_correction.py']


def paths():
    config=read_json(CONFIG)
    output=ROOT/config['output']
    output.mkdir(parents=True,exist_ok=True)
    return config,output,ROOT/config['inputs']


def initialize():
    config,out,_=paths()
    gate=[]
    hashes={}
    for case in ['iid_sparse','markov_sparse','clustered_sparse']:
        folder=ROOT/'results/rare_context_development_v1'/case
        data=read_json(folder/'summary.json')
        errors={method:float(np.mean([r['absolute_log_error'] for r in data
            if r['method']==method and r['m']<r['n']]))
            for method in ['source_full','source_jackknife','multinomial']}
        positive=True
        for path in folder.glob('forecast_*.npz'):
            with np.load(path) as z:
                positive &= all(float(z[key])>0 for key in z.files if key.endswith('jackknife_mean_raw'))
        assert errors['source_jackknife']<errors['source_full'] and positive, f'Development gate failed: {case}'
        gate.append(dict(case=case,errors=errors,all_scalar_forecasts_positive=bool(positive),passed=True))
        hashes[str((folder/'summary.json').relative_to(ROOT))]=sha(folder/'summary.json')
    contract=dict(config_sha256=sha(CONFIG),protocol_sha256=sha(PROTOCOL),
        code_sha256={name:sha(ROOT/name) for name in CODE},development_results_sha256=hashes,
        development_gate=gate,external_preregistration=False)
    signature=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    path=out/'method_saved.json'
    if path.exists():
        assert read_json(path)['signature']==signature,'Code or design changed after saving the method.'
    else:
        assert not list(out.glob('*/observed_n*.npz'))
        write_json(path,dict(saved_utc=utc(),signature=signature,**contract))
        snapshot=out/'code'
        snapshot.mkdir(exist_ok=True)
        for name in CODE:
            target=snapshot/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ROOT/name).read_bytes())
    return signature


def prepared(corpus):
    config,_,inputs=paths()
    folder=inputs/corpus
    done=read_json(folder/'complete.json')
    assert done['config_sha256']==sha(CONFIG)
    for name,digest in done['files_sha256'].items():
        assert sha(folder/name)==digest,name
    return folder


def forecast(corpus):
    signature=initialize()
    config,out,_=paths()
    folder=prepared(corpus)
    dest=out/corpus
    dest.mkdir(exist_ok=True)
    (evaluation,mask,prior,gram,cross),_,_,_,original=load_context(folder)
    assert len(evaluation)==config['evaluation_chunks']
    for key in ['radius','smoothing','ridge','mask_rate']:
        assert original[key]==config[key]
    offsets=offset_list(config['radius'])
    kw=dict(smoothing=config['smoothing'],ridge=config['ridge'])
    started=time.perf_counter()
    files=[]
    for bank in range(config['reference_banks']):
        with np.load(folder/f'reference{bank}.npz') as z:
            tokens=z['ids']
        records=read_json(folder/f'reference{bank}.json')
        for m in [*config['reference_sizes'],config['larger_reference_size']]:
            path=dest/f'forecast_bank{bank}_m{m}.npz'
            files.append(path)
            if path.exists():
                with np.load(path) as z:
                    assert str(z['signature'])==signature
                continue
            groups=source_groups(records[:m])
            write_json(dest/'progress.json',dict(stage='forecasting',bank=bank,m=m,saved_utc=utc()))
            args=(tokens[:m],groups,evaluation,mask,prior,offsets,gram,cross,config['sizes'])
            if m==config['larger_reference_size']:
                predictions=reference_forecasts(*args,**kw)
            else:
                predictions=source_split_jackknife(*args,**kw,
                    seed=config['seed']+10000*config['corpora'].index(corpus)+100*bank+m)
            data=dict(signature=np.array(signature),saved_utc=np.array(utc()),
                      sources=len(groups),chunks=m)
            for n,value in predictions.items():
                for key in ['point_forecasts','point_weight','strata','minimum_row_count',
                            'jackknife_raw','jackknife_mean','jackknife_mean_raw','negative_point_fraction']:
                    if key in value:
                        data[f'n{n}_{key}']=value[key]
            np.savez_compressed(path,**data)
            print(f'{corpus} bank={bank+1}/{config["reference_banks"]} m={m}: all sizes saved ({time.perf_counter()-started:.1f}s)',flush=True)
    baseline=[]
    for bank in range(config['reference_banks']):
        with np.load(dest/f'forecast_bank{bank}_m{config["larger_reference_size"]}.npz') as z:
            n=config['decision']['baseline_size']
            baseline.append(float(z[f'n{n}_point_forecasts'][2]@z[f'n{n}_point_weight']))
    target=config['decision']['target_fraction']*float(np.median(baseline))
    write_json(dest/'target_saved.json',dict(saved_utc=utc(),target=target,
        baseline_size=config['decision']['baseline_size'],baseline_forecasts=baseline,
        fraction=config['decision']['target_fraction'],signature=signature))
    write_json(dest/'forecasts_complete.json',dict(saved_utc=utc(),signature=signature,
        input_manifest_sha256=sha(folder/'complete.json'),
        files_sha256={p.name:sha(p) for p in [*files,dest/'target_saved.json']}))
    write_json(dest/'progress.json',dict(stage='forecasts complete, outcomes not scored',saved_utc=utc()))


def freeze():
    signature=initialize()
    config,out,_=paths()
    path=out/'forecasts_saved.json'
    if path.exists():
        saved=read_json(path)
        assert saved['signature']==signature
        for name,digest in saved['files_sha256'].items():
            assert sha(out/name)==digest,name
        return saved
    assert not list(out.glob('*/observed_n*.npz')),'Outcomes were already scored'
    files={}
    for corpus in config['corpora']:
        prepared(corpus)
        done=read_json(out/corpus/'forecasts_complete.json')
        assert done['signature']==signature
        for name,digest in done['files_sha256'].items():
            assert sha(out/corpus/name)==digest
            files[f'{corpus}/{name}']=digest
    saved=dict(saved_utc=utc(),signature=signature,files_sha256=files,
        status='All methods, forecasts and decision targets saved before any new A/B outcome.',
        external_preregistration=False)
    write_json(path,saved)
    print('All-corpus forecasts saved before outcomes.',flush=True)
    return saved


def score(corpus):
    saved=freeze()
    config,out,_=paths()
    folder=prepared(corpus)
    dest=out/corpus
    (evaluation,mask,prior,gram,cross),_,_,_,_=load_context(folder)
    offsets=offset_list(config['radius'])
    started=time.perf_counter()
    for n in config['sizes']:
        path=dest/f'observed_n{n}.npz'
        if path.exists():
            with np.load(path) as z:
                assert str(z['forecast_freeze_sha256'])==sha(out/'forecasts_saved.json')
            continue
        points,observations,losses=[],[],[]
        for pair in range(config['pairs']):
            predictions=[]
            arms=[]
            for arm in ['a','b']:
                with np.load(folder/f'pair{pair}_{arm}.npz') as z:
                    tokens=z['ids'][:n]
                records=read_json(folder/f'pair{pair}_{arm}.json')[:n]
                validate_records(tokens,records)
                counts=lag.pair_counts(tokens,offsets,len(prior))
                tables=lag.conditional_tables(counts,prior,config['smoothing'])
                rows,targets,prediction=lag.masked_prediction(evaluation,mask,tables,prior,
                    offsets,gram,cross,config['ridge'])
                predictions.append(prediction)
                arms.append(per_sequence_squared_error(rows,prediction.loss(targets),len(evaluation)).mean())
            distance=lag.squared_difference(*predictions)
            assert np.isfinite(distance).all() and np.min(distance,initial=0)>=-1e-12
            if pair in [0,config['pairs']-1]:
                for index in [0,len(rows)//2,len(rows)-1]:
                    a,b=[p.residual[index].toarray().ravel()+p.prior_weight[index]*prior for p in predictions]
                    np.testing.assert_allclose(distance[index],np.sum((a-b)**2),rtol=2e-10,atol=1e-12)
            points.append(distance)
            observations.append(per_sequence_squared_error(rows,distance,len(evaluation)))
            losses.append(arms)
        prior_loss=per_sequence_squared_error(rows,1-2*prior[targets]+prior@prior,len(evaluation)).mean()
        np.savez_compressed(path,point_observed=np.asarray(points),observed=np.asarray(observations),
            mse=np.asarray(losses),unigram_mse=prior_loss,relative_mse_improvement=1-np.asarray(losses)/prior_loss,
            saved_utc=np.array(utc()),signature=np.array(saved['signature']),
            forecast_freeze_sha256=np.array(sha(out/'forecasts_saved.json')))
        print(f'{corpus} n={n}: scored {config["pairs"]} new pairs ({time.perf_counter()-started:.1f}s)',flush=True)
    write_json(dest/'progress.json',dict(stage='outcomes complete',saved_utc=utc()))


def write_csv(path,records):
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def summarize():
    freeze()
    config,out,_=paths()
    rows,decisions,coverage=[],[],[]
    all_forecasts,all_observed=[],[]
    for ci,corpus in enumerate(config['corpora']):
        dest=out/corpus
        target=read_json(dest/'target_saved.json')['target']
        observations=[]
        for n in config['sizes']:
            with np.load(dest/f'observed_n{n}.npz') as z:
                observations.append(z['observed'].mean(axis=1))
        observed=np.asarray(observations).T
        predictions=np.zeros((config['reference_banks'],len(config['reference_sizes']),len(config['sizes']),3))
        large=np.zeros((config['reference_banks'],len(config['sizes'])))
        for ni,n in enumerate(config['sizes']):
            with np.load(dest/f'observed_n{n}.npz') as z:
                point_observed=z['point_observed'].mean(axis=0)
            for mi,m in enumerate(config['reference_sizes']):
                for bank in range(config['reference_banks']):
                    with np.load(dest/f'forecast_bank{bank}_m{m}.npz') as z:
                        point=z[f'n{n}_point_forecasts']
                        weight=z[f'n{n}_point_weight']
                        predictions[bank,mi,ni]=[point[2]@weight,float(z[f'n{n}_jackknife_mean']),point[0]@weight]
                        for category,label in enumerate(['0','1-4','5-19','20-99','100+','no_visible_offset']):
                            w=weight*(z[f'n{n}_strata']==category)
                            coverage.append(dict(corpus=corpus,n=n,m=m,bank=bank,stratum=label,
                                mass=float(w.sum()),observed=float(point_observed@w),
                                source_full=float(point[2]@w),source_jackknife_raw=float(z[f'n{n}_jackknife_raw']@w),
                                multinomial=float(point[0]@w)))
                inferred=crossed_interval(predictions[:,mi,ni],observed[:,ni],seed=config['seed']+ci*100000+mi*100+ni)
                for j,method in enumerate(config['methods']):
                    rows.append(dict(corpus=corpus,n=n,m=m,method=method,banks=config['reference_banks'],pairs=config['pairs'],
                        forecast=float(predictions[:,mi,ni,j].mean()),observed=float(observed[:,ni].mean()),
                        ratio=float(inferred['ratio'][j]),ratio_low=float(inferred['ratio_ci'][j,0]),ratio_high=float(inferred['ratio_ci'][j,1]),
                        mean_bank_absolute_log_error=float(inferred['mean_bank_absolute_log_error'][j]),
                        paired_error_difference=float(inferred['paired_error_difference'][j]),
                        difference_low=float(inferred['paired_error_difference_ci'][j,0]),difference_high=float(inferred['paired_error_difference_ci'][j,1])))
            for bank in range(config['reference_banks']):
                with np.load(dest/f'forecast_bank{bank}_m{config["larger_reference_size"]}.npz') as z:
                    large[bank,ni]=z[f'n{n}_point_forecasts'][2]@z[f'n{n}_point_weight']
            inferred=crossed_interval(large[:,ni,None],observed[:,ni],seed=config['seed']+ci*10000+ni)
            rows.append(dict(corpus=corpus,n=n,m=config['larger_reference_size'],method='source_large_reference',
                banks=config['reference_banks'],pairs=config['pairs'],forecast=float(large[:,ni].mean()),observed=float(observed[:,ni].mean()),
                ratio=float(inferred['ratio'][0]),ratio_low=float(inferred['ratio_ci'][0,0]),ratio_high=float(inferred['ratio_ci'][0,1]),
                mean_bank_absolute_log_error=float(inferred['mean_bank_absolute_log_error'][0]),paired_error_difference='',difference_low='',difference_high=''))
        rng=np.random.default_rng(config['seed']+90000+ci)
        samples=rng.integers(config['pairs'],size=(5000,config['pairs']))
        intervals=np.quantile(observed[samples].mean(axis=1),[.025,.975],axis=0).T
        for bank in range(config['reference_banks']):
            for mi,m in enumerate(config['reference_sizes']):
                for j,method in enumerate(config['methods']):
                    eligible=np.flatnonzero(predictions[bank,mi,:,j]<=target)
                    selected=int(eligible[0]) if len(eligible) else None
                    decisions.append(dict(corpus=corpus,bank=bank,m=m,method=method,target=target,
                        selected_n='' if selected is None else config['sizes'][selected],
                        forecast='' if selected is None else float(predictions[bank,mi,selected,j]),
                        observed='' if selected is None else float(observed[:,selected].mean()),
                        observed_low='' if selected is None else float(intervals[selected,0]),
                        observed_high='' if selected is None else float(intervals[selected,1]),
                        outcome='abstain' if selected is None else ('upper_interval_below_target' if intervals[selected,1]<=target else
                            ('mean_below_target' if observed[:,selected].mean()<=target else 'mean_above_target'))))
        all_forecasts.append(predictions)
        all_observed.append(observed)
        np.savez_compressed(dest/'analysis_arrays.npz',forecasts=predictions,observed=observed,large=large)
    # One crossed draw per corpus preserves every method, size and reference
    # prefix jointly. Corpora are held fixed rather than resampled as a sample
    # of three independent text domains.
    errors=[]
    primary_observed=[]
    rng=np.random.default_rng(config['seed']+888888)
    for forecasts,observed in zip(all_forecasts,all_observed):
        banks=rng.integers(config['reference_banks'],size=(5000,config['reference_banks']))
        pairs=rng.integers(config['pairs'],size=(5000,config['pairs']))
        sample_mean=observed[pairs].mean(axis=1)
        condition_errors=[]
        actual=[]
        for m,n in config['primary_cells']:
            mi,ni=config['reference_sizes'].index(m),config['sizes'].index(n)
            condition_errors.append(np.abs(np.log(forecasts[banks,mi,ni]/sample_mean[:,None,ni,None])).mean(axis=1))
            actual.append(np.abs(np.log(forecasts[:,mi,ni]/observed[:,ni].mean())).mean(axis=0))
        errors.append(np.mean(condition_errors,axis=0))
        primary_observed.append(np.mean(actual,axis=0))
    distribution=np.mean(errors,axis=0)
    estimate=np.mean(primary_observed,axis=0)
    primary=[]
    for j,method in enumerate(config['methods']):
        primary.append(dict(method=method,mean_absolute_log_error=float(estimate[j]),
            interval=np.quantile(distribution[:,j],[.025,.975]).tolist(),
            difference_vs_source=float(estimate[j]-estimate[0]),
            difference_vs_source_interval=np.quantile(distribution[:,j]-distribution[:,0],[.025,.975]).tolist()))
    write_csv(out/'summary.csv',rows)
    write_csv(out/'decisions.csv',decisions)
    write_csv(out/'coverage.csv',coverage)
    write_json(out/'primary_result.json',dict(primary_cells=config['primary_cells'],methods=primary,
        fixed_corpora=config['corpora'],banks_per_corpus=config['reference_banks'],pairs_per_corpus=config['pairs'],
        reference_and_pair_uncertainty=True,auxiliary_and_evaluation_conditioned_on=True,
        jackknife_vs_multinomial=float(estimate[1]-estimate[2]),
        jackknife_vs_multinomial_interval=np.quantile(distribution[:,1]-distribution[:,2],[.025,.975]).tolist()))
    write_json(out/'complete.json',dict(complete=True,saved_utc=utc(),
        forecast_freeze_sha256=sha(out/'forecasts_saved.json'),
        files_sha256={name:sha(out/name) for name in ['summary.csv','decisions.csv','coverage.csv','primary_result.json']}))
    print(json.dumps(read_json(out/'primary_result.json'),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['initialize','forecast','freeze','score','summarize'])
    parser.add_argument('--corpus',choices=['tinystories','wikitext','cnn_dailymail'])
    args=parser.parse_args()
    if args.stage in ['forecast','score']:
        if not args.corpus:
            parser.error('--corpus is required')
        globals()[args.stage](args.corpus)
    else:
        globals()[args.stage]()
