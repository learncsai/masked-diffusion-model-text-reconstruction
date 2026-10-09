"""Frozen reference-ratio follow-up: prepare, forecast, freeze, then score."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.evaluate_lag_disagreement import (CORPORA,CORE,METHODS,lag,offset_list,
    read_json,sha,source_groups,utc,validate_records,write_csv,write_json)
from scripts.evaluate_lag_disagreement_extension import load_context
from diffusion_lm_rmt.reference_ratio import reference_forecasts

OUT=ROOT/'results/reference_ratio_v1'
OLD=ROOT/'results/lag_reference_sensitivity_v1'
MATCHED=ROOT/'results/lag_disagreement_matched_v1'
NEW_CELLS={(512,512),(512,1024),(1024,1024),(512,4096),(1024,4096),
           (2048,4096),(512,8192),(1024,8192),(2048,8192),(2048,6144)}
FRESH_CELLS={(2048,2048),(2048,4096),(2048,6144),(2048,8192),
             (4096,4096),(4096,8192),(4096,12288)}
REFERENCE_GRID={512:[512],1024:[512,1024],2048:[2048],4096:[512,1024,2048,4096],
                6144:[2048],8192:[512,1024,2048,4096],12288:[4096]}
CODE=[*CORE,'src/diffusion_lm_rmt/reference_ratio.py','src/diffusion_lm_rmt/neural_data.py',
      'scripts/run_reference_ratio.py','scripts/evaluate_lag_disagreement.py',
      'scripts/evaluate_lag_disagreement_extension.py']
BIN_NAMES=['0','1-4','5-19','20-99','100+','no_visible_offset']


def initialize():
    frozen=read_json(OUT/'protocol_freeze.json')
    assert sha(ROOT/frozen['protocol_path'])==frozen['protocol_sha256']
    contract=dict(protocol_freeze_sha256=sha(OUT/'protocol_freeze.json'),
        code_sha256={name:sha(ROOT/name) for name in CODE},
        prior_inputs_manifest_sha256=sha(OLD/'inputs/manifest.json'),
        new_cells=sorted(NEW_CELLS),fresh_cells=sorted(FRESH_CELLS),
        separate_reference_cells=[[2048,4096],[4096,4096]],methods=list(METHODS),
        source_seed=20261017,corpora=list(CORPORA),pairs=3,
        definition='mean forecast divided by arithmetic mean observed disagreement',
        status='Post hoc hypothesis, specified follow-up. Fresh A/B outcomes scored only after global forecast freeze.')
    signature=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    path=OUT/'analysis_plan.json'
    if path.exists():
        assert read_json(path)['signature']==signature,'Code or design changed after freeze'
    else:
        write_json(path,dict(saved_utc=utc(),signature=signature,**contract))
    return signature


def prepare(corpus):
    from transformers import AutoTokenizer
    from diffusion_lm_rmt.neural_data import _chunk_split
    initialize()
    dest=OUT/'inputs'/corpus
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'prepared.json').exists():
        validate_inputs(corpus)
        return
    old=OLD/'inputs'/corpus
    manifest=read_json(OLD/'inputs/manifest.json')
    for rel,digest in manifest['inputs_sha256'].items():
        if rel.startswith(corpus+'/'): assert sha(OLD/'inputs'/rel)==digest,rel
    rawdir=ROOT/'data/diffusion_llm/lag_disagreement_extension_v1'/corpus
    provenance=read_json(rawdir/'provenance.json')
    assert sha(rawdir/'raw_documents.jsonl')==provenance['raw_sha256']
    raw=[json.loads(line) for line in (rawdir/'raw_documents.jsonl').read_text(encoding='utf-8').splitlines()]
    assert all(hashlib.sha256(d['text'].encode()).hexdigest()==d['sha256'] for d in raw)
    lookup=read_json(old/'source_text_hashes.json')
    excluded=read_json(old/'excluded_sources.json')
    ids,texts,chunks=(set(excluded[k]) for k in ('source_ids','text_hashes','chunk_hashes'))
    for role in read_json(old/'source_audit.json')['roles'].values():
        ids.update(role['source_ids'])
    record_files=[old/'reference_max.json',old/'evaluation_records.json']
    record_files += [old/f'pair{pair}_{arm}.json' for pair in (1,2,3) for arm in ('a','b')]
    for path in record_files:
        records=read_json(path)
        ids.update(r['document_id'] for r in records)
        chunks.update(r['sha256'] for r in records)
    texts.update(lookup[s] for s in ids if s in lookup)
    eligible=[d for d in raw if d['document_id'] not in ids and d['sha256'] not in texts]
    initial=len(eligible)
    seed=20261017+CORPORA.index(corpus)
    np.random.default_rng(seed).shuffle(eligible)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'data/diffusion_llm/gpt2_tokenizer',local_files_only=True)
    vocab_sha=hashlib.sha256(json.dumps(sorted(tokenizer.get_vocab().items(),key=lambda x:x[1]),
        ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    oldprep=read_json(old/'extension_complete.json')
    assert vocab_sha==oldprep['tokenizer_sha256']
    role_records={}
    for role in ['reference_extension','reference_separate',*[f'fresh{p}_{a}' for p in (1,2,3) for a in ('a','b')]]:
        print(f'Allocate {corpus} {role}: {len(eligible)} eligible sources',flush=True)
        tokens,attention,records=_chunk_split(tokenizer,eligible,4096,64,forbidden=chunks,
            max_chunks_per_document=oldprep['max_chunks_per_document'])
        assert len(tokens)==4096,f'Insufficient unused sources: {corpus} {role}'
        assert bool((attention==1).all())
        new_ids={r['document_id'] for r in records}
        new_texts={lookup[s] for s in new_ids}
        new_chunks={r['sha256'] for r in records}
        assert not new_ids&ids and not new_texts&texts and not new_chunks&chunks
        ids.update(new_ids); texts.update(new_texts); chunks.update(new_chunks)
        eligible=[d for d in eligible if d['document_id'] not in ids and d['sha256'] not in texts]
        array=tokens.numpy().astype(np.int32)
        validate_records(array,records)
        np.savez_compressed(dest/f'{role}.npz',ids=array)
        write_json(dest/f'{role}.json',records)
        role_records[role]=dict(chunks=len(records),sources=len(new_ids),source_ids=sorted(new_ids))
    with np.load(old/'reference_max.npz') as z: old_reference=z['ids']
    with np.load(dest/'reference_extension.npz') as z: extension=z['ids']
    combined=np.concatenate([old_reference,extension])
    records=read_json(old/'reference_max.json')+read_json(dest/'reference_extension.json')
    validate_records(combined,records)
    assert len({r['sha256'] for r in records})==12288
    np.savez_compressed(dest/'reference_nested.npz',ids=combined)
    write_json(dest/'reference_nested.json',records)
    for name in ('context.npz','evaluation_records.json','original_manifest.json','reference_original.npz','reference_original.json'):
        shutil.copyfile(old/name,dest/name)
    write_json(dest/'roles.json',role_records)
    write_json(dest/'prepared.json',dict(prepared_utc=utc(),corpus=corpus,seed=seed,
        initial_eligible_sources=initial,remaining_eligible_sources=len(eligible),
        source_text_chunk_overlaps=0,source_pool_provenance=provenance,tokenizer_sha256=vocab_sha,
        files_sha256={p.name:sha(p) for p in sorted(dest.iterdir()) if p.is_file() and p.name!='prepared.json'}))
    print(f'Prepared {corpus}: all new roles disjoint, {len(eligible)} sources remain',flush=True)


def validate_inputs(corpus):
    initialize()
    folder=OUT/'inputs'/corpus
    prep=read_json(folder/'prepared.json')
    for name,digest in prep['files_sha256'].items(): assert sha(folder/name)==digest,name
    return folder


def known_forecast(corpus,n,m):
    if m==2048 and n in (512,1024,2048):
        return MATCHED/'per_chunk'/f'{corpus}_n{n}_forecasts.npz'
    if m in (2048,4096,8192) and n in (4096,8192):
        return OLD/corpus/f'ref{m}_n{n}_forecasts.npz'
    return None


def forecast(corpus):
    signature=initialize()
    folder=validate_inputs(corpus)
    dest=OUT/corpus
    dest.mkdir(exist_ok=True)
    (evaluation,mask,prior,gram,cross),_,_,eval_records,config=load_context(folder)
    offsets=offset_list(config['radius'])
    checks=[]
    jobs=[('nested',m,ns) for m,ns in REFERENCE_GRID.items()]+[('separate',4096,[2048,4096])]
    for bank,m,sizes in jobs:
        paths={n:dest/f'{bank}_m{m}_n{n}_forecasts.npz' for n in sizes}
        missing=[n for n,p in paths.items() if not p.exists()]
        if not missing:
            continue
        with np.load(folder/f'reference_{bank}.npz') as z: reference=z['ids'][:m]
        records=read_json(folder/f'reference_{bank}.json')[:m]
        validate_records(reference,records)
        groups=source_groups(records)
        started=time.perf_counter()
        write_json(dest/'progress.json',dict(status='forecasting',bank=bank,m=m,sizes=missing,saved_utc=utc()))
        print(f'{corpus}: {bank} m={m}, target sizes {missing}, {len(groups)} sources',flush=True)
        results=reference_forecasts(reference,groups,evaluation,mask,prior,offsets,gram,cross,missing,
            smoothing=config['smoothing'],ridge=config['ridge'],
            progress=lambda done,total: print(f'{corpus} {bank} m={m}: {done}/{total} sources',flush=True))
        for n,result in results.items():
            prior_file=known_forecast(corpus,n,m) if bank=='nested' else None
            if prior_file is not None:
                with np.load(prior_file) as z:
                    expected=z['forecasts'][:3]
                np.testing.assert_allclose(result['forecasts'],expected,rtol=2e-11,atol=1e-12)
                checks.append(dict(n=n,m=m,prior_path=prior_file.relative_to(ROOT).as_posix(),
                    prior_sha256=sha(prior_file),max_difference=float(np.max(np.abs(result['forecasts']-expected)))))
            np.savez_compressed(paths[n],**result,methods=np.array(METHODS),signature=np.array(signature),
                saved_utc=np.array(utc()),source_ids=np.array([r['document_id'] for r in eval_records]),
                reference_sources=np.array(len(groups)),runtime_seconds=np.array(time.perf_counter()-started))
        print(f'{corpus}: saved {bank} m={m} forecasts ({time.perf_counter()-started:.1f}s)',flush=True)
    all_paths=[dest/f'{b}_m{m}_n{n}_forecasts.npz' for b,m,ns in jobs for n in ns]
    for p in all_paths:
        with np.load(p) as z: assert str(z['signature'])==signature
    write_json(dest/'forecasts_complete.json',dict(completed_utc=utc(),signature=signature,checks=checks,
        files_sha256={p.name:sha(p) for p in all_paths},prepared_sha256=sha(folder/'prepared.json')))
    write_json(dest/'progress.json',dict(status='forecasts_complete',saved_utc=utc()))


def freeze():
    signature=initialize()
    output=OUT/'forecasts_frozen.json'
    if output.exists(): return read_json(output)
    files={}
    for corpus in CORPORA:
        validate_inputs(corpus)
        done=read_json(OUT/corpus/'forecasts_complete.json')
        assert done['signature']==signature
        for name,digest in done['files_sha256'].items():
            assert sha(OUT/corpus/name)==digest
            files[f'{corpus}/{name}']=digest
        assert not (OUT/corpus/'fresh_n2048_observed.npz').exists()
        assert not (OUT/corpus/'fresh_n4096_observed.npz').exists()
    record=dict(saved_utc=utc(),signature=signature,files_sha256=files,
        status='All forecasts saved before scoring any fresh A/B corpus',external_preregistration=False)
    write_json(output,record)
    print('Global forecast freeze complete',flush=True)
    return record


def observed_points(folder,cohort,n,evaluation,mask,prior,offsets,gram,cross,config):
    points=[]
    for pair in (1,2,3):
        predictions=[]
        for arm in ('a','b'):
            name=f'fresh{pair}_{arm}' if cohort=='fresh' else f'pair{pair}_{arm}'
            with np.load(folder/f'{name}.npz') as z: tokens=z['ids'][:n]
            records=read_json(folder/f'{name}.json')[:n]
            validate_records(tokens,records)
            tables=lag.conditional_tables(lag.pair_counts(tokens,offsets,len(prior)),prior,config['smoothing'])
            rows,_,prediction=lag.masked_prediction(evaluation,mask,tables,prior,offsets,gram,cross,config['ridge'])
            predictions.append(prediction)
        distance=lag.squared_difference(*predictions)
        for index in (0,len(rows)//2,len(rows)-1):
            a,b=[p.residual[index].toarray().ravel()+p.prior_weight[index]*prior for p in predictions]
            np.testing.assert_allclose(distance[index],np.sum((a-b)**2),rtol=1e-11,atol=1e-12)
        assert np.isfinite(distance).all() and (distance>=0).all()
        points.append(distance)
    return rows,np.array(points)


def expectation(n,m):
    return {1:(.82,.88),2:(.93,.96),3:(.97,1.03),4:(.95,1.10)}.get(m/n)


def score(corpus):
    signature=initialize()
    frozen=read_json(OUT/'forecasts_frozen.json')
    assert frozen['signature']==signature
    for rel,digest in frozen['files_sha256'].items(): assert sha(OUT/rel)==digest,rel
    folder=validate_inputs(corpus)
    dest=OUT/corpus
    (evaluation,mask,prior,gram,cross),_,_,_,config=load_context(folder)
    offsets=offset_list(config['radius'])
    observations={}
    from diffusion_lm_rmt.categorical_lag import per_sequence_squared_error
    for cohort,sizes in [('known',(512,1024,2048)),('fresh',(2048,4096))]:
        for n in sizes:
            path=dest/f'{cohort}_n{n}_observed.npz'
            if path.exists():
                with np.load(path) as z:
                    assert str(z['signature'])==signature
                    observations[cohort,n]=(z['rows'].copy(),z['point_observed'].copy(),z['observed'].copy())
                continue
            source=folder if cohort=='fresh' else OLD/'inputs'/corpus
            rows,points=observed_points(source,cohort,n,evaluation,mask,prior,offsets,gram,cross,config)
            actual=np.array([per_sequence_squared_error(rows,p,len(evaluation)) for p in points])
            if cohort=='known':
                with np.load(MATCHED/'per_chunk'/f'{corpus}_n{n}_observed.npz') as z:
                    np.testing.assert_allclose(actual,z['observed'],rtol=1e-12,atol=1e-14)
            np.savez_compressed(path,rows=rows,point_observed=points,observed=actual,
                signature=np.array(signature),saved_utc=np.array(utc()),forecast_freeze_sha256=np.array(sha(OUT/'forecasts_frozen.json')))
            observations[cohort,n]=(rows,points,actual)
            print(f'{corpus}: {cohort} n={n}, mean observed {actual.mean():.8f}',flush=True)
    summaries,pair_rows,strata_rows=[],[],[]
    jobs=[('known','nested',n,m) for n,m in sorted(NEW_CELLS)]
    jobs += [('fresh','nested',n,m) for n,m in sorted(FRESH_CELLS)]
    jobs += [('fresh','separate',n,4096) for n in (2048,4096)]
    for cohort,bank,n,m in jobs:
        rows,point_observed,actual=observations[cohort,n]
        path=dest/f'{bank}_m{m}_n{n}_forecasts.npz'
        with np.load(path) as z:
            np.testing.assert_array_equal(rows,z['rows'])
            forecasts=z['forecasts'];point_forecasts=z['point_forecasts'];weights=z['point_weight'];bins=z['strata']
        means=actual.mean(axis=1)
        mean=means.mean()
        margin=t.ppf(.975,len(means)-1)*means.std(ddof=1)/np.sqrt(len(means))
        for j,method in enumerate(METHODS):
            predicted=forecasts[j].mean()
            ratio=predicted/mean
            expected=expectation(n,m) if method=='source_full' else None
            row=dict(corpus=corpus,cohort=cohort,bank=bank,n=n,m=m,reference_target_ratio=m/n,
                method=method,pairs=3,predicted_mean=float(predicted),observed_mean=float(mean),
                ratio=float(ratio),lower=float(predicted/(mean+margin)),
                upper=float(predicted/(mean-margin)) if mean>margin else None,
                mean_absolute_pair_log_error=float(np.mean(np.abs(np.log(predicted/means)))),
                expected_lower=expected[0] if expected else None,expected_upper=expected[1] if expected else None,
                within_expected=bool(expected[0]<=ratio<=expected[1]) if expected else None)
            summaries.append(row)
            for pair,obs in enumerate(means,1):
                pair_rows.append(dict(corpus=corpus,cohort=cohort,bank=bank,n=n,m=m,method=method,
                    pair=pair,predicted=float(predicted),observed=float(obs),ratio=float(predicted/obs)))
            predicted_sum=0.;observed_sum=0.
            for b,label in enumerate(BIN_NAMES):
                selected=bins==b
                pe=float(np.sum(point_forecasts[j,selected]*weights[selected]))
                oe=float(np.mean(point_observed[:,selected] @ weights[selected]))
                predicted_sum+=pe;observed_sum+=oe
                strata_rows.append(dict(corpus=corpus,cohort=cohort,bank=bank,n=n,m=m,method=method,
                    stratum=label,masked_positions=int(selected.sum()),position_weight=float(weights[selected].sum()),
                    predicted_contribution=pe,observed_contribution=oe,signed_error_contribution=pe-oe,
                    observed_energy_share=oe/mean,predicted_observed_ratio=pe/oe if oe>0 else None))
            np.testing.assert_allclose([predicted_sum,observed_sum],[predicted,mean],rtol=1e-11,atol=1e-13)
    write_csv(dest/'summary.csv',summaries)
    write_csv(dest/'per_pair.csv',pair_rows)
    write_csv(dest/'coverage_strata.csv',strata_rows)
    write_json(dest/'complete.json',dict(completed_utc=utc(),signature=signature,
        forecast_freeze_sha256=sha(OUT/'forecasts_frozen.json'),
        output_sha256={p.name:sha(p) for p in dest.iterdir() if p.suffix in ('.csv','.npz')},
        known_observations_reconciled=True,dense_distance_checks=True,strata_sum_checks=True))
    write_json(dest/'progress.json',dict(status='complete',saved_utc=utc()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['initialize','prepare','forecast','freeze','score'])
    p.add_argument('--corpus',choices=CORPORA)
    args=p.parse_args()
    if args.stage=='initialize': initialize()
    elif args.stage=='freeze': freeze()
    else:
        for c in (args.corpus,) if args.corpus else CORPORA:
            globals()[args.stage](c)


if __name__=='__main__':main()
