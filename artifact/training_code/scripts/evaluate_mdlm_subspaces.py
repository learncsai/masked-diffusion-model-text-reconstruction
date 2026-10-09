"""Evaluate frozen directional comparison using saved models, with no training."""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import csv
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from diffusion_lm_rmt.disagreement_subspaces import (
    vocabulary_readout, source_folds, second_moments, projected_prediction,
    forecast_matrices, capture, random_capture)
from diffusion_lm_rmt.categorical_lag import offset_list, source_groups
from diffusion_lm_rmt.matched_mdlm import digest, object_digest, read_json, write_json, load_model
from evaluate_mdlm_topk import prepare_jobs, CORPORA
from evaluate_full_context_linear import load_data, RUN
from evaluate_lag_disagreement import validate_records, write_csv

OUT=ROOT/'results/mdlm_subspaces_v1'
INPUT=ROOT/'results/lag_disagreement_matched_v1/inputs'
KS=(1,2,4,8,16)
METHODS=('reconstructor','source_full','source_same_lag','multinomial','affine','unigram')


def save(path, signature, **arrays):
    np.savez_compressed(path,signature=np.array(signature),**arrays)


def cached(path, signature):
    if not path.exists():return None
    with np.load(path) as z:
        assert str(z['signature'])==signature, f'Changed inputs/code: {path}'
        return {k:z[k].copy() for k in z.files if k!='signature'}


@torch.inference_mode()
def neural_delta(jobs, corpus, n, pair, updates, context, config, readout, device):
    selected=sorted([j for j in jobs if (j['corpus'],j['n'],j['pair'],j['updates'])==
                     (corpus,n,pair,updates)],key=lambda j:j['arm'])
    models=[load_model(config,Path(j['checkpoint']),device).eval() for j in selected]
    projection=torch.as_tensor(readout,dtype=torch.float32,device=device)
    ids,mask=context['evaluation'],context['mask']
    differences,full_energy=[],[]
    for start in range(0,len(ids),4):
        clean=torch.as_tensor(ids[start:start+4],device=device,dtype=torch.long)
        hidden=torch.as_tensor(mask[start:start+4],device=device,dtype=torch.bool)
        corrupt=clean.masked_fill(hidden,config['mask_token_id'])
        times=torch.full((len(clean),),config['mask_rate'],device=device)
        p=[m(corrupt,torch.ones_like(clean,dtype=torch.bool),times)[...,:len(readout)][hidden].float().softmax(-1)
           for m in models]
        delta=p[1]-p[0]
        differences.append((delta@projection).double().cpu().numpy())
        energy=delta.double().square().sum(-1).cpu().numpy()
        at=0
        for count in mask[start:start+4].sum(1):
            full_energy.append(float(energy[at:at+count].mean()))
            at+=count
    old=read_json(RUN/corpus/'scores'/f'pair{pair}_n{n}_step{updates}.json')['per_chunk']['disagreement']
    np.testing.assert_allclose(full_energy,old,rtol=2e-5,atol=2e-7)
    del models
    torch.cuda.empty_cache()
    return np.concatenate(differences),np.array(full_energy)


def summarize(out):
    rows=[]
    for meta_path in sorted(out.glob('*_inputs.json')):
        corpus=meta_path.stem.removesuffix('_inputs')
        meta=read_json(meta_path)
        for n in (512,1024,2048):
            reference_path=out/f'{corpus}_n{n}_reference.npz'
            if not reference_path.exists():continue
            with np.load(reference_path) as z:
                forecast={m:z[m] for m in ('source_full','source_same_lag','multinomial','unigram')}
            for pair in (1,2,3):
                ep=out/f'{corpus}_n{n}_pair{pair}_empirical.npz'
                if not ep.exists():continue
                with np.load(ep) as z:empirical={m:z[m] for m in ('reconstructor','affine')}
                for updates in (8000,12000):
                    path=out/f'{corpus}_n{n}_pair{pair}_step{updates}_neural.npz'
                    if not path.exists():continue
                    with np.load(path) as z: neural=z['neural']
                    for dimension in (64,128):
                        target=neural[:,:dimension,:dimension]
                        for mode in ('source_crossfit','pooled'):
                            null=random_capture(target,crossfit=mode=='source_crossfit')
                            for method,matrix in {**forecast,**empirical}.items():
                                source=matrix[:,:dimension,:dimension]
                                for j,k in enumerate(KS):
                                    rows.append(dict(corpus=corpus,n=n,pair=pair,updates=updates,
                                        dimension=dimension,mode=mode,method=method,k=k,
                                        capture=capture(source,target,k,crossfit=mode=='source_crossfit'),
                                        random_expectation=k/dimension,
                                        random_mean=float(null[:,j].mean()),
                                        random_p95=float(np.quantile(null[:,j],.95)),
                                        neural_trace=float(np.trace(target.sum(0))/128),
                                        method_trace=float(np.trace(source.sum(0))/128)))
    if not rows:return
    write_csv(out/'per_pair.csv',rows)
    grouped={}
    keys=('corpus','n','updates','dimension','mode','method','k')
    for r in rows:grouped.setdefault(tuple(r[k] for k in keys),[]).append(r)
    summary=[]
    for key,cell in sorted(grouped.items()):
        if len(cell)!=3:continue
        values=[r['capture'] for r in cell]
        summary.append(dict(zip(keys,key))|dict(pairs=3,capture=float(np.mean(values)),
            minimum=min(values),maximum=max(values),random_expectation=cell[0]['random_expectation'],
            random_p95_mean=float(np.mean([r['random_p95'] for r in cell]))))
    if summary:write_csv(out/'summary.csv',summary)
    print(json.dumps(dict(stage='summarized',pair_rows=len(rows),summary_rows=len(summary))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--corpora',nargs='+',default=list(CORPORA))
    p.add_argument('--sizes',nargs='+',type=int,default=[512,1024,2048])
    p.add_argument('--device',default='cuda')
    p.add_argument('--stage',choices=('all','forecast','empirical','neural','summarize'),default='all')
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if args.stage=='summarize':summarize(out);return
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    started=time.perf_counter()
    jobs,contexts,evidence=prepare_jobs(RUN,ROOT/'results/lag_topk_matched_v1',args.corpora)
    protocol=digest(ROOT/'docs/mdlm_subspace_protocol.txt')
    codes={p:digest(ROOT/p) for p in ('scripts/evaluate_mdlm_subspaces.py',
        'src/diffusion_lm_rmt/disagreement_subspaces.py')}
    readout=vocabulary_readout(50257)
    readout_path=out/'readout.npy'
    if readout_path.exists():np.testing.assert_array_equal(np.load(readout_path),readout)
    else:np.save(readout_path,readout)
    frozen=read_json(INPUT/'manifest.json')['inputs_sha256']
    for corpus,(config,ctx) in contexts.items():
        _,trains,records,original_hashes=load_data(corpus,config,ctx)
        folder=INPUT/corpus
        for name in ('reference_original.npz','reference_original.json','source_audit.json','context.npz'):
            assert digest(folder/name)==frozen[f'{corpus}/{name}']
        with np.load(folder/'context.npz') as z:
            for name in ('evaluation','mask','prior','gram','cross'):np.testing.assert_array_equal(ctx[name],z[name])
        with np.load(folder/'reference_original.npz') as z:reference=z['ids']
        ref_records=read_json(folder/'reference_original.json')
        validate_records(reference,ref_records)
        ref_sources={r['document_id'] for r in ref_records}
        assert not ref_sources&set(ctx['source_ids'])
        assert all(not ref_sources&{r['document_id'] for r in v} for v in records.values())
        folds,fold_counts=source_folds(ctx['source_ids'])
        meta=dict(protocol_sha256=protocol,code_sha256=codes,evidence=evidence[corpus],
            readout_sha256=digest(readout_path),source_ids=ctx['source_ids'].tolist(),
            folds=folds.tolist(),fold_chunks=fold_counts.tolist(),
            reference_chunks=len(reference),reference_sources=len(ref_sources),
            original_inputs=original_hashes,reference_sha256=digest(folder/'reference_original.npz'),
            affine_weights={p.name:digest(p) for p in sorted((ROOT/'results/full_context_linear_v1/weights').glob(f'{corpus}_*.npz'))},
            checkpoints=[j for j in jobs if j['corpus']==corpus])
        signature=object_digest(meta)
        path=out/f'{corpus}_inputs.json'
        if path.exists():assert read_json(path)==meta
        else:write_json(path,meta)
        kwargs=dict(prior=ctx['prior'],offsets=offset_list(config['radius']),gram=ctx['gram'],
            cross=ctx['cross'],smoothing=config['smoothing'],ridge=config['ridge'],readout=readout)
        for n in args.sizes:
            progress=lambda done,total:print(json.dumps(dict(stage='source_covariance',corpus=corpus,n=n,
                sources=done,total=total,elapsed=round(time.perf_counter()-started,1))),flush=True)
            path=out/f'{corpus}_n{n}_reference.npz'
            if args.stage in ('all','forecast') and cached(path,signature) is None:
                matrices=forecast_matrices(reference,source_groups(ref_records),ctx['evaluation'],ctx['mask'],
                    folds,n=n,progress=progress,**kwargs)
                mean=ctx['prior']@readout
                unigram=readout.T@(ctx['prior'][:,None]*readout)-np.outer(mean,mean)
                matrices['unigram']=fold_counts[:,None,None]*unigram
                save(path,signature,**matrices)
                print(json.dumps(dict(stage='forecast_saved',corpus=corpus,n=n)),flush=True)
            for pair in (1,2,3):
                path=out/f'{corpus}_n{n}_pair{pair}_empirical.npz'
                if args.stage in ('all','empirical') and cached(path,signature) is None:
                    predictions=[];affines=[]
                    for arm in ('a','b'):
                        train=trains[f'pair{pair}_{arm}'][:n]
                        predictions.append(projected_prediction(ctx['evaluation'],ctx['mask'],train,**kwargs))
                        wp=ROOT/'results/full_context_linear_v1/weights'/f'{corpus}_pair{pair}_{arm}_n{n}.npz'
                        with np.load(wp) as z:weights=z['weights']
                        assert weights.shape==(128,n)
                        # Coefficients are already saved for these exact queries/masks.
                        affines.append(np.tensordot(weights,readout[train],axes=(1,0))[ctx['mask']])
                    lag_delta=predictions[1]-predictions[0];affine_delta=affines[1]-affines[0]
                    save(path,signature,reconstructor=second_moments(lag_delta,ctx['mask'],folds),
                         affine=second_moments(affine_delta,ctx['mask'],folds),
                         lag_delta=lag_delta,affine_delta=affine_delta)
                    print(json.dumps(dict(stage='empirical_saved',corpus=corpus,n=n,pair=pair)),flush=True)
                for updates in (8000,12000):
                    path=out/f'{corpus}_n{n}_pair{pair}_step{updates}_neural.npz'
                    if args.stage in ('all','neural') and cached(path,signature) is None:
                        delta,full=neural_delta(jobs,corpus,n,pair,updates,ctx,config,readout,args.device)
                        save(path,signature,neural=second_moments(delta,ctx['mask'],folds),
                             delta=delta,full_disagreement=full)
                        print(json.dumps(dict(stage='neural_saved',corpus=corpus,n=n,pair=pair,updates=updates,
                            elapsed=round(time.perf_counter()-started,1))),flush=True)
        write_json(out/'progress.json',dict(last_corpus=corpus,stage=args.stage,elapsed=time.perf_counter()-started))
    summarize(out)
    print(json.dumps(dict(stage='complete',elapsed=time.perf_counter()-started)),flush=True)


if __name__=='__main__':main()
