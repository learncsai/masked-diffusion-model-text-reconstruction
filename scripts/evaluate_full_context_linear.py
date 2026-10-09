"""Run the frozen full-context linear comparison against saved MDLM checkpoints."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
import torch
from scipy.stats import spearmanr
from diffusion_lm_rmt.full_context_linear import FullContextRidge, scores_from_weights, prediction_metrics
from diffusion_lm_rmt.matched_mdlm import digest, object_digest, read_json, write_json, source_groups, load_model, utc
from evaluate_mdlm_topk import CORPORA, prepare_jobs
from evaluate_lag_topk import write_csv

GRID = (1e-5, 1e-4, 1e-3, .01, .1, 1.)
SIZES = (512, 1024, 2048)
KS = (1, 2, 4, 8, 16, 32, 64)
RUN = ROOT/'results/mdlm_runpod_main_v1'


def csv_rows(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_data(corpus, config, context):
    folder = RUN/corpus
    manifest = read_json(folder/'manifest.json')
    inputs = manifest['protocol']['inputs']
    needed = [p for p in inputs if p.endswith(('replicate3_train_a.pt', '/eval.pt')) or
              (f'/{corpus}/pair' in p and p.endswith(('.npz', '.json')))]
    for p in needed:
        if digest(ROOT/p) != inputs[p]:
            raise ValueError(f'Changed training/evaluation input {p}')
    auxiliary = torch.load(ROOT/next(p for p in needed if p.endswith('replicate3_train_a.pt')),
                           map_location='cpu', weights_only=True)
    groups = source_groups(auxiliary['chunk_records'])
    np.random.default_rng(config['seed']).shuffle(groups)
    train_ix, val_ix = np.concatenate(groups[:len(groups)//2]), np.concatenate(groups[len(groups)//2:])
    aux_train = auxiliary['input_ids'].numpy()[train_ix]
    aux_val = auxiliary['input_ids'].numpy()[val_ix][:128]
    np.testing.assert_array_equal(aux_val, context['validation'])
    audit = read_json(folder/'source_audit.json')['roles']
    aux_sources = {auxiliary['chunk_records'][i]['document_id'] for i in train_ix}
    assert aux_sources == set(audit['auxiliary_train']['source_ids'])
    val_sources = {auxiliary['chunk_records'][i]['document_id'] for i in val_ix}
    assert val_sources == set(audit['auxiliary_validation']['source_ids'])
    source_sets = [aux_sources, val_sources, set(context['source_ids'])]
    trains, records = {}, {}
    for pair in (1, 2, 3):
        for arm in ('a', 'b'):
            name = f'pair{pair}_{arm}'
            path = next(ROOT/p for p in needed if p.endswith(f'/{name}.npz'))
            with np.load(path) as z:
                trains[name] = z['ids'].copy()
            records[name] = read_json(path.with_suffix('.json'))
            sources = {r['document_id'] for r in records[name]}
            assert sources == set(audit[name]['source_ids'])
            assert all(not (sources & other) for other in source_sets)
            source_sets.append(sources)
            for n in SIZES:
                complete = read_json(folder/'models'/f'{name}_n{n}'/'complete.json')
                data_hash = hashlib.sha256(np.asarray(trains[name][:n], dtype=np.int32).tobytes()).hexdigest()
                assert data_hash == complete['identity']['data_sha256']
    return aux_train, trains, records, {p: inputs[p] for p in needed}


def select_ridge(aux, context, out, corpus, device, signature):
    path = out/f'{corpus}_selection.json'
    if path.exists():
        saved = read_json(path)
        assert saved['signature'] == signature
        return saved['selected_ridge']
    model = FullContextRidge(aux, device=device)
    losses = {ridge: [] for ridge in GRID}
    for row, (query, mask) in enumerate(zip(context['validation'], context['validation_mask'])):
        positions = np.flatnonzero(mask)
        for ridge in GRID:
            w, _ = model.weights(query, mask, ridge)
            score = scores_from_weights(aux, w, positions, len(context['prior']))
            truth = score[np.arange(len(positions)), query[positions]]
            losses[ridge].append(float(np.mean(1-2*truth+np.square(score).sum(1))))
        if (row+1) % 32 == 0:
            print(json.dumps(dict(stage='auxiliary_selection', corpus=corpus, chunks=row+1)), flush=True)
    selected = min(GRID, key=lambda ridge: (np.mean(losses[ridge]), -ridge))
    write_json(path, dict(signature=signature, selected_ridge=selected,
        boundary_selection=selected in (GRID[0], GRID[-1]), auxiliary_training_chunks=len(aux),
        validation_chunks=128, scores=[dict(ridge=r, mse=float(np.mean(losses[r])), per_chunk=losses[r]) for r in GRID],
        maximum_solver_residual=model.maximum_relative_residual,
        selection_uses='auxiliary validation MSE only', frozen_utc=utc()))
    print(json.dumps(dict(stage='selected', corpus=corpus, ridge=selected)), flush=True)
    return selected


def fit_arm(train, context, ridge, path, device, signature):
    if path.exists():
        with np.load(path) as z:
            assert str(z['signature']) == signature
            return {k: z[k].copy() for k in z.files}
    model = FullContextRidge(train, device=device)
    weights, metrics, diagnostics = [], [], []
    for i, (query, mask) in enumerate(zip(context['evaluation'], context['mask'])):
        w, check = model.weights(query, mask, ridge)
        pos = np.flatnonzero(mask)
        score = scores_from_weights(train, w, pos, len(context['prior']))
        weights.append(w)
        metrics.append(prediction_metrics(score, query[pos]))
        diagnostics.append(check['relative_residual'])
    result = dict(weights=np.asarray(weights), source_ids=context['source_ids'],
                  signature=np.array(signature), ridge=np.array(ridge),
                  solver_residual=np.array(diagnostics),
                  **{key: np.array([m[key] for m in metrics]) for key in metrics[0]})
    np.savez_compressed(path, **result)
    return result


@torch.inference_mode()
def neural_comparison(corpus, pair, n, updates, train_a, train_b, wa, wb,
                      context, config, device, jobs):
    selected = [j for j in jobs if (j['corpus'],j['pair'],j['n'],j['updates']) == (corpus,pair,n,updates)]
    selected.sort(key=lambda j:j['arm'])
    assert [j['arm'] for j in selected] == ['a','b']
    models = [load_model(config, Path(j['checkpoint']), device).eval() for j in selected]
    ids, masks = context['evaluation'], context['mask']
    result = {key: [] for key in ('neural_d','linear_d','inner','residual','neural_mse_a','neural_mse_b')}
    classes = len(context['prior'])
    batch = 4
    for start in range(0,len(ids),batch):
        clean = torch.as_tensor(ids[start:start+batch], dtype=torch.long, device=device)
        hidden = torch.as_tensor(masks[start:start+batch], dtype=torch.bool, device=device)
        corrupt = clean.masked_fill(hidden, config['mask_token_id'])
        attention = torch.ones_like(clean, dtype=torch.bool)
        times = torch.full((len(clean),), config['mask_rate'], device=device)
        probabilities = [m(corrupt,attention,times)[...,:classes][hidden].float().softmax(-1).cpu().numpy()
                         for m in models]
        offset=0
        for local in range(len(clean)):
            row=start+local
            pos=np.flatnonzero(masks[row])
            pa,pb=[v[offset:offset+len(pos)].astype(np.float64) for v in probabilities]
            offset += len(pos)
            la=scores_from_weights(train_a,wa[row],pos,classes)
            lb=scores_from_weights(train_b,wb[row],pos,classes)
            neural,linear=pb-pa,lb-la
            target=ids[row,pos]
            vals=dict(neural_d=np.square(neural).sum(1).mean(),linear_d=np.square(linear).sum(1).mean(),
                inner=(neural*linear).sum(1).mean(),residual=np.square(neural-linear).sum(1).mean(),
                neural_mse_a=(1-2*pa[np.arange(len(pos)),target]+np.square(pa).sum(1)).mean(),
                neural_mse_b=(1-2*pb[np.arange(len(pos)),target]+np.square(pb).sum(1)).mean())
            for key,value in vals.items(): result[key].append(value)
    result={key:np.asarray(value) for key,value in result.items()}
    saved=read_json(RUN/corpus/'scores'/f'pair{pair}_n{n}_step{updates}.json')['per_chunk']
    for key,old in [('neural_d','disagreement'),('neural_mse_a','loss_a'),('neural_mse_b','loss_b')]:
        np.testing.assert_allclose(result[key],saved[old],rtol=2e-5,atol=2e-7)
    return result


def summarize(out, contexts):
    rows, chunk_rows, tops = [], [], []
    old=csv_rows(ROOT/'results/mdlm_paper_comparison_v1/per_chunk.csv')
    lookup={(r['corpus'],int(r['n']),int(r['pair']),int(r['updates']),int(r['chunk'])):r for r in old}
    for corpus,(_,context) in contexts.items():
        baseline=context['baseline_mse']
        for n in SIZES:
            for pair in (1,2,3):
                armfiles=[out/'weights'/f'{corpus}_pair{pair}_{arm}_n{n}.npz' for arm in ('a','b')]
                if not all(p.exists() for p in armfiles): continue
                with np.load(armfiles[0]) as a,np.load(armfiles[1]) as b:
                    mse=.5*(a['mse']+b['mse'])
                    for k in KS:
                        tops.append(dict(corpus=corpus,n=n,pair=pair,k=k,
                            accuracy=float(.5*(a[f'accuracy_{k}'].mean()+b[f'accuracy_{k}'].mean()))))
                    ridge=float(a['ridge'])
                for updates in (8000,12000):
                    path=out/'comparisons'/f'{corpus}_pair{pair}_n{n}_step{updates}.npz'
                    if not path.exists():continue
                    with np.load(path) as z:
                        scores={k:z[k].copy() for k in ('neural_d','linear_d','inner','residual')}
                    dlin,dnet,inner,resid=[float(scores[k].mean()) for k in ('linear_d','neural_d','inner','residual')]
                    lag=np.array([float(lookup[(corpus,n,pair,updates,i)]['lag_disagreement']) for i in range(128)])
                    r=dict(corpus=corpus,n=n,pair=pair,updates=updates,ridge=ridge,
                        linear_disagreement=dlin,neural_disagreement=dnet,ratio=dlin/dnet,
                        mse=float(mse.mean()),relative_mse_improvement=float(1-mse.mean()/baseline),
                        spearman=float(spearmanr(scores['linear_d'],scores['neural_d']).statistic),
                        lag_spearman=float(spearmanr(lag,scores['neural_d']).statistic),
                        cosine=inner/np.sqrt(dlin*dnet),relative_vector_error=np.sqrt(resid/dnet),
                        inner=inner,residual=resid)
                    rows.append(r)
                    for i in range(128):
                        previous=lookup[(corpus,n,pair,updates,i)]
                        chunk_rows.append(dict(corpus=corpus,n=n,pair=pair,updates=updates,chunk=i,
                            source_id=context['source_ids'][i],linear_mse=float(mse[i]),
                            **{k:float(v[i]) for k,v in scores.items()},
                            lag_d=float(previous['lag_disagreement']),source_forecast=float(previous['source_full']),
                            multinomial_forecast=float(previous['multinomial'])))
    if not rows:return
    summary=[]
    for corpus,n,updates in sorted({(r['corpus'],r['n'],r['updates']) for r in rows}):
        cell=[r for r in rows if (r['corpus'],r['n'],r['updates'])==(corpus,n,updates)]
        numeric=[key for key in cell[0] if key not in ('corpus','n','pair','updates')]
        s=dict(corpus=corpus,n=n,updates=updates,pairs=len(cell),
               **{key:float(np.mean([r[key] for r in cell])) for key in numeric})
        s.update(ratio=s['linear_disagreement']/s['neural_disagreement'],
                 cosine=s['inner']/np.sqrt(s['linear_disagreement']*s['neural_disagreement']),
                 relative_vector_error=np.sqrt(s['residual']/s['neural_disagreement']))
        for key in ('linear_disagreement','spearman','cosine','relative_mse_improvement'):
            s['minimum_'+key]=min(r[key] for r in cell)
            s['maximum_'+key]=max(r[key] for r in cell)
        summary.append(s)
    write_csv(out/'per_pair.csv',rows)
    write_csv(out/'per_chunk.csv',chunk_rows)
    write_csv(out/'summary.csv',summary)
    write_csv(out/'topk_per_pair.csv',tops)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/full_context_linear_v1')
    p.add_argument('--device',default='cuda')
    p.add_argument('--corpora',nargs='+',default=list(CORPORA),choices=CORPORA)
    p.add_argument('--stage',choices=('all','fit','compare','summarize'),default='all')
    args=p.parse_args()
    out=args.output
    for folder in (out,out/'weights',out/'comparisons'):folder.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    jobs,contexts,evidence=prepare_jobs(RUN,ROOT/'results/lag_topk_matched_v1',args.corpora)
    if args.stage=='summarize':summarize(out,contexts);return
    sources={}
    for corpus in args.corpora:
        sources[corpus]=dict(manifest=digest(RUN/corpus/'manifest.json'),context=digest(RUN/corpus/'context.npz'))
    identity=dict(protocol_sha256=digest(ROOT/'docs/full_context_linear_protocol.txt'),sources=sources,
                  estimator_sha256=digest(ROOT/'src/diffusion_lm_rmt/full_context_linear.py'),
                  ridge_grid=GRID,corpora=args.corpora)
    signature=object_digest(identity)
    meta=out/'manifest.json'
    if meta.exists():assert read_json(meta)['signature']==signature
    else:write_json(meta,dict(signature=signature,identity=identity,created_utc=utc(),
                             python=sys.version,torch=torch.__version__,numpy=np.__version__,device=args.device))
    start=time.perf_counter()
    completed=0
    validation=[]
    for corpus in args.corpora:
        config,context=contexts[corpus]
        aux,trains,records,inputs=load_data(corpus,config,context)
        write_json(out/f'{corpus}_inputs.json',dict(inputs=inputs,evidence=evidence[corpus],
            evaluation_source_ids=context['source_ids'].tolist(),
            training_sources={key:[r['document_id'] for r in rs] for key,rs in records.items()}))
        ridge=select_ridge(aux,context,out,corpus,args.device,signature)
        for n in SIZES:
            for pair in (1,2,3):
                fitted=[]
                for arm in ('a','b'):
                    path=out/'weights'/f'{corpus}_pair{pair}_{arm}_n{n}.npz'
                    fitted.append(fit_arm(trains[f'pair{pair}_{arm}'][:n],context,ridge,path,args.device,signature))
                    completed+=1
                    state=dict(status='running',stage='linear_fit',corpus=corpus,n=n,pair=pair,arm=arm,
                               completed_models=completed,total_models=len(args.corpora)*18,
                               elapsed_seconds=time.perf_counter()-start)
                    write_json(out/'progress.json',state);print(json.dumps(state),flush=True)
                torch.cuda.empty_cache() if args.device.startswith('cuda') else None
                if args.stage=='fit':continue
                for updates in (8000,12000):
                    path=out/'comparisons'/f'{corpus}_pair{pair}_n{n}_step{updates}.npz'
                    if path.exists():
                        with np.load(path) as z:assert str(z['signature'])==signature
                        continue
                    scores=neural_comparison(corpus,pair,n,updates,
                        trains[f'pair{pair}_a'][:n],trains[f'pair{pair}_b'][:n],
                        fitted[0]['weights'],fitted[1]['weights'],context,config,args.device,jobs)
                    np.savez_compressed(path,signature=np.array(signature),source_ids=context['source_ids'],**scores)
                    print(json.dumps(dict(stage='neural_comparison',corpus=corpus,n=n,pair=pair,updates=updates)),flush=True)
                summarize(out,contexts)
    write_json(out/'progress.json',dict(status='fit_complete' if args.stage=='fit' else 'complete',
        completed_models=completed,total_models=len(args.corpora)*18,elapsed_seconds=time.perf_counter()-start))
    summarize(out,contexts)


if __name__=='__main__':main()
