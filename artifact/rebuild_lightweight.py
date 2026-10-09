"""Recompute top-k, occupancy and the reported directional test without tokens."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
from plot_topk import plot_summary

NAMES={'tinystories':'TinyStories','wikitext':'WikiText-103','cnn_dailymail':'CNN/DailyMail'}

def read_csv(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def write_json(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

def topk(root):
    out=root/'05_topk/results/lag_topk_matched_v1'
    models=read_csv(out/'per_model.csv');saved=read_csv(out/'summary.csv');summary=[]
    for row in saved:
        c,n,k=row['corpus'],int(row['n']),int(row['k'])
        with np.load(out/'per_chunk'/f'{c}_evaluation.npz') as z:
            base=float(z['unigram_accuracy_by_chunk'][list(z['k']).index(k)].mean())
        acc=[]
        for pair in (1,2,3):
            for arm in ('a','b'):
                with np.load(out/'per_chunk'/f'{c}_pair{pair}_{arm}_n{n}.npz') as z:
                    val=float(z['accuracy_by_chunk'][list(z['k']).index(k)].mean())
                model=next(r for r in models if (r['corpus'],int(r['n']),int(r['k']),int(r['pair']),r['arm'])==(c,n,k,pair,arm))
                np.testing.assert_allclose(val,float(model['reconstructor_accuracy']),rtol=0,atol=1e-14)
                acc.append(val)
        mean=float(np.mean(acc));pair_means=np.array(acc).reshape(3,2).mean(axis=1)
        calculated=dict(reconstructor_accuracy=mean,unigram_accuracy=base,
            accuracy_gain_pp=100*(mean-base),relative_topk_error_reduction=1-(1-mean)/(1-base),
            pair_mean_accuracy_min=float(pair_means.min()),pair_mean_accuracy_max=float(pair_means.max()))
        for key,value in calculated.items():
            np.testing.assert_allclose(value,float(row[key]),rtol=0,atol=1e-12)
        # Tie-bound columns are historical diagnostics, not inputs to the figure.
        result={key:(value if key=='corpus' else float(value)) for key,value in row.items()}
        result.update(calculated);summary.append(result)
    with (out/'rebuilt_summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    plot_summary(out,summary)
    write_json(out/'rebuild_validation.json',dict(status='passed',conditions=len(summary),
        models=len(models),scope='Exact equal-chunk and equal-model aggregation from per-chunk hit rates for all corpora. No token ranks or inference.'))

def occupancy(root):
    folder=root/'01_original_counts';saved=json.loads((folder/'queried_rows.json').read_text())
    rows=[];recomputed=[]
    with np.load(folder/'queried_count_histograms.npz') as z:
        for record in saved:
            c,n=record['corpus'],record['fitted_chunks']
            counts=z[f'{c}_{n}_counts'];freq=z[f'{c}_{n}_frequency'];total=int(freq.sum())
            values=dict(queried_rows=total,empty_fraction=float(freq[counts==0].sum()/total),
                median_row_total=float(np.median(np.repeat(counts,freq))),
                below_smoothing_fraction=float(freq[counts<5].sum()/total),
                mean_smoothing_share=float(np.sum(freq*5/(counts+5))/total))
            for key,value in values.items():np.testing.assert_allclose(value,record[key],rtol=0,atol=1e-14)
            recomputed.append(dict(corpus=c,n=n,**values))
            rows.append(f"{NAMES[c]} & {n} & {100*values['empty_fraction']:.1f} & {values['median_row_total']:.0f} & {100*values['below_smoothing_fraction']:.1f} & {100*values['mean_smoothing_share']:.1f}")
    write_json(folder/'recomputed.json',recomputed)
    (folder/'tab_occupancy.tex').write_text('\n'.join(rows)+'\n',encoding='utf-8')

def directions(root):
    folder=root/'07_directions/results/mdlm_subspaces_v1'
    expected=json.loads((folder/'expected.json').read_text());rebuilt=[]
    def capture(cov,neural,svd=False):
        total=0.
        for fold in (0,1):
            matrix=cov[1-fold];matrix=(matrix+matrix.T)/2
            if svd:U=np.linalg.svd(matrix,hermitian=True)[0][:,:4]
            else:U=np.linalg.eigh(matrix)[1][:,-4:]
            total+=np.trace(U.T@neural[fold]@U)
        return float(total/np.trace(neural.sum(axis=0)))
    for r in expected:
        c,method=r['corpus'],r['method'];values=[]
        for pair in (1,2,3):
            name=f'{c}_n2048_reference.npz' if method=='unigram' else f'{c}_n2048_pair{pair}_empirical.npz'
            with np.load(folder/name) as z:cov=z[method]
            with np.load(folder/f'{c}_n2048_pair{pair}_step12000_neural.npz') as z:neural=z['neural']
            value=capture(cov,neural)
            np.testing.assert_allclose(value,capture(cov,neural,True),rtol=0,atol=1e-10)
            values.append(value)
        mean=float(np.mean(values));np.testing.assert_allclose(mean,float(r['capture']),rtol=0,atol=1e-12)
        rebuilt.append(dict(corpus=c,method=method,n=2048,updates=12000,dimension=64,
            mode='source_crossfit',k=4,capture=mean,random_expectation=4/64,
            minimum=float(min(values)),maximum=float(max(values))))
    with (folder/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rebuilt[0]));writer.writeheader();writer.writerows(rebuilt)
    write_json(folder/'validation.json',dict(status='passed',rows=len(rebuilt),
        scope='Reported rank-4, dimension-64, n=2048, 12000-update test. Exact covariance blocks. Independent SVD check. Random expectation is k/d, not replayed Monte Carlo draws.'))

if __name__=='__main__':
    root=Path(sys.argv[1]);topk(root);occupancy(root);directions(root)
    print('PASS: top-k, occupancy and reported directional measurements recomputed.')
