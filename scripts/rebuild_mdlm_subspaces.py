"""Rebuild all directional summaries from compact saved covariance matrices."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse
import csv
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from diffusion_lm_rmt.disagreement_subspaces import capture, random_capture


def write(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def build(out):
    rows=[];ks=(1,2,4,8,16)
    for corpus in ('cnn_dailymail','tinystories','wikitext'):
        for n in (512,1024,2048):
            with np.load(out/f'{corpus}_n{n}_reference.npz') as z:
                forecasts={m:z[m] for m in ('source_full','source_same_lag','multinomial','unigram')}
            for pair in (1,2,3):
                with np.load(out/f'{corpus}_n{n}_pair{pair}_empirical.npz') as z:
                    empirical={m:z[m] for m in ('reconstructor','affine')}
                for updates in (8000,12000):
                    with np.load(out/f'{corpus}_n{n}_pair{pair}_step{updates}_neural.npz') as z:neural=z['neural']
                    for d in (64,128):
                        target=neural[:,:d,:d]
                        for mode in ('source_crossfit','pooled'):
                            null=random_capture(target,crossfit=mode=='source_crossfit')
                            for method,matrix in {**forecasts,**empirical}.items():
                                source=matrix[:,:d,:d]
                                for j,k in enumerate(ks):
                                    rows.append(dict(corpus=corpus,n=n,pair=pair,updates=updates,
                                        dimension=d,mode=mode,method=method,k=k,
                                        capture=capture(source,target,k,crossfit=mode=='source_crossfit'),
                                        random_expectation=k/d,random_mean=float(null[:,j].mean()),
                                        random_p95=float(np.quantile(null[:,j],.95)),
                                        neural_trace=float(np.trace(target.sum(0))/128),
                                        method_trace=float(np.trace(source.sum(0))/128)))
    write(out/'per_pair.csv',rows)
    keys=('corpus','n','updates','dimension','mode','method','k');grouped={}
    for r in rows:grouped.setdefault(tuple(r[k] for k in keys),[]).append(r)
    summary=[]
    for key,cell in sorted(grouped.items()):
        assert len(cell)==3
        v=[r['capture'] for r in cell]
        summary.append(dict(zip(keys,key))|dict(pairs=3,capture=float(np.mean(v)),minimum=min(v),maximum=max(v),
            random_expectation=cell[0]['random_expectation'],random_p95_mean=float(np.mean([r['random_p95'] for r in cell]))))
    write(out/'summary.csv',summary)
    print(f'Rebuilt {len(rows)} pair rows and {len(summary)} summary rows.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/mdlm_subspaces_v1')
    build(p.parse_args().output)
