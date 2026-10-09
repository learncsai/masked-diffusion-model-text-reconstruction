"""Recompute all follow-up summaries from saved scalar forecasts and pair scores."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CORPORA=['tinystories','wikitext','cnn_dailymail']
SIZES=[512,1024,2048,4096];MS=[1024,2048,4096]
CORE=['source_full','source_jackknife','multinomial','multinomial_jackknife']

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p,x): p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def csvwrite(p,rows):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def infer(arrays,cells,seed=810191,draws=5000):
    estimates=[];sim=[]
    rng=np.random.default_rng(seed)
    for f,o in arrays:
        B,P=len(f),len(o)
        bi=rng.integers(B,size=(draws,B));pi=rng.integers(P,size=(draws,P))
        obs=o[pi].mean(axis=1)
        estimates.append(np.mean([np.abs(np.log(f[:,mi,ni]/o[:,ni].mean())).mean(axis=0) for mi,ni in cells],axis=0))
        sim.append(np.mean([np.abs(np.log(f[bi,mi,ni]/obs[:,None,ni,None])).mean(axis=1) for mi,ni in cells],axis=0))
    est=np.mean(estimates,axis=0);distribution=np.mean(sim,axis=0)
    return dict(error=est.tolist(),interval=np.quantile(distribution,[.025,.975],axis=0).T.tolist(),
                corrected_multinomial_minus_plain=float(est[3]-est[2]),
                corrected_multinomial_difference_interval=np.quantile(distribution[:,3]-distribution[:,2],[.025,.975]).tolist())

def existing(root,out,core_only=False):
    cfg=read(root/'configs/correction_followup_v1.json');methods=CORE.copy()
    if not core_only: methods+=['source_bootstrap','bootstrap_jackknife','direct_inverse_size','direct_constant']
    methods+=['source_times_1.25','multinomial_times_1.25','source_gated','multinomial_gated']
    arrays=[];decisions=[];mc=[];cellrows=[]
    for ci,c in enumerate(CORPORA):
        old=root/cfg['existing_study']/c
        with np.load(old/'analysis_arrays.npz') as z:
            f=z['forecasts'];obs=z['observed']
        corr=np.load(old/'multinomial_jackknife_exploratory.npy')
        f=np.concatenate([f,corr[...,None]],axis=3)
        if not core_only:
            extra=np.zeros((6,3,4,4))
            for b in range(6):
                for mi,m in enumerate(MS):
                    direct=np.load(out/'baselines'/c/f'direct_bank{b}_m{m}.npz')
                    for ni,n in enumerate(SIZES):
                        z=np.load(out/'baselines'/c/f'bootstrap_bank{b}_m{m}_n{n}.npz')
                        assert float(z['raw_jackknife'])>0,'Nonpositive corrected bootstrap scalar'
                        extra[b,mi,ni,:2]=z['forecasts']
                        extra[b,mi,ni,2:]=[direct['per_chunk'].mean()*float(direct['harmonic_half_size'])/n,direct['per_chunk'].mean()]
                        for j,batch in enumerate([z['batches'][0],z['jackknife_batches']]):
                            mc.append(dict(corpus=c,bank=b,m=m,n=n,method=methods[4+j],
                                           forecast=float(batch.mean()),mc_se=float(batch.std(ddof=1)/np.sqrt(len(batch))),
                                           relative_mc_se=float(batch.std(ddof=1)/np.sqrt(len(batch))/batch.mean())))
            f=np.concatenate([f,extra],axis=3)
        gate=np.array([[m<=n for n in SIZES] for m in MS])
        f=np.concatenate([f,(1.25*f[...,0])[...,None],(1.25*f[...,2])[...,None],
            np.where(gate[None],f[...,1],f[...,0])[...,None],
            np.where(gate[None],f[...,3],f[...,2])[...,None]],axis=3)
        assert (f>0).all();arrays.append((f,obs))
        np.savez_compressed(out/f'{c}_followup_arrays.npz',forecasts=f,observed=obs,methods=methods)
        for mi,m in enumerate(MS):
            for ni,n in enumerate(SIZES):
                for j,method in enumerate(methods):
                    cellrows.append(dict(corpus=c,m=m,n=n,method=method,
                        ratio=float(f[:,mi,ni,j].mean()/obs[:,ni].mean()),
                        error=float(np.abs(np.log(f[:,mi,ni,j]/obs[:,ni].mean())).mean())))
        rng=np.random.default_rng(20261003+90000+ci)
        interval=np.quantile(obs[rng.integers(20,size=(5000,20))].mean(axis=1),[.025,.975],axis=0).T
        original=read(old/'target_saved.json')['target']
        for label,target in [('original',original),*[(str(t),t) for t in cfg['absolute_decision_targets']]]:
            feasible=np.flatnonzero(interval[:,1]<=target)
            gridmin=int(SIZES[feasible[0]]) if len(feasible) else None
            for b in range(6):
                for mi,m in enumerate(MS):
                    for j,method in enumerate(methods):
                        eligible=np.flatnonzero(f[b,mi,:,j]<=target)
                        index=int(eligible[0]) if len(eligible) else None
                        n=None if index is None else SIZES[index]
                        status='abstain' if index is None else ('success' if interval[index,1]<=target else
                            ('failure' if obs[:,index].mean()>target else 'uncertain'))
                        decisions.append(dict(corpus=c,bank=b,m=m,method=method,target_name=label,target=target,
                            selected_n=n,observed_grid_min=gridmin,status=status,
                            excess_chunks=None if n is None or gridmin is None else n-gridmin))
    cells=[(i,j) for i in range(3) for j in range(4)]
    result=dict(status='Exploratory additions to already observed twenty-pair study',methods=methods,
        primary=infer(arrays,[(0,2),(0,3),(1,3)]),full_grid=infer(arrays,cells),
        m_le_n=infer(arrays,[(i,j) for i,j in cells if MS[i]<=SIZES[j]]),
        m_gt_n=infer(arrays,[(i,j) for i,j in cells if MS[i]>SIZES[j]]),
        m_eq_n=infer(arrays,[(i,j) for i,j in cells if MS[i]==SIZES[j]]))
    counts=[]
    for target in ['original',*[str(t) for t in cfg['absolute_decision_targets']]]:
        for method in methods:
            records=[r for r in decisions if r['target_name']==target and r['method']==method]
            counts.append(dict(target=target,method=method,
                **{s:sum(r['status']==s for r in records) for s in ['success','failure','uncertain','abstain']},
                successes_at_grid_min=sum(r['status']=='success' and r['excess_chunks']==0 for r in records),
                successes_above_grid_min=sum(r['status']=='success' and r['excess_chunks']>0 for r in records)))
    result['decisions']=counts
    write(out/('core_summary.json' if core_only else 'summary.json'),result)
    csvwrite(out/'calibration.csv',cellrows);csvwrite(out/'decisions.csv',decisions)
    if mc: csvwrite(out/'bootstrap_mc.csv',mc)
    print(json.dumps({k:result[k] for k in ['methods','primary','full_grid','m_gt_n']},indent=2))

def confirmation(root,out):
    cfg=read(root/'configs/correction_followup_v1.json');d=cfg['confirmatory'];arrays=[];ratios=[]
    freeze=read(out/'confirmation_forecasts_saved.json')
    for c in CORPORA:
        folder=out/'confirmation'/c
        f=np.load(folder/'forecasts.npz');o=np.load(folder/'observed.npz')
        assert str(f['saved_utc'])<freeze['saved_utc']<str(o['saved_utc'])
        arrays.append((f['forecasts'],o['observed']))
        ratios.append(dict(corpus=c,forecast_mean=f['forecasts'][:,0,1].mean(axis=0).tolist(),
                           observed_mean=float(o['observed'][:,1].mean()),
                           ratios=(f['forecasts'][:,0,1].mean(axis=0)/o['observed'][:,1].mean()).tolist()))
    result=dict(status='Prospective fresh-source confirmation',methods=CORE,
        primary=infer(arrays,[(0,1)],seed=cfg['seed']+199),
        full_grid=infer(arrays,[(i,j) for i in range(2) for j in range(2)],seed=cfg['seed']+199),
        ratios=ratios,all_forecasts_precede_all_outcomes=True,
        primary_method=d['primary_method'],primary_comparator=d['primary_comparator'])
    write(out/'confirmation_summary.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--core-only',action='store_true');ap.add_argument('--confirmation',action='store_true')
    a=ap.parse_args();cfg=read(a.root/'configs/correction_followup_v1.json');out=a.root/cfg['output']
    if a.confirmation: confirmation(a.root,out)
    else: existing(a.root,out,a.core_only)
