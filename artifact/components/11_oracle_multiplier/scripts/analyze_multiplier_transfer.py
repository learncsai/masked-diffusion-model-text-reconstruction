"""Retrospective, outcome-separated calibration-factor transfer checks.

Training factors use only the explicitly named training cells/corpora.
Bootstrap draws resample entire banks and pairs, retaining dependence of
nested size measurements. Factors are reselected in every bootstrap draw.
"""
from pathlib import Path
import argparse
import csv
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CORPORA=['tinystories','wikitext','cnn_dailymail']
CORE=['source_full','source_jackknife','multinomial','multinomial_jackknife']


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def logs(arrays,rng,draws):
    point=[];boot=[]
    for f,o in arrays:
        point.append(np.log(f/o.mean(axis=0)[None,None,:,None]))
        bi=rng.integers(len(f),size=(draws,len(f)))
        pi=rng.integers(len(o),size=(draws,len(o)))
        boot.append(np.log(f[bi]/o[pi].mean(axis=1)[:,None,None,:,None]))
    return np.asarray(point),np.stack(boot,axis=1)


def take_cells(a,cells):
    # Cell and method axes are always the final three dimensions.
    return np.stack([a[...,i,j,:] for i,j in cells],axis=-2)


def train_factor(a,cells,corpora=None):
    # Returns log factor. Input: [draws,] corpus,bank,m,n,method.
    if corpora is not None:a=np.take(a,corpora,axis=-5)
    value=take_cells(a,cells)[...,2]
    return -np.median(value.reshape(*value.shape[:-3],-1),axis=-1)


def score(a,cells,logfactor=None,corpora=None):
    if corpora is not None:a=np.take(a,corpora,axis=-5)
    value=take_cells(a,cells)
    if logfactor is not None:
        value=value[...,2:3]+np.asarray(logfactor)[...,None,None,None,None]
    return np.mean(np.abs(value),axis=(-4,-3,-2))


def pack(estimate,distribution):
    return dict(error=np.asarray(estimate).tolist(),
                interval=np.quantile(distribution,[.025,.975],axis=0).T.tolist())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT);a=ap.parse_args()
    root=a.root;config=read(root/'configs/oracle_multiplier_v1.json')
    out=root/config['output'];out.mkdir(parents=True,exist_ok=True)
    original=[];fresh=[]
    for c in CORPORA:
        with np.load(root/f'results/correction_followup_v1/{c}_followup_arrays.npz') as z:
            names=z['methods'].tolist()
            original.append((z['forecasts'][...,[names.index(m) for m in CORE]],z['observed']))
        with np.load(root/f'results/correction_followup_v1/confirmation/{c}/forecasts.npz') as z:f=z['forecasts']
        with np.load(root/f'results/correction_followup_v1/confirmation/{c}/observed.npz') as z:o=z['observed']
        fresh.append((f,o))
    rng=np.random.default_rng(config['seed'])
    op,ob=logs(original,rng,config['bootstrap_draws'])
    fp,fb=logs(fresh,rng,config['bootstrap_draws'])
    full=[(i,j) for i in range(3) for j in range(4)]
    primary=[(0,2),(0,3),(1,3)]
    freshfull=[(i,j) for i in range(2) for j in range(2)]
    freshprimary=[(0,1)]
    scale=train_factor(op,full);bscale=train_factor(ob,full)
    corpus_scales=[train_factor(op,full,[j for j in range(3) if j!=i]) for i in range(3)]
    corpus_boot=[train_factor(ob,full,[j for j in range(3) if j!=i]) for i in range(3)]
    summary=dict(status=config['status'],global_factor=float(np.exp(scale)),
                 heldout_corpus_factors=dict(zip(CORPORA,np.exp(corpus_scales).tolist())))
    names=CORE+['multinomial_times_1.25','multinomial_calibrated_original','multinomial_calibrated_other_corpora']
    summary['methods']=names
    for label,cells in [('primary',freshprimary),('full_grid',freshfull)]:
        estimates=[score(fp,cells),score(fp,cells,np.log(config['fixed_multiplier'])),score(fp,cells,scale),
            np.mean([score(fp,cells,corpus_scales[i],[i]) for i in range(3)],axis=0)]
        distributions=[score(fb,cells),score(fb,cells,np.log(config['fixed_multiplier'])),score(fb,cells,bscale),
            np.mean([score(fb,cells,corpus_boot[i],[i]) for i in range(3)],axis=0)]
        estimate=np.concatenate(estimates);distribution=np.concatenate(distributions,axis=1)
        summary[label]=pack(estimate,distribution)
        summary[label]['paired_minus_jackknife']=pack(estimate-estimate[3],distribution-distribution[:,3,None])
    pscale=train_factor(op,primary);pbs=train_factor(ob,primary)
    summary['original_primary_to_fresh_primary']={
        'factor':float(np.exp(pscale)),**pack(score(fp,freshprimary,pscale),score(fb,freshprimary,pbs))}
    transfers=[];folds=[]
    for name,groups in [('target_size',[(str(n),[(i,j) for i in range(3)]) for j,n in enumerate([512,1024,2048,4096])]),
        ('size_ratio',[(str(r),[(i,j) for i,m in enumerate([1024,2048,4096]) for j,n in enumerate([512,1024,2048,4096]) if m/n==r])
                       for r in [.25,.5,1,2,4,8]])]:
        all_est=[];all_boot=[]
        for held,cells in groups:
            train=[x for x in full if x not in cells]
            lf=train_factor(op,train);blf=train_factor(ob,train)
            est=np.concatenate([score(op,cells)[[2,3]],score(op,cells,np.log(1.25)),score(op,cells,lf)])
            dist=np.concatenate([score(ob,cells)[:,[2,3]],score(ob,cells,np.log(1.25)),score(ob,cells,blf)],axis=1)
            all_est.append(est*len(cells)/len(full));all_boot.append(dist*len(cells)/len(full))
            folds.append(dict(protocol=name,held_out=held,factor=float(np.exp(lf)),cells=cells,**pack(est,dist)))
        est=sum(all_est);dist=sum(all_boot)
        transfers.append(dict(protocol=name,methods=['multinomial','jackknife','fixed_1.25','calibrated'],
                              **pack(est,dist),paired_calibrated_minus_jackknife=pack(est[-1]-est[1],dist[:,-1]-dist[:,1])))
    summary['within_original_transfers']=transfers;summary['folds']=folds
    # Cell-specific optimum and the full prespecified multiplier sweep.
    cells=[];sweep=[]
    factors=np.linspace(*config['multiplier_grid'][:2],config['multiplier_grid'][2])
    for ci,c in enumerate(CORPORA):
        for mi,m in enumerate([1024,2048,4096]):
            for ni,n in enumerate([512,1024,2048,4096]):
                lr=op[ci,:,mi,ni,2];blr=ob[:,ci,:,mi,ni,2]
                optimal=np.exp(-np.median(lr))
                intervals=np.quantile(np.exp(-np.median(blr,axis=1)),[.025,.975])
                cells.append(dict(corpus=c,m=m,n=n,ratio=m/n,factor=float(optimal),low=float(intervals[0]),high=float(intervals[1])))
                for factor in factors:
                    sweep.append(dict(corpus=c,m=m,n=n,factor=float(factor),error=float(abs(lr+np.log(factor)).mean())))
    for filename,values in [('multiplier_cells.csv',cells),('multiplier_sweep.csv',sweep)]:
        with (out/filename).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    # Reconcile the original implementation and the independent aggregation.
    old=read(root/'results/correction_followup_v1/confirmation_summary.json')
    for key in ('primary','full_grid'):
        np.testing.assert_allclose(summary[key]['error'][:4],old[key]['error'],rtol=1e-12)
    (out/'multiplier_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['global_factor','heldout_corpus_factors','primary','full_grid','within_original_transfers']},indent=2))


if __name__=='__main__':main()
