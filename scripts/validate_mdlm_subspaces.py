"""Independently rebuild capture fractions and verify saved directional matrices."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))


def direct_direction_check(corpus, matrices, projection):
    """Independent real-data scalar variance along one fixed vocabulary vector.

    Center each source explicitly, without the runner's outer-product
    expansion or its covariance implementation.
    """
    from diffusion_lm_rmt import sparse_categorical_lag as lag
    from diffusion_lm_rmt.categorical_lag import source_groups, offset_list, pattern_weights
    folder=ROOT/'results/lag_disagreement_matched_v1/inputs'/corpus
    with np.load(folder/'context.npz') as z:
        ids,mask,prior,gram,cross=[z[k] for k in ('evaluation','mask','prior','gram','cross')]
    with np.load(folder/'reference_original.npz') as z:reference=z['ids']
    groups=source_groups(read(folder/'reference_original.json'))
    cfg=read(folder/'original_manifest.json')['protocol']['config'];offsets=offset_list(cfg['radius'])
    folds=np.array(read(ROOT/f'results/mdlm_subspaces_v1/{corpus}_inputs.json')['folds'])
    rows,_,contexts,visible,patterns=lag.context_design(ids,mask,offsets)
    w=np.zeros((len(rows),len(offsets)))
    for code in np.unique(patterns):
        active=tuple(j for j in range(len(offsets)) if code&(1<<j))
        if active:w[np.ix_(np.flatnonzero(patterns==code),active)]=pattern_weights(gram,cross,active,cfg['ridge'])
    v=np.random.default_rng(4711).standard_normal(64);v/=np.linalg.norm(v)
    direction=projection[:,:64]@v;alpha=cfg['smoothing'];n=2048;scale=n/len(reference)
    counts=lag.pair_counts(reference,offsets,len(prior));mu=prior@direction;second=prior@(direction**2)
    lookup=[];multi=np.zeros(len(rows))
    for j,d in enumerate(offsets):
        total=np.asarray(counts[d].sum(1)).ravel();z=counts[d]@direction
        denom=scale*total+alpha;P=(scale*z+alpha*mu)/denom
        pref=(z+alpha*mu)/(total+alpha)
        psecond=(counts[d]@(direction**2)+alpha*second)/(total+alpha)
        ix=np.flatnonzero(visible[j]);a=contexts[j][ix]
        multi[ix]+=w[ix,j]**2*2*scale*total[a]/denom[a]**2*(psecond[a]-pref[a]**2)
        lookup.append((total,z,denom,P,ix,a))
    full=np.zeros(len(rows));same=np.zeros(len(rows))
    for group in groups:
        gc=lag.pair_counts(reference[group],offsets,len(prior));parts=[]
        for j,d in enumerate(offsets):
            total,z,denom,P,ix,a=lookup[j]
            gtotal=np.asarray(gc[d].sum(1)).ravel();gz=gc[d]@direction
            r=gz-len(group)/len(reference)*z
            rt=gtotal-len(group)/len(reference)*total
            u=np.zeros(len(rows));u[ix]=w[ix,j]*(r[a]-rt[a]*P[a])/denom[a]
            parts.append(u);same+=u*u
        full+=sum(parts)**2
    factor=2*(n*len(groups)/len(reference))/(len(groups)-1)
    errors={}
    for method,point in [('source_full',factor*full),('source_same_lag',factor*same),('multinomial',multi)]:
        expected=np.array([np.sum(point[folds[rows]==f]/mask.sum(1)[rows[folds[rows]==f]]) for f in (0,1)])
        actual=np.einsum('i,fij,j->f',v,matrices[method][:,:64,:64],v)
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-13)
        errors[method]=float(np.max(np.abs(actual-expected)))
    return errors


def validate(out,workspace=True):
    with (out/'per_pair.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    with (out/'summary.csv').open(newline='') as f:summary=list(csv.DictReader(f))
    assert len(rows)==6480 and len(summary)==2160
    maximum_symmetry=0.;minimum_eigenvalue=0.;delta_checks=0;trace_checks=0;matrix_checks=0
    index={(r['corpus'],int(r['n']),int(r['pair']),int(r['updates']),int(r['dimension']),r['mode'],r['method'],int(r['k'])):r for r in rows}
    max_capture_error=0.;real_direction_checks={}
    for corpus in ('tinystories','wikitext','cnn_dailymail'):
        meta=read(out/f'{corpus}_inputs.json');folds=np.array(meta['folds']);sources=meta['source_ids']
        assert len(folds)==128 and all(len({folds[i] for i,s in enumerate(sources) if s==name})==1 for name in set(sources))
        assert list(np.bincount(folds))==meta['fold_chunks']
        if workspace:
            with np.load(ROOT/f'results/mdlm_runpod_main_v1/{corpus}/context.npz') as z:mask=z['mask']
            point_rows=np.nonzero(mask)[0]
        for n in (512,1024,2048):
            with np.load(out/f'{corpus}_n{n}_reference.npz') as z:
                forecasts={m:z[m] for m in ('source_full','source_same_lag','multinomial','unigram')}
            if workspace:
                with np.load(ROOT/f'results/lag_disagreement_matched_v1/per_chunk/{corpus}_n{n}_forecasts.npz') as z:
                    for m,old in zip(z['methods'],z['forecasts']):
                        # Orthonormal projection cannot increase a PSD trace.
                        projected=np.trace(forecasts[str(m)],axis1=1,axis2=2)
                        original=np.array([old[folds==f].sum() for f in (0,1)])
                        assert np.all(projected>=0) and np.all(projected<=original+1e-10)
                        trace_checks+=1
                if n==2048:
                    real_direction_checks[corpus]=direct_direction_check(corpus,forecasts,np.load(out/'readout.npy'))
            for pair in (1,2,3):
                with np.load(out/f'{corpus}_n{n}_pair{pair}_empirical.npz') as z:
                    empirical={m:z[m] for m in ('reconstructor','affine')}
                    if workspace:
                        for m,delta in [('reconstructor',z['lag_delta']),('affine',z['affine_delta'])]:
                            direct=np.zeros((2,128,128))
                            for chunk in range(128):
                                a=delta[point_rows==chunk]
                                direct[folds[chunk]]+=np.einsum('pi,pj->ij',a,a)/len(a)
                            np.testing.assert_allclose(direct,empirical[m],atol=1e-12,rtol=1e-11)
                            delta_checks+=1
                for updates in (8000,12000):
                    with np.load(out/f'{corpus}_n{n}_pair{pair}_step{updates}_neural.npz') as z:
                        target=z['neural']
                        if workspace:
                            direct=np.zeros_like(target)
                            saved_delta=z['delta']
                            for chunk in range(128):
                                a=saved_delta[point_rows==chunk]
                                direct[folds[chunk]]+=np.einsum('pi,pj->ij',a,a)/len(a)
                            np.testing.assert_allclose(direct,target,atol=1e-12,rtol=1e-11)
                            original=read(ROOT/f'results/mdlm_runpod_main_v1/{corpus}/scores/pair{pair}_n{n}_step{updates}.json')
                            np.testing.assert_allclose(z['full_disagreement'],original['per_chunk']['disagreement'],rtol=2e-5,atol=2e-7)
                            delta_checks+=1
                    matrices={**forecasts,**empirical,'neural':target}
                    for matrix in matrices.values():
                        maximum_symmetry=max(maximum_symmetry,float(np.max(np.abs(matrix-matrix.transpose(0,2,1)))))
                        minimum_eigenvalue=min(minimum_eigenvalue,float(np.linalg.eigvalsh(matrix).min()))
                        matrix_checks+=2
                    assert maximum_symmetry<1e-10 and minimum_eigenvalue>-1e-10
                    for d in (64,128):
                        nt=target[:,:d,:d];denom=np.trace(nt.sum(0))
                        for m,matrix in {**forecasts,**empirical}.items():
                            mt=matrix[:,:d,:d]
                            for mode in ('source_crossfit','pooled'):
                                # Independent SVD rather than the runner's eigh helper.
                                energies=np.zeros(d)
                                if mode=='source_crossfit':
                                    for f in (0,1):
                                        u,_,_=np.linalg.svd(mt[1-f],hermitian=True)
                                        energies+=np.sum(u*(nt[f]@u),axis=0)/denom
                                else:
                                    u,_,_=np.linalg.svd(mt.sum(0),hermitian=True)
                                    energies=np.sum(u*(nt.sum(0)@u),axis=0)/denom
                                cumulative=np.cumsum(energies)
                                for k in (1,2,4,8,16):
                                    r=index[corpus,n,pair,updates,d,mode,m,k]
                                    err=abs(cumulative[k-1]-float(r['capture']))
                                    max_capture_error=max(max_capture_error,err)
                                    assert err<1e-9 and 0<=float(r['capture'])<=1
                                    assert float(r['random_expectation'])==k/d
                                    assert abs(float(r['random_mean'])-k/d)<.02
    keys=('corpus','n','updates','dimension','mode','method','k')
    groups={}
    for r in rows:groups.setdefault(tuple(r[k] for k in keys),[]).append(float(r['capture']))
    for s in summary:
        values=groups[tuple(s[k] for k in keys)];assert len(values)==3
        np.testing.assert_allclose(np.array([s['capture'],s['minimum'],s['maximum']],dtype=float),
                                   [np.mean(values),min(values),max(values)],rtol=1e-12)
    if workspace:
        R=np.load(out/'readout.npy')
        np.testing.assert_allclose(R.T@R,np.eye(128),atol=1e-12)
        np.testing.assert_allclose(R.sum(0),0,atol=1e-12)
    report=dict(status='passed',per_pair_rows=len(rows),summary_rows=len(summary),
        independently_rebuilt_delta_matrices=delta_checks,projection_trace_bounds=trace_checks,
        matrix_checks=matrix_checks,maximum_symmetry_error=maximum_symmetry,
        direct_real_direction_checks=real_direction_checks,
        minimum_eigenvalue=minimum_eigenvalue,maximum_capture_error=max_capture_error,
        workspace_checks=workspace,scope='No neural training rerun. Matrix laws also checked against independent dense toy formulas and existing scalar source traces.',
        files={p.name:sha(p) for p in out.glob('*.csv')})
    report_name='independent_validation.json' if workspace else 'matrix_rebuild_validation.json'
    (out/report_name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/mdlm_subspaces_v1')
    p.add_argument('--saved-matrices-only',action='store_true')
    a=p.parse_args();validate(a.output,workspace=not a.saved_matrices_only)
