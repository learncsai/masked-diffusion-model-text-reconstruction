"""The existing source law evaluated through reusable sparse source moments.

No statistical approximation is introduced. Only feature pairs that co-occur
in an evaluation input are contracted. Sources remain the independent units.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from . import sparse_categorical_lag as lag
from .categorical_lag import pattern_weights, per_sequence_squared_error


def pair_inner(left,right,first,second,batch=64):
    values=np.empty(len(first))
    for begin in range(0,len(first),batch):
        end=begin+batch
        values[begin:end]=np.asarray(left[first[begin:end]].multiply(
            right[second[begin:end]]).sum(axis=1)).ravel()
    return values


def reference_forecasts(reference,groups,evaluation,mask,prior,offsets,gram,cross,
                        sizes,*,smoothing,ridge,progress=None):
    S,m,K=len(groups),len(reference),len(prior)
    if S<2:
        raise ValueError('At least two independent sources are needed')
    assignment=np.full(m,-1,dtype=np.int64)
    lengths=np.array([len(g) for g in groups])
    for index,group in enumerate(groups):
        if len(group)==0 or len(np.unique(group))!=len(group) or np.any((group<0)|(group>=m)):
            raise ValueError('Invalid source group')
        if np.any(assignment[group]!=-1):
            raise ValueError('Source groups overlap')
        assignment[group]=index
    if np.any(assignment<0):
        raise ValueError('Source groups must partition reference chunks')
    rows,_,contexts,visible,patterns=lag.context_design(evaluation,mask,offsets)
    P,J=len(rows),len(offsets)
    weights=np.zeros((P,J))
    for code in np.unique(patterns):
        if code:
            active=tuple(j for j in range(J) if code&(1<<j))
            weights[np.ix_(np.flatnonzero(patterns==code),active)]=pattern_weights(gram,cross,active,ridge)
    feature=np.full((P,J),-1,dtype=np.int64)
    lookups=[]
    F=0
    for j in range(J):
        chosen=np.unique(contexts[j][visible[j]])
        lookup=np.full(K,-1,dtype=np.int64)
        lookup[chosen]=np.arange(F,F+len(chosen))
        feature[visible[j],j]=lookup[contexts[j][visible[j]]]
        F+=len(chosen)
        lookups.append(lookup)
    if F==0:
        result={}
        for n in sizes:
            result[n]=dict(forecasts=np.zeros((3,len(evaluation))),point_forecasts=np.zeros((3,P)),
                rows=rows,point_weight=np.zeros(P),strata=np.full(P,5),minimum_row_count=np.full(P,np.inf))
        return result
    sources,features,targets=[],[],[]
    for j,d in enumerate(offsets):
        if d>0:
            context,target=reference[:,:-d],reference[:,d:]
        else:
            context,target=reference[:,-d:],reference[:,:d]
        f=lookups[j][context.ravel()]
        selected=f>=0
        features.append(f[selected])
        targets.append(target.ravel()[selected])
        sources.append(np.repeat(assignment,context.shape[1])[selected])
    sf=np.concatenate(features)
    st=np.concatenate(targets)
    sg=np.concatenate(sources)
    values=np.ones(len(sf))
    # Columns of X identify (source,target-token) pairs. Compressing these IDs
    # avoids an S*K sparse row pointer while preserving exact inner products.
    _,source_target=np.unique(sg*K+st,return_inverse=True)
    X=sparse.coo_matrix((values,(sf,source_target)),shape=(F,int(source_target.max(initial=-1))+1)).tocsr()
    N=sparse.coo_matrix((values,(sg,sf)),shape=(S,F)).tocsr()
    Z=sparse.coo_matrix((prior[st],(sg,sf)),shape=(S,F)).tocsr()
    C=sparse.coo_matrix((values,(sf,st)),shape=(F,K)).tocsr()
    W=sparse.coo_matrix((lengths[sg].astype(float),(sf,st)),shape=(F,K)).tocsr()
    totals=np.asarray(C.sum(axis=1)).ravel()
    weighted_totals=np.asarray(W.sum(axis=1)).ravel()
    cp=np.asarray(C@prior).ravel()
    wp=np.asarray(W@prior).ravel()
    p2=float(prior@prior)
    # Every oriented pair is retained so B_fh and B_hf can be read directly.
    codes=[]
    for j in range(J):
        for k in range(J):
            selected=(feature[:,j]>=0)&(feature[:,k]>=0)
            codes.append(feature[selected,j]*F+feature[selected,k])
    codes=np.unique(np.concatenate(codes))
    f,h=codes//F,codes%F
    reverse=np.searchsorted(codes,h*F+f)
    A=pair_inner(X,X,f,h)
    NT,ZT=N.T.tocsr(),Z.T.tocsr()
    D=pair_inner(NT,NT,f,h)
    Bp=pair_inner(ZT,NT,f,h)
    G=pair_inner(C,C,f,h)
    WG=pair_inner(W,C,f,h)
    B=np.zeros(len(codes))
    # For each feature f, contract count vectors with only the neighbor rows
    # h actually used by an evaluation input: sum_g N_gh <C_gf,C_ref,h>.
    order=np.argsort(sf,kind='stable')
    ordered_feature=sf[order]
    boundaries=np.searchsorted(ordered_feature,np.arange(F+1))
    N_csc=N.tocsc()
    for feature_id in range(F):
        where=np.flatnonzero(f==feature_id)
        selected=order[boundaries[feature_id]:boundaries[feature_id+1]]
        if len(selected):
            Cf=sparse.coo_matrix((np.ones(len(selected)),(sg[selected],st[selected])),shape=(S,K)).tocsr()
            neighbor_counts=N_csc[:,h[where]].T@Cf
            B[where]=np.asarray(neighbor_counts.multiply(C[h[where]]).sum(axis=1)).ravel()
        if progress and (feature_id+1)%500==0:
            progress(feature_id+1,F)
    minimum=np.full(P,np.inf)
    for j in range(J):
        chosen=feature[:,j]>=0
        minimum[chosen]=np.minimum(minimum[chosen],totals[feature[chosen,j]])
    bins=np.select([np.isinf(minimum),minimum==0,minimum<5,minimum<20,minimum<100],
                   [5,0,1,2,3],default=4)
    hidden=np.bincount(rows,minlength=len(evaluation))
    point_weight=1/(len(evaluation)*hidden[rows])
    collision=(pair_inner(C,C,np.arange(F),np.arange(F))+2*smoothing*cp+smoothing**2*p2)/(totals+smoothing)**2
    result={}
    for n in sizes:
        scale=n/m
        denominator=scale*totals+smoothing
        df,dh=denominator[f],denominator[h]
        p_inner=(scale**2*G+scale*smoothing*(cp[f]+cp[h])+smoothing**2*p2)/(df*dh)
        raw=A-(scale*B+smoothing*Bp)/dh-(scale*B[reverse]+smoothing*Bp[reverse])/df+D*p_inner
        weighted=smoothing/dh*(WG-totals[h]*wp[f]-weighted_totals[f]*(
            scale*(G-totals[h]*cp[f])+smoothing*(cp[h]-totals[h]*p2))/df)
        ref_inner=smoothing**2/(df*dh)*(G-totals[h]*cp[f]-totals[f]*cp[h]+totals[f]*totals[h]*p2)
        centered=raw-(weighted+weighted[reverse])/m+np.sum(lengths**2)*ref_inner/m**2
        full,same,multi=np.zeros(P),np.zeros(P),np.zeros(P)
        for j in range(J):
            chosen=feature[:,j]>=0
            ff=feature[chosen,j]
            multi[chosen]+=weights[chosen,j]**2*2*scale*totals[ff]/denominator[ff]**2*np.maximum(1-collision[ff],0)
            for k in range(j+1):
                selected=(feature[:,j]>=0)&(feature[:,k]>=0)
                first,second=feature[selected,j],feature[selected,k]
                pair=np.searchsorted(codes,first*F+second)
                contribution=weights[selected,j]*weights[selected,k]/denominator[first]/denominator[second]*centered[pair]
                full[selected]+=(1 if j==k else 2)*contribution
                if j==k:
                    same[selected]+=contribution
        if np.min(full,initial=0)<-1e-9 or np.min(same,initial=0)<-1e-9:
            raise ArithmeticError('Negative source variance')
        multiplier=2*(n*S/m)/(S-1)
        points=np.asarray([multi,multiplier*np.maximum(same,0),multiplier*np.maximum(full,0)])
        forecasts=np.asarray([per_sequence_squared_error(rows,p,len(evaluation)) for p in points])
        np.testing.assert_allclose(points@point_weight,forecasts.mean(axis=1),rtol=1e-11,atol=1e-13)
        result[n]=dict(forecasts=forecasts,point_forecasts=points,rows=rows,point_weight=point_weight,
                       strata=bins,minimum_row_count=minimum,source_count=S,reference_chunks=m)
    return result


def source_split_jackknife(reference,groups,evaluation,mask,prior,offsets,gram,cross,
                           sizes,*,smoothing,ridge,seed):
    if len(groups)<4:
        raise ValueError('Each source half needs at least two sources')
    kw=dict(smoothing=smoothing,ridge=ridge)
    full=reference_forecasts(reference,groups,evaluation,mask,prior,offsets,gram,cross,sizes,**kw)
    order=np.random.default_rng(seed).permutation(len(groups))
    halves=[]
    fractions=[]
    for chosen in np.array_split(order,2):
        indices=np.concatenate([groups[i] for i in chosen])
        bounds=np.cumsum([0,*[len(groups[i]) for i in chosen]])
        local=[np.arange(a,b) for a,b in zip(bounds[:-1],bounds[1:])]
        halves.append(reference_forecasts(reference[indices],local,evaluation,mask,prior,
            offsets,gram,cross,sizes,**kw))
        fractions.append(len(chosen)/len(groups))
    for n,value in full.items():
        half=sum(w*h[n]['point_forecasts'][2] for w,h in zip(fractions,halves))
        corrected=2*value['point_forecasts'][2]-half
        value['jackknife_raw']=corrected
        value['jackknife_mean_raw']=float(corrected@value['point_weight'])
        value['jackknife_mean']=max(value['jackknife_mean_raw'],0.)
        value['negative_point_fraction']=float(np.mean(corrected<0))
        value['half_source_fractions']=np.asarray(fractions)
    return full
