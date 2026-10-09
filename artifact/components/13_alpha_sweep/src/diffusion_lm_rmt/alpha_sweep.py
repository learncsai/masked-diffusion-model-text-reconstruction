"""Reusable exact contractions for a paired smoothing sweep.

The analytic algebra is adapted from source_moment_forecast.py. Counts and
source moments are calculated once, then smoothing and auxiliary weights vary.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from . import sparse_categorical_lag as lag
from .source_moment_forecast import pair_inner
from .categorical_lag import pattern_weights,per_sequence_squared_error,draw_reference_chunks


def prepare_moments(reference,groups,evaluation,mask,prior,offsets,progress=None):
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
    if F==0: raise ValueError('At least one visible lag is required')
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
    return dict(c2=np.asarray(C.multiply(C).sum(axis=1)).ravel(), S=S, m=m, K=K, rows=rows, P=P, J=J, patterns=patterns, feature=feature, totals=totals, weighted_totals=weighted_totals, cp=cp, wp=wp, p2=p2, codes=codes, f=f, h=h, reverse=reverse, A=A, D=D, Bp=Bp, G=G, WG=WG, B=B, lengths=lengths, minimum=minimum, bins=bins, point_weight=point_weight)

def evaluate_moments(state,gram,cross,sizes,*,smoothing,ridge):
    S=state['S']
    m=state['m']
    K=state['K']
    rows=state['rows']
    P=state['P']
    J=state['J']
    patterns=state['patterns']
    feature=state['feature']
    totals=state['totals']
    F=len(totals)
    weighted_totals=state['weighted_totals']
    cp=state['cp']
    wp=state['wp']
    p2=state['p2']
    codes=state['codes']
    f=state['f']
    h=state['h']
    reverse=state['reverse']
    A=state['A']
    D=state['D']
    Bp=state['Bp']
    G=state['G']
    WG=state['WG']
    B=state['B']
    lengths=state['lengths']
    minimum=state['minimum']
    bins=state['bins']
    point_weight=state['point_weight']
    weights=np.zeros((P,J))
    for code in np.unique(patterns):
        if code:
            active=tuple(j for j in range(J) if code&(1<<j))
            weights[np.ix_(np.flatnonzero(patterns==code),active)]=pattern_weights(gram,cross,active,ridge)
    evaluation=range(int(rows.max())+1)
    collision=(state["c2"]+2*smoothing*cp+smoothing**2*p2)/(totals+smoothing)**2
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


def geometry(evaluation,mask,prior,offsets):
    rows,positions,contexts,visible,patterns=lag.context_design(evaluation,mask,offsets)
    K,J=len(prior),len(offsets)
    feature=np.full((len(rows),J),-1,dtype=int);chosen=[];F=0
    for j in range(J):
        tokens=np.unique(contexts[j][visible[j]])
        lookup=np.full(K,-1,dtype=int);lookup[tokens]=np.arange(F,F+len(tokens))
        feature[visible[j],j]=lookup[contexts[j][visible[j]]]
        chosen.append(tokens);F+=len(tokens)
    codes=[]
    for j in range(J):
        for k in range(J):
            sel=(feature[:,j]>=0)&(feature[:,k]>=0)
            codes.append(feature[sel,j]*F+feature[sel,k])
    codes=np.unique(np.concatenate(codes))
    pw=1/(len(evaluation)*np.bincount(rows,minlength=len(evaluation))[rows])
    return dict(rows=rows,targets=evaluation[rows,positions],contexts=contexts,visible=visible,
        patterns=patterns,feature=feature,chosen=chosen,F=F,codes=codes,f=codes//F,h=codes%F,
        pw=pw,prior=prior,offsets=offsets,chunks=len(evaluation))


def weights_and_metric(design,gram,cross,ridge=.01):
    feature=design['feature'];J=feature.shape[1];weights=np.zeros_like(feature,dtype=float)
    for code in np.unique(design['patterns']):
        if code:
            active=tuple(j for j in range(J) if code&(1<<j))
            weights[np.ix_(np.flatnonzero(design['patterns']==code),active)]=pattern_weights(gram,cross,active,ridge)
    metric=np.zeros(len(design['codes']))
    for j in range(J):
        for k in range(J):
            sel=(feature[:,j]>=0)&(feature[:,k]>=0)
            pair=np.searchsorted(design['codes'],feature[sel,j]*design['F']+feature[sel,k])
            metric+=np.bincount(pair,weights=design['pw'][sel]*weights[sel,j]*weights[sel,k],minlength=len(metric))
    return weights,metric


def queried_counts(counts,design):
    return sparse.vstack([counts[d][tokens] for d,tokens in zip(design['offsets'],design['chosen'])],format='csr')


def bootstrap_sweep(reference,groups,design,alphas,metrics,*,n,draws,rng):
    """Exact twice sample variance using reusable row inner products.

    metrics has shape weight-arms x alphas x queried-feature-pairs. Every
    setting uses the same source draws. Full K-dimensional inner products
    are retained. No vocabulary projection or row independence is introduced.
    """
    p=design['prior'];p2=float(p@p);f,h=design['f'],design['h'];F=design['F']
    Q=[sparse.csr_matrix((F,len(p)),dtype=float) for _ in alphas]
    beta=np.zeros((len(alphas),F));norms=np.zeros(metrics.shape[:2])
    for _ in range(draws):
        sample=draw_reference_chunks(reference,groups,n,rng)
        C=queried_counts(lag.pair_counts(sample,design['offsets'],len(p)),design)
        totals=np.asarray(C.sum(axis=1)).ravel();cp=C@p
        kernel=pair_inner(C,C,f,h)-totals[h]*cp[f]-totals[f]*cp[h]+totals[f]*totals[h]*p2
        for ai,alpha in enumerate(alphas):
            den=totals+alpha
            norms[:,ai]+=metrics[:,ai]@(kernel/(den[f]*den[h]))
            Q[ai]=Q[ai]+C.multiply((1/den)[:,None])
            beta[ai]-=totals/den
    result=np.empty_like(norms)
    for ai in range(len(alphas)):
        cp=Q[ai]@p;b=beta[ai]
        kernel=pair_inner(Q[ai],Q[ai],f,h)+b[f]*cp[h]+b[h]*cp[f]+b[f]*b[h]*p2
        centered=norms[:,ai]-metrics[:,ai]@kernel/draws
        if centered.min() < -1e-9:raise ArithmeticError('Negative bootstrap variance')
        result[:,ai]=2*np.maximum(centered,0)/(draws-1)
    return result


def exact_top1(prediction):
    """Exact sparse-plus-prior argmax, with smallest-token-ID tie breaking."""
    residual=prediction.residual.copy();residual.eliminate_zeros();residual.sort_indices()
    beta,p=prediction.prior_weight,prediction.prior
    P,K=residual.shape;row=np.repeat(np.arange(P),np.diff(residual.indptr))
    values=residual.data+beta[row]*p[residual.indices]
    best=np.full(P,-np.inf);nonempty=np.flatnonzero(np.diff(residual.indptr))
    if len(nonempty):best[nonempty]=np.maximum.reduceat(values,residual.indptr[nonempty])
    ids=np.full(P,K,dtype=int)
    if len(nonempty):
        candidate=np.where(values==best[row],residual.indices,K)
        ids[nonempty]=np.minimum.reduceat(candidate,residual.indptr[nonempty])
    for sign in (True,False):
        active=np.flatnonzero((beta>=0)==sign)
        order=np.lexsort((np.arange(K),-p if sign else p))
        # With beta==0 all outside scores tie at zero, requiring ID order.
        if sign:active=active[beta[active]!=0]
        for token in order:
            if not len(active):break
            missing=np.asarray(residual[active,np.full(len(active),token)]).ravel()==0
            which=active[missing];v=beta[which]*p[token]
            change=(v>best[which])|((v==best[which])&(token<ids[which]))
            best[which[change]]=v[change];ids[which[change]]=token
            active=active[~missing]
    active=np.flatnonzero(beta==0)
    for token in range(K):
        if not len(active):break
        missing=np.asarray(residual[active,np.full(len(active),token)]).ravel()==0
        which=active[missing];change=(0>best[which])|((best[which]==0)&(token<ids[which]))
        ids[which[change]]=token;active=active[~missing]
    return ids


def row_decomposition(ca,cb,design,alpha,weights,bins,bin_count):
    """Weighted same-lag energy and coverage categories, with fixed row bins."""
    p=design['prior'];p2=float(p@p)
    A=queried_counts(ca,design);B=queried_counts(cb,design)
    na=np.asarray(A.sum(axis=1)).ravel();nb=np.asarray(B.sum(axis=1)).ravel()
    delta=A.multiply((1/(na+alpha))[:,None])-B.multiply((1/(nb+alpha))[:,None])
    b=-na/(na+alpha)+nb/(nb+alpha)
    energy=np.asarray(delta.multiply(delta).sum(axis=1)).ravel()+2*b*(delta@p)+b*b*p2
    result=np.zeros(bin_count);coverage=np.zeros(2)
    for j in range(weights.shape[1]):
        selected=design['feature'][:,j]>=0;f=design['feature'][selected,j]
        term=design['pw'][selected]*weights[selected,j]**2*energy[f]
        result+=np.bincount(bins[f],weights=term,minlength=bin_count)
        coverage[0]+=term[(na[f]>0)&(nb[f]>0)].sum()
        coverage[1]+=term[(na[f]>0)^(nb[f]>0)].sum()
    np.testing.assert_allclose(result.sum(),coverage.sum(),rtol=1e-10,atol=1e-12)
    return result,coverage
