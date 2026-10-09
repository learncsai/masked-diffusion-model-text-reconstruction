"""Independent dense and trace checks for the directional covariance laws."""
import numpy as np
from diffusion_lm_rmt.disagreement_subspaces import (
    forecast_matrices, vocabulary_readout, source_folds, second_moments, capture)
from diffusion_lm_rmt.sparse_categorical_influence import source_influence_disagreement
from diffusion_lm_rmt import sparse_categorical_lag as lag
from diffusion_lm_rmt.categorical_lag import pattern_weights


def test_matrix_extension_matches_existing_traces_and_dense_covariances():
    rng=np.random.default_rng(432)
    reference=rng.integers(0,7,(11,6));ids=rng.integers(0,7,(8,6))
    mask=rng.random(ids.shape)<.5;mask[:,2]=True
    groups=[np.arange(0,1),np.arange(1,4),np.arange(4,6),np.arange(6,11)]
    folds=np.arange(8)%2;prior=rng.dirichlet(np.ones(7))
    offsets=(-2,-1,1,2);gram=np.eye(4);cross=np.array([.2,-.1,.4,.3])
    n=23;smoothing=3.;ridge=.1;R=np.eye(7)
    kw=dict(n=n,smoothing=smoothing,ridge=ridge)
    got=forecast_matrices(reference,groups,ids,mask,folds,prior,offsets,gram,cross,readout=R,**kw)
    for name,same in [('source_full',False),('source_same_lag',True)]:
        old=source_influence_disagreement(reference,groups,ids,mask,prior,offsets,gram,cross,same_lag=same,**kw)
        np.testing.assert_allclose(np.trace(got[name],axis1=1,axis2=2),[old[folds==f].sum() for f in (0,1)],atol=1e-12)
    # Independent direct coordinate formula, one masked position at a time.
    c={d:v.toarray() for d,v in lag.pair_counts(reference,offsets,7).items()}
    cg=[{d:v.toarray() for d,v in lag.pair_counts(reference[g],offsets,7).items()} for g in groups]
    dense={key:np.zeros((2,7,7)) for key in got}
    for row,pos in zip(*np.nonzero(mask)):
        active=tuple(j for j,d in enumerate(offsets) if 0<=pos-d<6 and not mask[row,pos-d])
        if not active:continue
        ws=pattern_weights(gram,cross,active,ridge)
        influences=np.zeros((len(groups),len(active),7))
        for jj,(j,w) in enumerate(zip(active,ws)):
            d=offsets[j];a=ids[row,pos-d];counts=c[d][a];total=counts.sum();N=n/11*total
            P=(n/11*counts+smoothing*prior)/(N+smoothing)
            pref=(counts+smoothing*prior)/(total+smoothing)
            dense['multinomial'][folds[row]]+=w*w*2*N/(N+smoothing)**2*(np.diag(pref)-np.outer(pref,pref))/mask[row].sum()
            for g,group in enumerate(groups):
                r=cg[g][d][a]-len(group)/11*counts
                influences[g,jj]=w*(r-r.sum()*P)/(N+smoothing)
        for name,array in [('source_full',influences.sum(1)[:,None,:]),('source_same_lag',influences)]:
            dense[name][folds[row]]+=2*(n*len(groups)/11)/(len(groups)-1)*np.einsum('gji,gjk->ik',array,array)/mask[row].sum()
    for name in got:
        np.testing.assert_allclose(got[name],dense[name],atol=1e-12)
        np.testing.assert_allclose(got[name].sum(-1),0,atol=1e-12)
    projection=vocabulary_readout(7,3)
    reduced=forecast_matrices(reference,groups,ids,mask,folds,prior,offsets,gram,cross,readout=projection,**kw)
    for name in got:np.testing.assert_allclose(reduced[name],projection.T@got[name]@projection,atol=1e-12)


def test_source_split_and_energy_not_pointwise_sign():
    ids=np.array(['a','a','b','c','c','c','d'])
    folds,counts=source_folds(ids)
    assert counts.sum()==len(ids)
    for name in set(ids):assert len(set(folds[ids==name]))==1
    R=vocabulary_readout(19,8)
    np.testing.assert_allclose(R.T@R,np.eye(8),atol=1e-12)
    np.testing.assert_allclose(R.sum(0),0,atol=1e-12)
    mask=np.ones((2,2),dtype=bool);f=np.array([0,1])
    delta=np.array([[1.,0],[2.,0],[-1.,0],[-2.,0]])
    m=second_moments(delta,mask,f)
    assert np.isclose(capture(m,m,1),1)
    np.testing.assert_array_equal(m,second_moments(-delta,mask,f))
