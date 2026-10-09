import numpy as np
import pytest
from scipy import sparse
from diffusion_lm_rmt import alpha_sweep as sweep,sparse_categorical_lag as lag
from diffusion_lm_rmt.source_moment_forecast import reference_forecasts


@pytest.mark.parametrize('alpha',[.5,5,50])
def test_cached_analytic_and_joint_bootstrap_match_original(alpha):
    rng=np.random.default_rng(28);K=19
    reference=rng.integers(K,size=(31,8));evaluation=rng.integers(K,size=(9,8))
    mask=rng.random(evaluation.shape)<.5;mask[:,3]=True
    p=rng.dirichlet(np.ones(K));g=np.eye(4);b=np.array([.7,-.2,.3,.1])
    groups=[np.arange(i,min(i+3,31)) for i in range(0,31,3)];offsets=(-2,-1,1,2)
    state=sweep.prepare_moments(reference,groups,evaluation,mask,p,offsets)
    actual=sweep.evaluate_moments(state,g,b,[12,53],smoothing=alpha,ridge=.01)
    expected=reference_forecasts(reference,groups,evaluation,mask,p,offsets,g,b,[12,53],smoothing=alpha,ridge=.01)
    for n in actual:np.testing.assert_allclose(actual[n]['point_forecasts'],expected[n]['point_forecasts'],rtol=1e-10,atol=1e-12)
    design=sweep.geometry(evaluation,mask,p,offsets);w,metric=sweep.weights_and_metric(design,g,b)
    fast=sweep.bootstrap_sweep(reference,groups,design,[alpha],metric[None,None,:],n=23,draws=7,rng=np.random.default_rng(10))[0,0]
    old,_=lag.bootstrap_disagreement(reference,groups,evaluation,mask,p,offsets,g,b,n=23,smoothing=alpha,ridge=.01,draws=7,law='source_cluster',rng=np.random.default_rng(10))
    np.testing.assert_allclose(fast,old.mean(),rtol=2e-10,atol=1e-12)


def test_sparse_top1_with_signed_scores_and_ties():
    rng=np.random.default_rng(52);p=np.array([0.,.3,.3,.4,0.])
    a=rng.normal(size=(40,5));a[rng.random(a.shape)<.7]=0;a[0]=0
    b=rng.choice([-2.,0.,1.,2.],40)
    pred=lag.Prediction(sparse.csr_matrix(a),b,p)
    np.testing.assert_array_equal(sweep.exact_top1(pred),np.argmax(pred.toarray(),axis=1))


def test_same_lag_coverage_partition_and_large_smoothing_limit():
    rng=np.random.default_rng(19);K=7;p=np.ones(K)/K;offsets=(-1,1)
    e=rng.integers(K,size=(4,6));mask=rng.random(e.shape)<.5;mask[:,2]=True
    a=rng.integers(5,size=(12,6));b=rng.integers(2,7,size=(12,6))
    ca=lag.pair_counts(a,offsets,K);cb=lag.pair_counts(b,offsets,K)
    design=sweep.geometry(e,mask,p,offsets);weights,_=sweep.weights_and_metric(design,np.eye(2),np.ones(2))
    bins=np.arange(design['F'])%3
    same,coverage=sweep.row_decomposition(ca,cb,design,5,weights,bins,3)
    np.testing.assert_allclose(same.sum(),coverage.sum())
    for counts in (ca,cb):
        _,_,pred=lag.masked_prediction(e,mask,lag.conditional_tables(counts,p,1e12),p,offsets,np.eye(2),np.ones(2),.01)
        np.testing.assert_allclose(pred.toarray(),np.broadcast_to(p,pred.residual.shape),atol=1e-10)
