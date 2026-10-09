"""Independent checks for source resampling and unequal-size direct extrapolation."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.run_correction_followup import halves
from diffusion_lm_rmt import sparse_categorical_lag as lag
from diffusion_lm_rmt.categorical_lag import draw_reference_chunks,per_sequence_squared_error

def test_source_bootstrap_matches_dense_sample_output_variance():
    rng=np.random.default_rng(133)
    tokens=rng.integers(0,11,size=(19,8))
    groups=[np.arange(a,b) for a,b in [(0,1),(1,5),(5,7),(7,13),(13,19)]]
    evaluation=rng.integers(0,13,size=(5,8));mask=rng.random((5,8))<.5;mask[:,1]=True
    prior=rng.dirichlet(np.ones(13));offsets=(-2,-1,1,2)
    gram=np.eye(4);cross=np.array([.3,-.2,.1,.4])
    # Independently materialize all K coordinates, using identical source draws.
    dense=[];drawrng=np.random.default_rng(777)
    for _ in range(16):
        sample=draw_reference_chunks(tokens,groups,27,drawrng)
        tables=lag.conditional_tables(lag.pair_counts(sample,offsets,13),prior,5)
        rows,_,p=lag.masked_prediction(evaluation,mask,tables,prior,offsets,gram,cross,.01)
        dense.append(p.residual.toarray()+p.prior_weight[:,None]*prior)
    expected=per_sequence_squared_error(rows,2*np.var(dense,axis=0,ddof=1).sum(axis=1),len(evaluation))
    actual,_=lag.bootstrap_disagreement(tokens,groups,evaluation,mask,prior,offsets,gram,cross,
        n=27,smoothing=5,ridge=.01,draws=16,law='source_cluster',rng=np.random.default_rng(777))
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-14)

def test_source_split_keeps_unequal_groups_whole_and_correctly_weighted():
    tokens=np.arange(19)[:,None]
    groups=[np.arange(a,b) for a,b in [(0,1),(1,5),(5,7),(7,13),(13,19)]]
    result=halves(tokens,groups,818)
    used=[]
    for sample,local,fraction in result:
        assert fraction==len(local)/len(groups)
        for g in local:
            original=sample[g,0]
            assert any(np.array_equal(original,old) for old in groups)
        used.append(set(sample[:,0]))
    assert not used[0]&used[1] and used[0]|used[1]==set(range(19))

def test_harmonic_half_size_recovers_inverse_size_disagreement():
    for m1,m2,n in [(7,11,64),(10,10,100),(3,50,20)]:
        trace=.73
        d_split=trace*(1/m1+1/m2)
        h=2/(1/m1+1/m2)
        np.testing.assert_allclose(d_split*h/n,2*trace/n,rtol=1e-14)
