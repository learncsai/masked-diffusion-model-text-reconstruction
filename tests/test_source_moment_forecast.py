import numpy as np
import pytest

from diffusion_lm_rmt.reference_ratio import reference_forecasts as original
from diffusion_lm_rmt.source_moment_forecast import reference_forecasts as moments
from diffusion_lm_rmt.reference_correction import source_split_jackknife as original_jackknife
from diffusion_lm_rmt.source_moment_forecast import source_split_jackknife as moment_jackknife


@pytest.mark.parametrize('seed',[7,19,31])
@pytest.mark.parametrize('alpha',[.1,5.,100.])
def test_exact_moment_contraction_matches_source_loop(seed,alpha):
    rng=np.random.default_rng(seed)
    reference=rng.integers(0,13,(31,8))
    evaluation=rng.integers(0,17,(7,8))
    mask=rng.random(evaluation.shape)<.5
    mask[:,2]=True
    prior=rng.dirichlet(np.ones(17))
    matrix=rng.normal(size=(4,4))
    gram=matrix.T@matrix+np.eye(4)
    cross=rng.normal(size=4)
    groups=[np.arange(a,b) for a,b in [(0,1),(1,5),(5,12),(12,18),(18,25),(25,31)]]
    args=(reference,groups,evaluation,mask,prior,(-2,-1,1,2),gram,cross,[9,31,80])
    expected=original(*args,smoothing=alpha,ridge=.01)
    actual=moments(*args,smoothing=alpha,ridge=.01)
    for n in expected:
        for key in ['forecasts','point_forecasts','point_weight','minimum_row_count','strata']:
            np.testing.assert_allclose(actual[n][key],expected[n][key],rtol=1e-10,atol=1e-12)
    # The jackknife must preserve exactly the same random source partition.
    a=original_jackknife(*args,smoothing=alpha,ridge=.01,seed=41)
    b=moment_jackknife(*args,smoothing=alpha,ridge=.01,seed=41)
    for n in a:
        np.testing.assert_allclose(a[n]['jackknife_raw'],b[n]['jackknife_raw'],rtol=1e-10,atol=1e-12)


def test_all_reference_rows_absent_is_zero_without_invented_uncertainty():
    reference=np.zeros((4,4),dtype=int)
    evaluation=np.ones((3,4),dtype=int)
    mask=np.zeros_like(evaluation,dtype=bool)
    mask[:,2]=True
    args=(reference,[np.array([i]) for i in range(4)],evaluation,mask,np.ones(2)/2,
          (-1,1),np.eye(2),np.ones(2),[4])
    result=moments(*args,smoothing=5.,ridge=.01)[4]
    np.testing.assert_array_equal(result['point_forecasts'],np.zeros((3,3)))
    np.testing.assert_array_equal(result['strata'],np.zeros(3))
