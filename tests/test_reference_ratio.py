import numpy as np
import pytest
from diffusion_lm_rmt.reference_ratio import reference_forecasts
from diffusion_lm_rmt.sparse_categorical_influence import source_influence_disagreement
from diffusion_lm_rmt.categorical_lag import offset_list
from scripts.evaluate_lag_disagreement import multinomial_forecast


@pytest.mark.parametrize('seed',[13,47])
def test_batched_points_match_frozen_laws(seed):
    rng=np.random.default_rng(seed)
    reference=rng.integers(0,13,(31,8))
    evaluation=rng.integers(0,17,(7,8))
    mask=rng.random((7,8))<.5
    mask[:,2]=True
    prior=rng.dirichlet(np.ones(17))
    offsets=offset_list(2)
    matrix=rng.normal(size=(4,4))
    gram=matrix.T@matrix+np.eye(4)
    cross=rng.normal(size=4)
    groups=[np.arange(a,b) for a,b in [(0,1),(1,5),(5,12),(12,18),(18,25),(25,31)]]
    kw=dict(smoothing=5.,ridge=.01)
    args=(reference,groups,evaluation,mask,prior,offsets,gram,cross)
    got=reference_forecasts(*args,[9,31,80],**kw)
    for n,data in got.items():
        expected=[multinomial_forecast(reference,evaluation,mask,prior,offsets,gram,cross,n=n,**kw)]
        expected += [source_influence_disagreement(*args,n=n,same_lag=same,**kw) for same in [True,False]]
        np.testing.assert_allclose(data['forecasts'],expected,rtol=2e-12,atol=1e-12)
        for method in range(3):
            terms=[float(np.sum(data['point_forecasts'][method][data['strata']==b]
                                *data['point_weight'][data['strata']==b])) for b in range(6)]
            assert sum(terms)==pytest.approx(data['forecasts'][method].mean(),rel=1e-12,abs=1e-12)
    assert 0 in got[9]['strata']  # Evaluation queries can be absent from the reference.
