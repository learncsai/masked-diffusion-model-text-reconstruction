import numpy as np
import pytest

from diffusion_lm_rmt import reference_correction as correction
from diffusion_lm_rmt.population_calibration import SparsePopulation
from diffusion_lm_rmt.sparse_categorical_lag import pair_counts


def test_source_split_cancels_inverse_source_bias_without_splitting_sources(monkeypatch):
    groups = [np.arange(a,b) for a,b in [(0,1),(1,4),(4,6),(6,10),(10,11),(11,16),(16,18)]]
    reference = np.arange(18)[:,None]
    identity = {int(i):k for k,g in enumerate(groups) for i in g}
    called = []
    def artificial(ref, local_groups, *args, **kwargs):
        seen=[]
        for group in local_groups:
            ids=ref[group,0]
            source=identity[int(ids[0])]
            assert all(identity[int(i)]==source for i in ids)
            assert set(ids)==set(groups[source])
            seen.append(source)
        called.append(set(seen))
        value=3.0+8.0/len(local_groups)
        return {99:dict(point_forecasts=np.full((3,2),value), point_weight=np.array([.4,.6]))}
    monkeypatch.setattr(correction,'reference_forecasts',artificial)
    result=correction.source_split_jackknife(reference,groups,None,None,None,None,None,None,
        [99],smoothing=5.,ridge=.01,seed=133)[99]
    assert not called[1]&called[2]
    assert called[1]|called[2]==called[0]
    np.testing.assert_allclose(result['jackknife_raw'],3.,atol=1e-14)


def test_crossed_intervals_keep_identical_methods_exactly_paired():
    forecasts=np.column_stack([[.8,1.1,1.3,.95],[.8,1.1,1.3,.95]])
    observations=np.linspace(.5,1.5,24)
    result=correction.crossed_interval(forecasts,observations,seed=14,draws=400)
    np.testing.assert_array_equal(result['paired_error_difference_ci'][1], [0.,0.])
    np.testing.assert_array_equal(result['paired_error_difference'],[0.,0.])


def test_nonpositive_forecast_is_not_silently_replaced_in_log_inference():
    with pytest.raises(ValueError,match='positive'):
        correction.crossed_interval(np.array([[0.,1.],[1.,2.]]),[1.,2.],seed=3)


def test_clustered_sampler_retains_known_lag_means_and_topic_dependence():
    population=SparsePopulation(dict(classes=3,chunk_length=4,repeat_rate=.25,
        topic_weight=1.,topic_size=1,topics=6,zipf_exponent=1.05),np.random.default_rng(3))
    ids,groups=correction.sample_source_chunks(population,80000,4,np.random.default_rng(42))
    # First token is sampled directly from the common one-token topic.
    assert all(np.all(ids[g,0]==ids[g[0],0]) for g in groups)
    counts=pair_counts(ids,(-1,1),3)
    for d,mean in population.mean_lag_counts((-1,1)).items():
        np.testing.assert_allclose(counts[d].toarray()/len(ids),mean.toarray(),atol=.018,rtol=0)


def test_exact_random_row_variance_identity_by_enumeration():
    # Two output categories, Binomial(3,.3) random row count, and conditional
    # target probability .7. Directly enumerate every (N,C_1) event.
    from math import comb
    pi,p,alpha=.7,.2,2.
    values,probabilities=[],[]
    e_noise,e_shrink,e_shrink2=0.,0.,0.
    for count in range(4):
        mass=comb(3,count)*.3**count*.7**(3-count)
        shrink=count/(count+alpha)
        e_noise+=mass*count/(count+alpha)**2
        e_shrink+=mass*shrink
        e_shrink2+=mass*shrink**2
        for first in range(count+1):
            probabilities.append(mass*comb(count,first)*pi**first*(1-pi)**(count-first))
            row=(first+alpha*p)/(count+alpha)
            values.append([row,1-row])
    probabilities=np.asarray(probabilities)
    values=np.asarray(values)
    mean=probabilities@values
    actual=probabilities@np.sum((values-mean)**2,axis=1)
    identity=e_noise*(1-pi**2-(1-pi)**2)+(e_shrink2-e_shrink**2)*2*(pi-p)**2
    assert actual==pytest.approx(identity,abs=1e-14)
