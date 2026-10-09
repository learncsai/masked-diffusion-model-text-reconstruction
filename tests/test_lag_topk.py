"""Check full-vocabulary ranking, ties, and the chunk-weighted metric."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_lag_topk import target_ranks, prediction_ranks, chunk_mean, relative_error_reduction
from diffusion_lm_rmt.sparse_categorical_lag import Prediction


def test_ranks_match_independent_full_sort_with_signed_scores_and_ties():
    rng=np.random.default_rng(42)
    scores=rng.integers(-3,4,size=(80,29)).astype(float)
    targets=rng.integers(0,29,size=80)
    ranks,best,worst=target_ranks(scores,targets)
    expected=[]
    for row,target in zip(scores,targets):
        order=sorted(range(len(row)),key=lambda j:(-row[j],j))
        expected.append(order.index(target)+1)
    np.testing.assert_array_equal(ranks,expected)
    assert np.all(best<=ranks) and np.all(ranks<=worst)


def test_sparse_plus_prior_ranks_include_zero_negative_and_positive_prior_weights():
    prior=np.array([.4,.3,.2,.1])
    residual=np.array([[0,0,.5,0],[-.6,.1,0,0],[.2,0,.2,0]])
    prediction=Prediction(sparse.csr_matrix(residual),np.array([1.,-1.,0.]),prior)
    targets=np.array([2,3,2])
    actual=prediction_ranks(prediction,targets,batch_rows=2)
    expected=target_ranks(prediction.toarray(),targets)
    for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)


def test_chunk_average_does_not_overweight_chunks_with_more_masked_tokens():
    rows=np.array([0,1,1,1]);hits=np.array([True,False,False,True])
    actual=chunk_mean(hits,rows,2)
    np.testing.assert_allclose(actual,[1.,1/3])
    assert actual.mean()==pytest.approx(2/3)
    assert actual.mean()!=pytest.approx(hits.mean())


def test_relative_error_reduction_uses_errors_and_handles_perfect_baseline():
    assert relative_error_reduction(.4,.2)==pytest.approx(.25)
    assert relative_error_reduction(.1,.2)==pytest.approx(-.125)
    assert relative_error_reduction(1.,1.) is None


def test_ties_have_a_fixed_nested_order_and_explicit_bounds():
    ranks,best,worst=target_ranks(np.zeros((3,4)),np.array([0,1,3]))
    np.testing.assert_array_equal(ranks,[1,2,4])
    np.testing.assert_array_equal(best,[1,1,1])
    np.testing.assert_array_equal(worst,[4,4,4])
