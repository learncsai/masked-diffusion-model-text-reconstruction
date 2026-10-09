"""Check the proposed rule at the boundary and against method-axis changes."""
import numpy as np
from scripts.summarize_bootstrap_confirmation import switching_forecast


def test_switch_preserves_banks_and_routes_equality_to_multinomial():
    methods = ['bootstrap_jackknife', 'unused', 'multinomial_jackknife']
    f = np.empty((2, 2, 3, 3))
    f[..., 0], f[..., 1], f[..., 2] = 10., 99., 1.
    f[1] *= 2
    actual = switching_forecast(f, methods, [1024, 2048], [512, 1024, 2048])
    np.testing.assert_array_equal(actual[0], [[10, 1, 1], [10, 10, 1]])
    np.testing.assert_array_equal(actual[1], 2*actual[0])
