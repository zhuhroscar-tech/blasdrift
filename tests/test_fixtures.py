import numpy as np
import pytest

from blasdrift.fixtures import FIXTURES, get_fixture


def test_all_fixtures_have_required_metadata():
    for fx in FIXTURES:
        assert fx.name
        assert fx.description
        assert fx.source
        assert fx.op in ("dot", "sum", "matvec")


def test_fixture_names_are_unique():
    names = [fx.name for fx in FIXTURES]
    assert len(names) == len(set(names))


def test_get_fixture_by_name():
    fx = get_fixture("qmcpack_near_zero_dot")
    assert fx.name == "qmcpack_near_zero_dot"


def test_get_fixture_unknown_raises():
    with pytest.raises(KeyError):
        get_fixture("does_not_exist")


def test_qmcpack_fixture_shape_matches_dot_op():
    fx = get_fixture("qmcpack_near_zero_dot")
    diff, row = fx.build()
    assert diff.shape == row.shape
    assert diff.dtype == np.float64


def test_large_magnitude_disparity_sum_reproduces_known_swamping():
    fx = get_fixture("large_magnitude_disparity_sum")
    (values,) = fx.build()
    # Sanity: naive left-to-right float64 summation of this exact
    # fixture is known to swamp the 10,000 small terms entirely.
    total = 0.0
    for v in values:
        total += float(v)
    assert total == 0.0


def test_fma_sensitive_matvec_returns_valid_shapes():
    fx = get_fixture("fma_sensitive_matvec")
    matrix, vector = fx.build()
    assert matrix.shape[1] == vector.shape[0]


def test_platinum_cpu_dot_is_deterministic_across_calls():
    fx = get_fixture("platinum_cpu_dot")
    a1, b1 = fx.build()
    a2, b2 = fx.build()
    # Same seeded RNG each call -> identical fixture data (not testing
    # numpy's dot itself, just that our fixture generator is reproducible).
    assert np.array_equal(a1, a2)
    assert np.array_equal(b1, b2)
