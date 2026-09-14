import math
from fractions import Fraction

import pytest

from blasdrift import reference


def test_exact_dot_matches_naive_float_for_exact_cases():
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    assert reference.exact_dot(a, b) == Fraction(32)


def test_exact_dot_length_mismatch_raises():
    with pytest.raises(ValueError):
        reference.exact_dot([1.0, 2.0], [1.0])


def test_exact_dot_is_order_independent_by_construction():
    # Exact rational arithmetic means summation order literally cannot
    # matter -- verify by reversing the vectors and confirming identical
    # results (this is the core guarantee the whole tool depends on).
    a = [1e16, 1.0, -1e16]
    b = [1.0, 1.0, 1.0]
    forward = reference.exact_dot(a, b)
    backward = reference.exact_dot(list(reversed(a)), list(reversed(b)))
    assert forward == backward


def test_exact_sum_catches_true_answer_despite_cancellation():
    # 1e16 + 1.0 - 1e16 should be exactly 1.0 mathematically; a naive
    # float64 left-to-right sum rounds the middle term away entirely.
    values = [1e16, 1.0, -1e16]
    exact = reference.exact_sum(values)
    assert exact == Fraction(1)
    naive_float_sum = 0.0
    for v in values:
        naive_float_sum += v
    assert naive_float_sum == 0.0  # demonstrates the float64 bug this exists to catch
    assert reference.fraction_to_float(exact) == 1.0


def test_exact_matvec_shapes_and_values():
    matrix = [[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]]
    vector = [5.0, 7.0]
    result = reference.exact_matvec(matrix, vector)
    assert result == [Fraction(5), Fraction(7), Fraction(31)]


def test_fraction_to_float_roundtrip():
    f = Fraction(1, 3)
    as_float = reference.fraction_to_float(f)
    assert math.isclose(as_float, 1.0 / 3.0, rel_tol=1e-15)
