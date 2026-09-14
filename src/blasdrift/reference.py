"""Independent, BLAS-free high-precision reference implementations.

Every fixture in this project is checked against these functions, which
deliberately avoid numpy's optimized dot/matmul/sum paths (the exact code
paths that vary across BLAS backends -- Accelerate, OpenBLAS, MKL -- and
across CPU microarchitectures within a single backend). Instead they use
Python's arbitrary-precision `fractions.Fraction` for exact rational
arithmetic, so the reference answer is correct regardless of which BLAS
library, thread count, or SIMD kernel the *installed* numpy happens to
pick on a given machine.

This mirrors the exact reproduction technique used in the real-world
bug reports this project is built from (numpy#30136, numpy#29393,
numpy#11655, numpy#20564): a hand-written, order-independent computation
used to characterize how a BLAS-accelerated operation drifts from the
"true" mathematical answer.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Sequence


def _to_fractions(values: Sequence[float]) -> list:
    return [Fraction(v) for v in values]


def exact_dot(a: Sequence[float], b: Sequence[float]) -> Fraction:
    """Exact dot product of two equal-length real vectors.

    Uses Python's native ``float`` -> ``Fraction`` conversion, which is
    exact (every finite float is itself a dyadic rational), then sums
    products left-to-right in exact rational arithmetic -- no rounding
    at any step, so the result is the true mathematical answer for the
    literal float64 inputs given, independent of any BLAS accumulation
    order.
    """
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    fa = _to_fractions(a)
    fb = _to_fractions(b)
    total = Fraction(0)
    for x, y in zip(fa, fb):
        total += x * y
    return total


def exact_matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list:
    """Exact matrix-vector product, row by row, via ``exact_dot``."""
    return [exact_dot(row, vector) for row in matrix]


def exact_sum(values: Sequence[float]) -> Fraction:
    """Exact sum of a sequence of floats (left-to-right, exact rationals)."""
    total = Fraction(0)
    for v in values:
        total += Fraction(v)
    return total


def fraction_to_float(value: Fraction) -> float:
    """Round an exact Fraction to the nearest representable float64.

    Uses the numerator/denominator directly rather than ``float(value)``
    (which is equivalent here but spelled out for clarity/testability),
    relying on CPython's correctly-rounded Fraction.__float__.
    """
    return float(value)
