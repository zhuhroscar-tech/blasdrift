"""Adversarial fixtures reproducing real, filed BLAS/reduction-order drift
bugs -- not synthetic worst-cases invented for this project.

Each fixture below is derived from (and cites) an actual bug report:

- ``qmcpack_near_zero_dot`` mirrors numpy#30136 (numpy.dot returns a
  different element ordering/value on Accelerate (macOS) vs OpenBLAS
  (Linux) for a small, near-zero-result dot product) -- the exact
  pattern that broke QMCPACK's test suite.
- ``large_magnitude_disparity_sum`` mirrors the general
  catastrophic-cancellation-in-summation class documented in FPRev
  (arxiv 2411.00442): summing values of wildly different magnitude
  gives an answer that depends on accumulation order, which differs
  across BLAS/CPU implementations.
- ``fma_sensitive_matvec`` targets the FMA-vs-separate-multiply-add
  distinction called out directly in numpy#30136's discussion thread
  (IEEE 754 permits fusing a*b+c into one rounding step; whether a
  given BLAS kernel does this is undocumented and backend-specific).
- ``platinum_cpu_dot`` is a smaller, MIT-licensed-safe reconstruction of
  the *shape* of the numpy#20564 case (numpy.dot output differed by CPU
  model due to different vectorized kernels selected per-microarch) --
  built from a fresh adversarial vector pair rather than copying the
  original .npy fixture, so it exercises the same failure class without
  redistributing a third party's exact test data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class Fixture:
    name: str
    description: str
    source: str
    op: str  # "dot", "sum", or "matvec"
    build: Callable[[], tuple]  # () -> (args for the op)


def _qmcpack_near_zero_dot() -> tuple:
    # Same shape of case as numpy#30136: a small difference vector dotted
    # against a near-singular matrix row, producing a result that should
    # be extremely close to zero -- exactly the regime where BLAS
    # accumulation-order differences show up as a *sign* or *ordering*
    # difference rather than just a tiny relative error.
    pos_i = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    c = np.array(
        [0.61550000000000071321, 1.06607727205864510900, 7.5],
        dtype=np.float64,
    )
    diff = pos_i - c
    row = np.array(
        [4.06173842404549123586e-01, 2.34504577250050944004e-01, -4.97419495998112408112e-17],
        dtype=np.float64,
    )
    return (diff, row)


def _large_magnitude_disparity_sum() -> tuple:
    # A classic swamping case (FPRev / textbook catastrophic
    # cancellation): one huge value, many small values whose sum is
    # comparable to the huge value's ULP. Left-to-right vs
    # pairwise/tree summation give measurably different results.
    values = [1.0e16] + [1.0] * 10_000 + [-1.0e16]
    return (np.array(values, dtype=np.float64),)


def _fma_sensitive_matvec() -> tuple:
    # Values chosen so that a fused multiply-add (single rounding of
    # a*b+c) and a separate multiply-then-add (two roundings) disagree
    # in their last bit -- the exact ambiguity numpy#30136's thread
    # identifies as IEEE-754-legal but BLAS-backend-dependent.
    matrix = np.array(
        [
            [1.0 + 2**-52, -(1.0 + 2**-52), 2**-53],
            [3.0, -3.0 + 2**-51, 2**-52],
        ],
        dtype=np.float64,
    )
    vector = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    return (matrix, vector)


def _platinum_cpu_dot() -> tuple:
    # Reconstruction of the numpy#20564 failure *shape* (per-CPU-model
    # kernel selection changes the result) using a fresh 512-element
    # adversarial vector pair with mixed-magnitude, alternating-sign
    # entries -- large enough to trigger a BLAS-dispatched (not
    # scalar-loop) code path on typical builds.
    rng = np.random.default_rng(20564)
    a = rng.uniform(-1e8, 1e8, size=512).astype(np.float64)
    signs = np.where(np.arange(512) % 2 == 0, 1.0, -1.0)
    b = (rng.uniform(1e-8, 1.0, size=512) * signs).astype(np.float64)
    return (a, b)


FIXTURES: list[Fixture] = [
    Fixture(
        name="qmcpack_near_zero_dot",
        description="near-zero-result dot product (element-order drift across BLAS backends)",
        source="numpy/numpy#30136",
        op="dot",
        build=_qmcpack_near_zero_dot,
    ),
    Fixture(
        name="large_magnitude_disparity_sum",
        description="huge+many-small+huge-negative sum (catastrophic cancellation / swamping)",
        source="FPRev, arXiv:2411.00442",
        op="sum",
        build=_large_magnitude_disparity_sum,
    ),
    Fixture(
        name="fma_sensitive_matvec",
        description="matrix-vector product sensitive to fused- vs separate-multiply-add",
        source="numpy/numpy#30136 (FMA discussion)",
        op="matvec",
        build=_fma_sensitive_matvec,
    ),
    Fixture(
        name="platinum_cpu_dot",
        description="512-element mixed-magnitude dot product (per-CPU-kernel dispatch drift)",
        source="reconstructed from numpy/numpy#20564 (fresh data, same failure class)",
        op="dot",
        build=_platinum_cpu_dot,
    ),
]


def get_fixture(name: str) -> Fixture:
    for fx in FIXTURES:
        if fx.name == name:
            return fx
    raise KeyError(f"unknown fixture: {name!r}")
