"""Core probes: run each fixture through numpy's BLAS-backed path and an
independent, BLAS-free exact reference, then quantify the drift.
"""
from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from . import reference
from .fixtures import FIXTURES, Fixture, get_fixture


def _float_to_ulp_int(x: float) -> int:
    """Map a float64 to a monotonic integer ordering (two's-complement
    style sign folding) so that ULP distance is a plain integer subtraction.
    """
    (bits,) = struct.unpack("<q", struct.pack("<d", x))
    if bits < 0:
        bits = -0x8000000000000000 - bits
    return bits


def ulp_distance(a: float, b: float) -> int:
    """Number of representable float64 values strictly between a and b
    (0 if bit-identical). NaN/inf pairs return -1 (undefined distance)."""
    if math.isnan(a) or math.isnan(b):
        return -1
    if math.isinf(a) or math.isinf(b):
        return -1 if a != b else 0
    return abs(_float_to_ulp_int(a) - _float_to_ulp_int(b))


@dataclass
class ProbeResult:
    fixture: str
    description: str
    source: str
    op: str
    numpy_value: object
    reference_value: object
    max_abs_error: float
    max_ulp_distance: int
    verdict: str  # "exact", "within_tolerance", "drift_detected"


_ULP_TOLERANCE = 4  # values within this many ULPs are ordinary FP rounding,
                     # not the kind of cross-backend drift this tool flags.


def _compare_scalar(numpy_val: float, ref_val) -> tuple:
    ref_float = reference.fraction_to_float(ref_val)
    abs_err = abs(numpy_val - ref_float)
    ulp = ulp_distance(numpy_val, ref_float)
    return ref_float, abs_err, ulp


def _compare_vector(numpy_vals, ref_vals) -> tuple:
    ref_floats = [reference.fraction_to_float(v) for v in ref_vals]
    abs_errs = [abs(float(n) - r) for n, r in zip(numpy_vals, ref_floats)]
    ulps = [ulp_distance(float(n), r) for n, r in zip(numpy_vals, ref_floats)]
    return ref_floats, max(abs_errs), max(ulps)


def run_probe(fixture: Fixture) -> ProbeResult:
    args = fixture.build()
    if fixture.op == "dot":
        a, b = args
        numpy_val = float(np.dot(a, b))
        ref_val = reference.exact_dot(a, b)
        ref_float, abs_err, ulp = _compare_scalar(numpy_val, ref_val)
        numpy_out, ref_out = numpy_val, ref_float
    elif fixture.op == "sum":
        (values,) = args
        numpy_val = float(np.sum(values))
        ref_val = reference.exact_sum(values)
        ref_float, abs_err, ulp = _compare_scalar(numpy_val, ref_val)
        numpy_out, ref_out = numpy_val, ref_float
    elif fixture.op == "matvec":
        matrix, vector = args
        numpy_val = np.dot(matrix, vector)
        ref_val = reference.exact_matvec(matrix, vector)
        ref_floats, abs_err, ulp = _compare_vector(numpy_val, ref_val)
        numpy_out, ref_out = [float(x) for x in numpy_val], ref_floats
    else:
        raise ValueError(f"unknown op: {fixture.op}")

    if ulp < 0:
        verdict = "drift_detected"  # NaN/inf mismatch is always significant
    elif ulp == 0:
        verdict = "exact"
    elif ulp <= _ULP_TOLERANCE:
        verdict = "within_tolerance"
    else:
        verdict = "drift_detected"

    return ProbeResult(
        fixture=fixture.name,
        description=fixture.description,
        source=fixture.source,
        op=fixture.op,
        numpy_value=numpy_out,
        reference_value=ref_out,
        max_abs_error=abs_err,
        max_ulp_distance=ulp,
        verdict=verdict,
    )


def run_all_probes() -> list:
    return [run_probe(fx) for fx in FIXTURES]


def get_backend_info() -> dict:
    """Best-effort identification of the active BLAS backend, without
    requiring threadpoolctl (kept as a soft optional dependency check)."""
    info = {
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "machine": os.uname().machine if hasattr(os, "uname") else "unknown",
    }
    try:
        import threadpoolctl  # type: ignore

        pools = threadpoolctl.threadpool_info()
        blas_pools = [p for p in pools if p.get("user_api") == "blas"]
        if blas_pools:
            info["blas_backends"] = [
                {"internal_api": p.get("internal_api"), "version": p.get("version"), "num_threads": p.get("num_threads")}
                for p in blas_pools
            ]
        else:
            # threadpoolctl ran successfully but reported zero BLAS pools.
            # This is the normal, expected result on Apple's Accelerate
            # framework (numpy's default BLAS on macOS since numpy>=1.26):
            # threadpoolctl only instruments OpenBLAS/MKL/BLIS internals,
            # not Accelerate's opaque vecLib. An empty list here does NOT
            # mean "no BLAS backend" -- numpy always has one -- so it must
            # never be returned bare; that reads as a confirmed absence
            # rather than "this backend isn't introspectable this way".
            info["blas_backends"] = (
                "not detected by threadpoolctl (likely Apple Accelerate/vecLib, "
                "which threadpoolctl does not instrument)"
            )
    except ImportError:
        info["blas_backends"] = "unknown (install threadpoolctl for details)"
    return info


def thread_sensitivity_check(fixture_name: str, timeout: float = 30.0) -> dict:
    """Run a fixture's numpy op in two fresh subprocesses -- one with
    OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS forced to 1, one
    left at the environment default -- and report whether the BLAS result
    itself changes with thread count (a real, documented source of
    non-reproducibility distinct from cross-machine BLAS differences).
    """
    fx = get_fixture(fixture_name)
    script = (
        "import json,sys; sys.path.insert(0, %r)\n"
        "from blasdrift.core import run_probe\n"
        "from blasdrift.fixtures import get_fixture\n"
        "r = run_probe(get_fixture(%r))\n"
        "print(json.dumps({'numpy_value': r.numpy_value}))\n"
    ) % (os.path.join(os.path.dirname(__file__), "..", "..", "src"), fixture_name)

    def _run(env_overrides: dict) -> Optional[dict]:
        env = dict(os.environ)
        env.update(env_overrides)
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return None
        if proc.returncode != 0:
            return None
        try:
            return json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            return None

    single = _run({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    default = _run({})

    if single is None or default is None:
        return {
            "fixture": fixture_name,
            "status": "inconclusive",
            "reason": "one or both subprocess runs failed or timed out",
        }

    single_val = single["numpy_value"]
    default_val = default["numpy_value"]
    if single_val == default_val:
        return {"fixture": fixture_name, "status": "stable", "single_thread": single_val, "default_threads": default_val}
    return {
        "fixture": fixture_name,
        "status": "thread_sensitive",
        "single_thread": single_val,
        "default_threads": default_val,
    }


def result_to_dict(result: ProbeResult) -> dict:
    return asdict(result)
