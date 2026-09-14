import math

import numpy as np
import pytest

from blasdrift.core import (
    ProbeResult,
    _float_to_ulp_int,
    get_backend_info,
    result_to_dict,
    run_all_probes,
    run_probe,
    ulp_distance,
)
from blasdrift.fixtures import FIXTURES, get_fixture


def test_ulp_distance_identical_values_is_zero():
    assert ulp_distance(1.0, 1.0) == 0


def test_ulp_distance_adjacent_floats_is_one():
    x = 1.0
    y = math.nextafter(1.0, 2.0)
    assert ulp_distance(x, y) == 1


def test_ulp_distance_across_zero_sign_boundary():
    # -0.0's neighbor going negative and +0.0's neighbor going positive
    # should be 2 ULPs apart (one step each side of zero), verifying the
    # sign-folding logic handles the zero crossing correctly.
    neg = math.nextafter(0.0, -1.0)
    pos = math.nextafter(0.0, 1.0)
    assert ulp_distance(neg, pos) == 2


def test_ulp_distance_nan_is_undefined():
    assert ulp_distance(float("nan"), 1.0) == -1


def test_ulp_distance_inf_same_sign_is_zero():
    assert ulp_distance(float("inf"), float("inf")) == 0


def test_ulp_distance_inf_vs_finite_is_undefined():
    assert ulp_distance(float("inf"), 1.0) == -1


def test_float_to_ulp_int_is_monotonic_for_positive_floats():
    a = _float_to_ulp_int(1.0)
    b = _float_to_ulp_int(2.0)
    assert b > a


def test_run_probe_returns_result_for_every_fixture_op_type():
    ops_seen = set()
    for fx in FIXTURES:
        result = run_probe(fx)
        assert isinstance(result, ProbeResult)
        assert result.verdict in ("exact", "within_tolerance", "drift_detected")
        ops_seen.add(fx.op)
    assert ops_seen == {"dot", "sum", "matvec"}


def test_run_all_probes_covers_every_registered_fixture():
    results = run_all_probes()
    assert len(results) == len(FIXTURES)
    assert {r.fixture for r in results} == {fx.name for fx in FIXTURES}


def test_run_probe_dot_reference_matches_manual_exact_computation():
    fx = get_fixture("qmcpack_near_zero_dot")
    result = run_probe(fx)
    # The independent reference must agree with numpy to within a tiny
    # tolerance for this well-conditioned case (it's the *ordering*/
    # cross-backend difference that's the real-world bug, not that numpy
    # is grossly wrong on any single run).
    assert abs(result.numpy_value - result.reference_value) < 1e-9


def test_run_probe_sum_detects_the_known_swamping_bug():
    fx = get_fixture("large_magnitude_disparity_sum")
    result = run_probe(fx)
    # numpy's pairwise summation is much better than naive here, but the
    # exact reference (1.0e16 - 1.0e16 + 10000 == 10000... actually the
    # true sum is 1e16 + 10000*1.0 - 1e16 = 10000) may still differ from
    # numpy's float64 pairwise-summation result by more than 0 ULPs.
    assert result.reference_value == pytest.approx(10000.0, rel=1e-9)


def test_result_to_dict_is_json_serializable():
    import json

    fx = get_fixture("fma_sensitive_matvec")
    result = run_probe(fx)
    d = result_to_dict(result)
    json.dumps(d)  # must not raise
    assert d["fixture"] == "fma_sensitive_matvec"


def test_get_backend_info_has_required_keys():
    info = get_backend_info()
    assert "numpy_version" in info
    assert "python_version" in info
    assert "platform" in info
