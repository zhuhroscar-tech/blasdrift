"""Command-line interface for blasdrift."""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .core import get_backend_info, result_to_dict, run_all_probes, run_probe, thread_sensitivity_check
from .fixtures import FIXTURES, get_fixture
from .style import bool_badge, print_fields, resolve_style, section, status_headline


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="blasdrift",
        description="Detect floating-point drift between numpy's BLAS-backed "
        "operations and an independent, BLAS-free exact reference -- across "
        "backends (Accelerate/OpenBLAS/MKL) and thread counts.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--no-color", action="store_true", help="disable ANSI color output")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    p.add_argument(
        "--fixture",
        choices=[fx.name for fx in FIXTURES],
        help="run a single named fixture instead of all fixtures",
    )
    p.add_argument("--list-fixtures", action="store_true", help="list available fixtures and exit")
    p.add_argument(
        "--thread-sensitivity",
        metavar="FIXTURE",
        choices=[fx.name for fx in FIXTURES],
        help="check whether a fixture's result changes between 1 thread and default thread count",
    )
    p.add_argument(
        "--check-drift",
        action="store_true",
        help="exit non-zero if any fixture shows drift_detected (for CI use)",
    )
    return p


def _print_backend_info(style) -> None:
    info = get_backend_info()
    section(style.bold("Environment"))
    rows = [
        ("numpy", info["numpy_version"]),
        ("python", info["python_version"]),
        ("platform", info["platform"]),
        ("machine", info["machine"]),
    ]
    backends = info.get("blas_backends")
    if isinstance(backends, list) and backends:
        for b in backends:
            rows.append(("blas backend", f"{b['internal_api']} {b['version']} (threads={b['num_threads']})"))
    elif isinstance(backends, list):
        # Empty list: threadpoolctl ran but found no BLAS pool (e.g. Apple
        # Accelerate isn't instrumented by it). Never let this collapse to
        # zero rows -- that reads as "no backend info was even attempted".
        rows.append(("blas backend", "not detected by threadpoolctl (see --json for detail)"))
    else:
        rows.append(("blas backend", str(backends)))
    print_fields(rows)


def _print_result_text(style, result) -> None:
    if result.verdict == "exact":
        level = "ok"
    elif result.verdict == "within_tolerance":
        level = "warn"
    else:
        level = "fail"
    print()
    print(status_headline(style, level, f"{result.fixture} ({result.op})"))
    print_fields(
        [
            ("description", result.description),
            ("source", result.source),
            ("max abs error", f"{result.max_abs_error:.3e}"),
            ("max ULP distance", result.max_ulp_distance if result.max_ulp_distance >= 0 else "undefined (NaN/inf)"),
            ("verdict", result.verdict),
        ]
    )


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    style = resolve_style(no_color_flag=args.no_color)

    if args.list_fixtures:
        for fx in FIXTURES:
            print(f"{fx.name}\t{fx.op}\t{fx.source}\t{fx.description}")
        return 0

    if args.thread_sensitivity:
        result = thread_sensitivity_check(args.thread_sensitivity)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            section(style.bold("Thread sensitivity check"))
            level = "warn" if result["status"] == "thread_sensitive" else "ok" if result["status"] == "stable" else "warn"
            print(status_headline(style, level, f"{result['fixture']}: {result['status']}"))
            for k, v in result.items():
                if k not in ("fixture", "status"):
                    print(f"  {k}: {v}")
        return 0

    if args.fixture:
        results = [run_probe(get_fixture(args.fixture))]
    else:
        results = run_all_probes()

    if args.json:
        print(json.dumps([result_to_dict(r) for r in results], indent=2))
    else:
        _print_backend_info(style)
        for r in results:
            _print_result_text(style, r)
        print()
        drifted = [r for r in results if r.verdict == "drift_detected"]
        if drifted:
            print(status_headline(style, "fail", f"{len(drifted)} of {len(results)} fixtures show drift beyond tolerance"))
        else:
            print(status_headline(style, "ok", f"all {len(results)} fixtures within tolerance"))

    if args.check_drift:
        drifted = [r for r in results if r.verdict == "drift_detected"]
        return 1 if drifted else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
