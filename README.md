[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# blasdrift

Measure floating-point drift between NumPy operations and an independent, BLAS-free exact reference. Use it to compare numerical behavior across BLAS backends, CPU architectures, and thread counts—not to force bit-identical results.

## What it reports

- Adversarial fixtures reconstructed from reported numerical failure patterns.
- An exact rational reference built with Python's `fractions.Fraction`.
- Absolute error, ULP distance, and `exact`, `within_tolerance`, or `drift_detected` verdicts.
- Human-readable or JSON output, plus an optional thread-sensitivity probe.

NumPy reductions and dot products can differ with accumulation order and backend implementation. A difference is not, by itself, proof of a BLAS bug; small rounding changes can still affect downstream equality checks or `floor()` calls.

## Install

Requires Python 3.9+ and NumPy 1.24+. Install from source:

```bash
git clone https://github.com/zhuhroscar-tech/blasdrift.git
cd blasdrift
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The `dev` extra includes pytest, coverage support, and `threadpoolctl` for backend inspection and thread control. Release wheels are available from [Releases](https://github.com/zhuhroscar-tech/blasdrift/releases); check the accompanying `SHA256SUMS.txt` before installation.

## Quick start

```bash
blasdrift
blasdrift --json
blasdrift --list-fixtures
blasdrift --fixture qmcpack_near_zero_dot
blasdrift --thread-sensitivity qmcpack_near_zero_dot
blasdrift --check-drift
```

`--check-drift` exits `1` if a selected probe reports `drift_detected`, otherwise `0`. Without that flag, ordinary probe results do not cause a failure exit. The separate thread-sensitivity mode reports its status rather than acting as a failure gate; inspect its output when scripting it.

## Scope and development

The fixture set is small, not an exhaustive floating-point fuzzer. Results characterize these inputs on the current environment; they do not guarantee reproducibility for your application. Keep application-specific numerical tests, and explicitly control the BLAS backend and thread count where reproducibility matters. blasdrift does not patch NumPy or change its algorithms.

Run `python -m pytest --cov` after installing the development extra. See [fixtures](src/blasdrift/fixtures.py) for source references and [CI](.github/workflows/ci.yml) for the Linux/macOS test and artifact-build workflow.

## License

[MIT](LICENSE).
