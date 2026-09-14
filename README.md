# blasdrift

Detect floating-point drift between numpy's BLAS-backed operations
(`np.dot`, `np.sum`, matrix-vector products) and an independent,
BLAS-free exact reference -- across BLAS backends (Apple Accelerate,
OpenBLAS, Intel MKL) and CPU microarchitectures.

## The problem

`numpy.dot`, `numpy.sum`, and friends do **not** guarantee bit-identical
results across machines. This is IEEE-754-legal (floating-point addition
is not associative, and fused multiply-add is an allowed optimization)
but it silently breaks real projects:

- **numpy#30136**: `numpy.dot()` returned different results on Linux
  (OpenBLAS) vs macOS ARM (Accelerate) for the same float64 inputs,
  breaking QMCPACK's test suite via a downstream `numpy.floor()` call.
- **numpy#29393**: `numpy.dot` on float32 matrices returned all zeros
  under OpenBLAS's SME-specific SGEMM kernel on Apple M4, but worked
  correctly under Accelerate on the same machine.
- **numpy#20564**: `numpy.dot` produced two distinct outputs depending
  on which Intel Xeon CPU model ran it, because different CPU
  microarchitectures dispatch to different vectorized BLAS kernels.
- **numpy#11655 / numpy#29933**: dot-product precision depends on
  thread count, because OpenBLAS/MKL parallelize the reduction
  differently across thread counts, changing the accumulation order.
- **FPRev (USENIX ATC 2025, arXiv:2411.00442)**: formalizes this whole
  class as "accumulation order is undocumented and backend-specific,"
  and builds a black-box probe to recover it.

If your ML pipeline, scientific code, or test suite assumes `np.dot`
gives the same answer on your Mac laptop and your Linux CI/production
box, it doesn't -- and the difference is usually silent until a
`floor()`, an exact-equality assertion, or a downstream branch on a
tiny numerical difference blows up in one environment but not the
other.

## What blasdrift does

1. Runs a small set of adversarial fixtures -- each one a fresh
   reconstruction of a real, filed bug's failure *shape*, not an
   invented worst case -- through numpy's BLAS-backed `np.dot`/`np.sum`.
2. Computes the same fixture through an independent, BLAS-free exact
   reference using Python's arbitrary-precision `fractions.Fraction`
   (which cannot depend on BLAS backend, thread count, or CPU kernel
   dispatch, because it never calls into compiled numpy math at all).
3. Reports the ULP (unit-in-last-place) distance and absolute error
   between the two, with an explicit verdict: `exact`,
   `within_tolerance` (ordinary FP rounding), or `drift_detected`
   (bigger than ordinary rounding -- worth investigating).
4. Optionally checks whether a fixture's result is sensitive to thread
   count alone (`--thread-sensitivity`), reproducing the numpy#11655 /
   numpy#29933 class directly.

## Install

Not yet published to PyPI. Install the latest GitHub Release wheel directly
(checksum-verified, CI-built):

```bash
pip install https://github.com/zhuhroscar-tech/blasdrift/releases/latest/download/blasdrift-0.1.0-py3-none-any.whl
```

Or from source:

```bash
git clone https://github.com/zhuhroscar-tech/blasdrift.git
cd blasdrift && pip install -e ".[dev]"
```

## Usage

```bash
# Run all fixtures, human-readable output
blasdrift

# Machine-readable JSON (for CI / scripting)
blasdrift --json

# Run a single fixture
blasdrift --fixture qmcpack_near_zero_dot

# List available fixtures and their real-bug sources
blasdrift --list-fixtures

# Check if a result is sensitive to thread count alone
blasdrift --thread-sensitivity qmcpack_near_zero_dot

# CI-friendly: exit 1 if any fixture drifts beyond tolerance
blasdrift --check-drift
```

## What this tool is NOT

- **Not a claim that BLAS is buggy.** Every discrepancy blasdrift finds
  is IEEE-754-legal floating-point behavior. The tool's job is to make
  an *undocumented, silent* difference *visible and quantified* before
  it surfaces as a confusing test failure on a different machine.
- **Not a fix.** blasdrift does not patch numpy, force a specific BLAS
  backend, or change accumulation order. If you need bit-reproducible
  results, pin your BLAS backend and thread count explicitly (the
  `--thread-sensitivity` check tells you whether thread count alone
  already matters for your case) -- blasdrift only tells you where and
  by how much drift exists.
- **Not exhaustive.** Four fixtures cover four distinct, real, filed
  bug classes. They are not a general floating-point fuzzer. Real
  numerical code in your own project may drift in ways these fixtures
  don't probe.

## Verification

- `pytest --cov` on this repo: unit tests for the exact reference
  (order-independence, exact rational arithmetic), fixture metadata and
  reproducibility, ULP-distance edge cases (adjacent floats, sign
  boundary, NaN/inf), and the CLI (JSON output, `--no-color`,
  `--check-drift` exit codes).
- CI runs on both `ubuntu-latest` and `macos-latest` GitHub Actions
  runners (matching this project's actual cross-platform BLAS claim --
  Accelerate is only linked on the macOS runner, OpenBLAS on the Linux
  runner) across Python 3.9 and 3.12.
- Released wheels/sdists are checksummed (`SHA256SUMS.txt`) and smoke
  tested from a clean venv as part of CI.

## License

MIT
