# Validation and reproduction

Use `uv sync --locked`, then `uv run pytest`. Tests require neither R nor the
reference directory. They compare independent five-update fixtures for all six
models (both binary methods), with JIT enabled and disabled, using
`atol=1e-8, rtol=1e-6`. Additional tests cover bootstrap, anchoring, input
validation, missing/constant observations, zero counts and extreme normal tails.

## Full datasets

| Model | Dataset | Size | Current evidence |
| --- | --- | --- | --- |
| Binary | Senate 109 | Full supplied votes | EM and variational parameters/variances match; 64/58 iterations |
| Ordinal | Asahi–Todai | 19,443 × 98 | Parameters/variances match; 170 iterations |
| Dynamic | Judicial voting | 45 × 5,164 | Parameters/variances match; 51 iterations |
| Hierarchical | Senate grouped voting | 1,879,731 observations | Parameters/variances match; 73 iterations |
| Network | US Twitter | 10,000 × 176 | Parameters/variances match at 500 iterations; neither fit converged |
| Poisson | Manifesto | 8,995 × 25 | Parameters/variances match; 649 iterations |

The consolidated full-data audit passes for all six models, including the
ordinal tail-precision improvement. Every check compares all returned means and
variances, iteration counts and convergence status. The 60-test suite also passes.
[validation-results.json](validation-results.json) records the audit output and
SHA-256 hashes of the tested source and baselines. Large input files and detailed
reports under `outputs/` are local artifacts and are not shipped in the package.

To reproduce all inputs and independent baselines from a fresh checkout with
the reference material available locally:

```sh
sh scripts/build_reference.sh
sh scripts/prepare_real_baselines.sh
uv run python scripts/validate_real_data.py
```

The build script compiles a copy under `outputs/reference-build/`, with R
packages installed under `outputs/r-library/`. The original source is read-only.
The hierarchical baseline is slow because it repeatedly scans all observations.
Do not start duplicate runs while a baseline process is active.

## Numerical conventions

- Floating-point comparisons use CPU float64; no bitwise equality is required.
- Constant-vector convergence uses maximum absolute change when correlation is
  undefined. Fixed gamma is excluded from network convergence; its default is
  absolute change, consistent with the executable network example.
- Log-CDF formulas stabilize truncated normal tails. Ordinal uses a higher-order
  tail expansion so moments remain accurate at predictors near ±30.
- Missing binary and ordinal responses retain untruncated latent expectations.
- Ordinal preserves the source algorithm's integer Gamma shape and lagged
  second-moment update. It implements three response categories.
- Dynamic results are zero outside each subject's inclusive tenure. Unlike the
  original contiguous-period aggregation, indexed aggregation also accepts
  unsorted items and periods with no observations.
- Network gamma remains fixed; Newton updates can encounter invalid curvature
  on unsuitable initial values. Check returned variances and convergence.
- Bootstrap retains finite nonconverged fits and exposes their status. Failed
  replicates are excluded from sample SD and reported explicitly; fewer than
  two successful fits produce NaN standard errors.

## Performance and packaging

`uv run python scripts/benchmark.py` measures compilation separately from
30 synchronized update calls for each small fixture and writes
`outputs/benchmark.json`. These timings describe the local CPU and fixture
sizes; they are not claims about GPU speed or large-data scaling.

`uv build` produces a wheel and source distribution. The wheel has been
installed into a separate uv environment under `/tmp`, and all six model examples were run from
outside the project with isolated Python imports. The CI workflow repeats
linting, tests, all-model examples and package construction on Linux.
