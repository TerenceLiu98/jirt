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
variances, iteration counts and convergence status. The original 60-test regression suite is retained.
The local report `outputs/real-validation.json` records the audit output and
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

## Logistic calibration

`tests/test_logistic.py` additionally checks Rasch, shared-slope 1PL, 2PL, 3PL
and full 4PL against an independent NumPy/SciPy implementation of the same
Gauss–Hermite marginal likelihood. Both calibrations fix the latent distribution
to N(0,1), use no item priors, and impose identical parameter constraints; the
reference optimizer uses numerical derivatives, while jirt uses JAX derivatives.
Near-population response-pattern frequencies provide deterministic inputs for
fitted-parameter comparisons; random data exercise missingness and JIT agreement.
The tests also cover analytic probability curves, nested-model reductions,
extreme predictors, invalid inputs, and explicit iteration-limit status.

This comparison validates a matched statistical objective; it is not a claim
of agreement with a different probit, joint-MAP or variational estimator.
### Cross-validation with R mirt

The local mirt 1.48.5 source (commit
`211851adae7a6bee580af830d37187f0e81a5ca8`) was built in an ignored
working copy with R 4.5.1. The comparison covers five deterministic synthetic
frequency tables (six items, approximately 10,000 subjects each), plus the
real LSAT7 frequency table (five items, 1,000 subjects) for Rasch, 1PL and 2PL.
These are numerical cross-software checks, not held-out predictive validation.

Both packages use the same 41 Gauss–Hermite nodes and weights, fixed N(0,1)
latent distribution, no item priors, neutral starts, and positive slopes.
Rasch fixes slopes at one; 1PL constrains all slopes equal. In mirt these are
specified through the 2PL item type so its default Rasch variance estimation
does not change the calibration target. mirt's `d` is converted to `b=-d/a`;
`g` corresponds to jirt's `c`. Fitted asymptotes have the same marginal bounds;
mirt lacks jirt's coupled `u-c >= 1e-6` constraint, so its fitted gaps are
checked explicitly. All exported fits satisfy it.

R uses EM without acceleration, `nlminb` M-steps and outer tolerance `1e-9`;
jirt uses SLSQP with mean-loss tolerance `1e-14`. Both report convergence in
all eight cases. These stopping criteria are not numerically interchangeable.
Maximum absolute differences from independently fitted estimates are:

| Dataset / model | Item parameters | Probability curves | Total log likelihood | EAP |
| --- | ---: | ---: | ---: | ---: |
| Synthetic Rasch | 4.55e-8 | 1.14e-8 | 4.37e-11 | 5.65e-9 |
| Synthetic 1PL | 7.01e-7 | 2.78e-7 | 7.13e-10 | 4.93e-8 |
| Synthetic 2PL | 6.62e-6 | 1.77e-6 | 7.76e-9 | 4.70e-6 |
| Synthetic 3PL | 1.55e-3 | 3.50e-4 | 1.89e-6 | 1.34e-4 |
| Synthetic 4PL | 4.99e-2 | 2.56e-2 | 2.23e-4 | 3.79e-3 |
| LSAT7 Rasch | 2.13e-6 | 5.32e-7 | 8.42e-10 | 1.03e-6 |
| LSAT7 1PL | 3.64e-6 | 1.16e-6 | 5.83e-10 | 5.62e-7 |
| LSAT7 2PL | 6.42e-5 | 1.39e-5 | 2.13e-8 | 2.92e-5 |

Probability curves are evaluated at 81 points over [-4,4]. The item-parameter
column is the maximum across a, b, c and u. A shared agreement target of
`1e-4` for parameters, curves and posterior moments, and `1e-6` for total log
likelihood, passes the six Rasch/1PL/2PL cases; **3PL and 4PL do not pass**.
For 4PL, the probability difference is about 2.56 percentage points.
The slightly higher likelihood in jirt does not establish a global optimum.
These fits illustrate sensitivity along a nearly flat likelihood surface;
multistart/global-optimum agreement remains unverified.

At the *same R-exported parameters*, all eight cases agree much more tightly:
maximum probability error `2.22e-15`, total log-likelihood error `3.64e-11`,
EAP error `6.22e-15`, and posterior-variance error `2.17e-15`. This isolates
agreement of the model and scoring calculations from the optimizer differences.
`tests/test_logistic_mirt.py` preserves these eight kernel comparisons and six
independent calibration checks against a committed R fixture. It does not claim
3PL/4PL fitted-parameter parity. R is not required for routine tests.

To regenerate the reference and diagnostic report from the repository root:

```sh
sh scripts/build_mirt_reference.sh
uv run python scripts/prepare_logistic_mirt.py
Rscript --vanilla scripts/export_logistic_mirt.R
uv run python scripts/check_logistic_mirt.py
```

On this macOS installation, building R dependencies required
`MACOSX_DEPLOYMENT_TARGET=14.0` for the first command. Build dependencies and
outputs stay under ignored `outputs/`; `references/` is never modified.
The checker writes `outputs/logistic-mirt/comparison.json` and explicitly lists
fit-agreement exceptions; its successful exit only asserts complete converged
exports and fixed-parameter agreement. Refresh the regression fixture only after
reviewing those exceptions:

```sh
cp outputs/logistic-mirt/reference.json tests/fixtures/logistic_mirt.json
uv run pytest tests/test_logistic_mirt.py
```

The reproducible `examples/logistic.py` demonstrates all five configurations
on synthetic data. General model descriptions and calibration references:

- Bock and Aitkin (1981), *Marginal maximum likelihood estimation of item
  parameters: Application of an EM algorithm*.
  https://doi.org/10.1007/BF02293801
- Barton and Lord (1981), *An Upper Asymptote for the Three-Parameter Logistic
  Item-Response Model*. https://doi.org/10.1002/j.2333-8504.1981.tb01255.x

The current implementation optimizes marginal likelihood directly with SLSQP;
it does not implement the Bock–Aitkin EM update sequence.

### Additional independent packages: ltm and sirt

The same six-item synthetic 3PL/4PL datasets were also evaluated using
**ltm 1.2.0** (`tpm`, 3PL) and **sirt 4.2.133** (`rasch.mml2`, 3PL/4PL).
These runs use the installed packages without modifying their implementation.
They provide additional optimization diagnostics, not a claim that every
package produces identical fitted parameters.

- ltm uses BFGS, 41 Gauss–Hermite points, no item priors and 10,000 maximum
  iterations. The exporter checks that its nodes and weights match jirt after
  sorting. Its public `tpm` wrapper does not forward arbitrary `reltol` values
  to BFGS; the native success flag alone does not certify tight stationarity.
- In the tested sirt version, the one-dimensional normal-distribution update
  overwrites the supplied `trait.weights`. Therefore the retained comparison
  uses 101 equally spaced nodes on [-8,8], explicitly fixes variance to one,
  centers the latent mean at zero and verifies the resulting normal weights.
  The earlier attempted custom-weight runs are excluded from the comparison.
- sirt uses no item priors, `glob.conv=1e-10`, `parm.conv=1e-8`, at most 20,000
  EM iterations, and at most 100 inner difficulty updates. Runs use its default
  `numdiff.parm=1e-5`; sensitivity runs reduce it to `1e-6`. Bounds are set to
  positive slopes and asymptotes in [1e-6,1-1e-6]. sirt also internally limits
  fitted asymptotes using twice its difference step. ltm does not expose the
  same slope/asymptote box bounds; the exported solutions are checked for
  positive slopes and valid asymptote gaps. These are not identical feasible
  sets at their boundaries.

Both R packages start from neutral parameters, the generating parameters,
jirt's best multistart estimate, and the previous mirt estimate. Starting from
jirt is a sensitivity check, **not independent parameter recovery**. jirt also
runs from neutral, generating and deterministically perturbed parameters, and
then refits the neutral-start estimates returned by each applicable R package.

Selected results below compare with the best of the three initial jirt fits.
`Delta LL` is jirt's total log likelihood minus the reference's likelihood,
recomputed on the common 41-point quadrature. Probability differences use 81
points over [-4,4]. All differences are absolute except the signed likelihood
gap. Positive gaps favor jirt on this objective.

| Model / reference run | Delta LL | Max item-parameter difference | Max probability difference | Native status |
| --- | ---: | ---: | ---: | --- |
| 3PL / ltm, neutral | 9.24e-2 | 2.87e-1 | 7.06e-2 | Success |
| 3PL / ltm, jirt start | 3.57e-10 | 1.58e-6 | 3.37e-7 | Success |
| 3PL / sirt, neutral, step 1e-5 | 1.24e-3 | 4.80e-2 | 1.04e-2 | Stopped before limit |
| 3PL / sirt, jirt start, step 1e-6 | 1.29e-5 | 5.00e-3 | 1.09e-3 | Stopped before limit |
| 4PL / sirt, neutral, step 1e-5 | 8.80e-3 | 3.83e-1 | 1.07e-1 | **20,000-iteration limit** |
| 4PL / sirt, jirt start, step 1e-6 | 8.57e-5 | 4.82e-2 | 7.93e-3 | **20,000-iteration limit** |

The sirt grid and 41-point Gauss–Hermite likelihoods are compared at each
exported parameter vector; the observed differences are of order 1e-9, far
smaller than the fitting gaps above. Native reported likelihoods are also
checked against an independent NumPy/SciPy calculation using the native grid.
For sirt, small residual differences can reflect its reported pre-update
E-step likelihood. The checker enforces a 1e-5 absolute consistency bound;
this is not a fitted-parameter agreement test.

There is direct evidence for both numerical stopping effects and weak curvature:

- At neutral-start solutions, the maximum gradient of the *mean* negative
  log likelihood is approximately 1.54e-4 for ltm 3PL, 1.67e-5 for sirt 3PL,
  and 3.78e-5 for sirt 4PL. The corresponding jirt solutions have gradients
  around 1e-8 to 1e-7. Gradients are evaluated in the same `(b,a,c,u)`
  coordinates, not compared across packages' internal transformations.
- sirt's `rasch_mml2_prob_genlogis_4pl_evaluate` uses
  `(p + h/2)/(1 + 2*h)` in the numerical M-step evaluation. Thus its difference
  step affects that calculation beyond derivative discretization alone.
  Reducing the step tenfold reduces the observed 3PL parameter discrepancy
  from about 0.048 to 0.005. This sensitivity is measured without patching sirt.
- The Hessian of jirt's mean negative log likelihood at its neutral-start
  estimates has smallest/largest eigenvalues about `5.23e-5 / 4.99` for 3PL
  and `1.57e-6 / 5.10` for 4PL: condition numbers about `9.5e4` and `3.2e6`
  in these parameter coordinates. These measurements support the explanation
  that small optimization errors can produce larger parameter differences.
- Restarting jirt from ltm's 3PL and sirt's 3PL/4PL neutral estimates recovers
  the previous best jirt likelihood to approximately 1e-10. This strengthens
  the local numerical evidence; it does not prove a global optimum.

The independent packages therefore support the likelihood calculation and
help explain the discrepancy, but **strict independent 3PL/4PL fitted-parameter
parity remains unestablished**, especially for the iteration-limited 4PL runs.

After generating the mirt reference above, reproduce the additional checks:

```sh
# On this macOS installation, prepend MACOSX_DEPLOYMENT_TARGET=14.0 if compiling.
Rscript --vanilla scripts/install_logistic_references.R
uv run python scripts/compare_logistic_packages.py prepare
Rscript --vanilla scripts/export_logistic_packages.R ltm
Rscript --vanilla scripts/export_logistic_packages.R sirt
IRT_COMPARISON_OUTPUT=outputs/logistic-mirt/other-packages-fine.json \
  Rscript --vanilla scripts/export_logistic_packages.R sirt_fine
uv run python scripts/compare_logistic_packages.py refine
uv run python scripts/compare_logistic_packages.py check
```

Exports, full per-start comparisons, gradients, Hessian eigenvalues and R fit
objects remain in ignored `outputs/logistic-mirt/`. No R dependency is added to
the Python package, and the existing mirt regression fixture is unchanged.

## Multidimensional logistic calibration

Issue #2 is covered by `multidimensional_irt`: exploratory/confirmatory
compensatory 2PL, orthogonal bifactor models and testlet random intercepts.
`tests/test_multidimensional.py` checks the one-dimensional reduction to
`logistic_irt`, structural zero loadings, shared testlet scales, zero-testlet
probability reduction, JIT/eager agreement, fully and partially missing responses,
invalid specifications, and parameter recovery. Exploratory recovery is compared
after orthogonal Procrustes alignment to a rotated generating loading matrix.
Structured recovery uses three general-only items to anchor the general factor
alongside two three-item groups; the earlier six-item bifactor design was too
weakly identified for a reliable parameter-recovery assertion.

`tests/test_multidimensional_mirt.py` adds eight regressions using R mirt 1.48.5
from reference commit `211851adae7a6bee580af830d37187f0e81a5ca8`, with R 4.5.1.
Four independent calibrations start from neutral loadings and zero intercepts;
four additional tests compare the probability, likelihood and EAP scoring kernels
at the R-exported item parameters. Data consist of deterministic rounded
response-pattern frequencies, approximately 200,000 subjects per case: six items
for the two-dimensional cases and nine items for bifactor/testlet (three factors).
These are numerical cross-software checks, not a broad empirical recovery study.

Both estimators fix latent means to zero, covariance to identity, item priors to
none, and use the same tensor product of 15 Gauss–Hermite nodes per factor.
Masks, positive anchor loadings, and within-testlet equality constraints match.
The exploratory case uses a lower-triangular anchor block; signed non-anchor
loadings remain free. R uses EM with Newton–Raphson M-steps, no acceleration and
outer tolerance 1e-8; jirt uses SLSQP with mean-loss tolerance 1e-13. All four
fits converged. Maximum absolute differences were:

| Specification | Loadings a | Intercepts d | Conditional probabilities | Total log likelihood |
| --- | ---: | ---: | ---: | ---: |
| Confirmatory 2D | 4.47e-7 | 1.51e-7 | 1.83e-7 | 4.19e-9 |
| Exploratory 2D, anchored | 4.19e-7 | 1.03e-7 | 1.67e-7 | 1.86e-9 |
| Bifactor | 6.70e-7 | 3.92e-7 | 2.17e-7 | 1.37e-8 |
| Testlet | 1.38e-6 | 3.76e-7 | 5.42e-7 | 4.35e-8 |

Predictions use 31 reproducibly sampled latent vectors per case. At identical
R parameters, maximum probability error is 2.0e-15, EAP error 9.2e-15, posterior
variance error 6.0e-15 and total likelihood error 3.3e-9. mirt automatically
classifies the triangular model as exploratory and normally rotates EAP scores.
The exporter disables that automatic rotation on a scoring copy of the fitted
object, preserving the explicitly anchored coordinates. This changes only the
scoring configuration, not the fitted parameters or reference package source.

For bifactor and testlet, a separate `bfactor` object with fixed JAX item
parameters checks conditional probabilities, with maximum difference 5.6e-16.
This is a check of the specialized engine's response model, **not an independent
bfactor calibration comparison**; the independent structured fits above use the
full-dimensional `mirt` engine and exactly matched tensor quadrature.

The implementation uses full tensor quadrature, not Cai's two-tier integration
reduction. Computational cost grows exponentially in factor count; no large-
dimensional performance claim is made. The present covariance convention is
orthogonal N(0,I), and masks are not automatically certified identifiable.
See `API.md` for scale, sign, rotation and testlet identification conventions.

Reproduce from the repository root (the optional R library stays in outputs):

```sh
sh scripts/build_mirt_reference.sh
uv run python scripts/prepare_multidimensional_reference.py
Rscript --vanilla scripts/export_multidimensional_reference.R
uv run python scripts/check_multidimensional_reference.py
# After reviewing the comparison:
cp outputs/multidimensional/fixture.json tests/fixtures/multidimensional_mirt.json
uv run pytest tests/test_multidimensional.py tests/test_multidimensional_mirt.py
uv run python examples/multidimensional.py
```

The checker verifies fixed-parameter probabilities, likelihood and posterior
moments, reports independent-fit differences, and prepares a compact fixture.
The regression tests enforce independent-fit agreement as well. Generated
reports remain ignored under `outputs/multidimensional/`; R is unnecessary for
routine Python regression tests. The structured example uses 3,000 simulated
subjects, nine items, and partial missingness; all three specifications converge.

## General polytomous calibration

GRM, PCM, GPCM and RSM are checked against independently fitted R mirt 1.48.5
(reference commit `211851adae7a6bee580af830d37187f0e81a5ca8`, R 4.5.1).
Both packages use fixed N(0,1), no item priors, and identical 41-point
Gauss–Hermite quadrature. Four-item synthetic response-pattern tables contain
approximately 100,000 subjects each. GRM/PCM/GPCM use category counts 2,3,4,3;
RSM uses three categories on each item. Independent SciPy formulas generate the
pattern frequencies. PCM/GPCM recovery includes deliberately unordered steps.

R uses `graded`, `gpcmIRT` (unit slopes for PCM), and `rsm`, explicitly fixing
latent variance. RSM's effective steps are b_k-c_j/a_j; these are converted to
jirt's item-plus-centered-shared-step convention. Both calibrations start with
unit slopes and evenly spaced steps, not reference estimates. mirt uses EM with
nlminb M-steps, outer tolerance 1e-9 and no acceleration; jirt uses SLSQP with
mean-loss tolerance 1e-13. All fits converge.

| Model | Max slope difference | Max step difference | Max probability difference | Total log-likelihood difference |
| --- | ---: | ---: | ---: | ---: |
| GRM | 2.60e-6 | 1.47e-6 | 7.67e-7 | 8.61e-9 |
| PCM | 0 | 8.45e-7 | 1.75e-7 | 1.34e-9 |
| GPCM | 4.30e-6 | 5.43e-6 | 7.51e-7 | 1.34e-8 |
| RSM | 0 | 6.67e-7 | 2.22e-7 | 2.50e-9 |

Probabilities are compared on 41 ability points over [-4,4]. At identical
parameters, probability errors are below 2.7e-15. EAP and posterior variance
comparisons are also enforced in the regression tests. The separate unit tests
cover normalization, extreme binary predictors, mixed category counts, binary
reductions, GRM ordering, PCM/RSM restrictions of GPCM, unobserved categories,
missingness, JIT/eager agreement and parameter recovery. Existing ordinal tests
remain part of the full suite. R is optional for normal Python tests.

```sh
sh scripts/build_mirt_reference.sh
uv run python scripts/prepare_polytomous_reference.py
Rscript --vanilla scripts/export_polytomous_reference.R
uv run python scripts/check_polytomous_reference.py
# After reviewing comparison.json:
cp outputs/polytomous/fixture.json tests/fixtures/polytomous_mirt.json
uv run pytest tests/test_polytomous.py tests/test_polytomous_mirt.py
uv run python examples/polytomous.py
```

Generated reports stay in ignored `outputs/polytomous/`; only the reproducible
regression fixture is tracked. These small matched cases validate implementation
agreement, not global optimization or identification for arbitrary sparse data.

## Nominal and nested distractor models

`nominal_irt` implements Bock's nominal response model and nested logistic
models with 2PL, 3PL, 3PLu or full 4PL correct-answer curves. Unit tests cover
softmax normalization, common-logit-shift invariance, varying category counts,
all four exact distractor-collapse identities, extreme abilities, binary nested
reduction, reference constraints, missing/unobserved categories, and JIT/eager
agreement. The runnable example covers all five fitting configurations.

Independent reference fits use R mirt 1.48.5, source commit
`211851adae7a6bee580af830d37187f0e81a5ca8`, R 4.5.1. Five-item synthetic tables
have category counts 3,4,3,4,3 and approximately 300,000 observations each.
Nested answer keys vary by item. Independent SciPy softmax/expit formulas
produce deterministic rounded response-pattern frequencies. Both packages fix
N(0,1), use no item priors, the same 41-point Gauss–Hermite rule and neutral
starts. mirt uses EM/nlminb without acceleration, TOL=1e-8; jirt uses SLSQP,
mean-loss tolerance 1e-13. All five pairs report convergence.

For mirt nominal items the multiplicative a1 is fixed to one and category
coefficients ak are freed except the zero reference, matching jirt's direct
category slopes. The default last-category ak constraint is removed. The first
item/category-one slope is positive. Nested types are `2PLNRM`, `3PLNRM`,
`3PLuNRM`, `4PLNRM`: distractor coefficients exclude the key and fix the first
non-key option. Binary intercept d converts to difficulty b=-d/a. Marginal
asymptote bounds match; mirt lacks the coupled 4PL gap constraint, so valid
exported gaps are required by the probability tests' model inputs.

| Model | Maximum parameter difference | Maximum probability difference | Total log-likelihood difference |
| --- | ---: | ---: | ---: |
| Nominal | 2.92e-6 | 6.90e-7 | 3.33e-8 |
| Nested 2PL | 2.01e-6 | 5.15e-7 | 4.80e-8 |
| Nested 3PL | 9.09e-5 | 2.08e-5 | 9.81e-7 |
| Nested 3PLu | 2.13e-4 | 1.29e-4 | 6.82e-6 |
| Nested 4PL | 2.43e-3 | 1.21e-3 | 8.96e-5 |

Probabilities use 41 abilities over [-4,4]. Fixed-parameter probability agreement
is within 2.3e-15 for every model. Regression tests also check fixed-parameter
likelihood, EAP and posterior variance. Independent nominal/2PL fits are checked
for strict parameter agreement. For asymptote models the tests require a
converged JAX objective at least as good as the R fit within numerical tolerance;
this is **not** a claim of parameter parity or global optimality. The table
preserves the remaining fitting differences rather than hiding them with wider
parameter tolerances. These data do not establish broad statistical recovery.

```sh
sh scripts/build_mirt_reference.sh
uv run python scripts/prepare_nominal_reference.py
Rscript --vanilla scripts/export_nominal_reference.R
uv run python scripts/check_nominal_reference.py
# After reviewing the report:
cp outputs/nominal/fixture.json tests/fixtures/nominal_mirt.json
uv run pytest tests/test_nominal.py tests/test_nominal_mirt.py
uv run python examples/nominal.py
```

Generated reports and R installation stay in ignored outputs/. R is not needed
for routine tests. The reference repositories are unchanged.
