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

## Noncompensatory and sequential models

Independent R mirt 1.48.5 (R 4.5.1) fits cover PC1PL, PC2PL, PC3PL,
sequential and Tutz. PC cases use eight binary items: three pure indicators
per factor and two items requiring both dimensions. Sequential cases use four
items with 2/3/4/3 categories and unordered stage difficulties. Independent
SciPy formulas generate rounded pattern frequencies (approximately 300,000
subjects). These synthetic cases test matched calibration, not broad recovery.

Both implementations fix abilities to N(0,I), with no item priors, and use the
same Gauss–Hermite nodes and weights: 15 per dimension for the two-dimensional
PC cases and 41 for sequential cases. Positive slope restrictions and guessing
bounds match. PC parameters map directly to mirt a/d/g. Sequential difficulties
map to -d/a, with unit a for Tutz. mirt uses EM/nlminb, no acceleration,
TOL=1e-8 and up to 10,000 cycles; jirt uses SLSQP with mean-loss tolerance 1e-13.

| Model | Maximum parameter difference | Maximum probability difference | Total log-likelihood difference |
| --- | ---: | ---: | ---: |
| PC1PL | 9.09e-6 | 1.69e-6 | 6.92e-8 |
| PC2PL | 1.18e-4 | 2.19e-5 | 9.70e-7 |
| PC3PL | 1.57e-3 | 6.07e-4 | 2.79e-5 |
| Sequential | 2.07e-6 | 5.91e-7 | 1.54e-8 |
| Tutz | 6.18e-7 | 9.21e-8 | 1.09e-8 |

All fits converged. Probabilities are evaluated on a 3-by-3 grid at -3/0/3 for
PC and 41 abilities in [-4,4] for sequential models. At identical R parameters,
maximum probability error is 1.78e-15. Regression tests also compare marginal
likelihood, EAP scores and posterior variances at fixed R parameters. Independent
fits require converged likelihood at least as good as R within numerical
tolerance; strict parameter parity is tested for PC1PL/sequential/Tutz. The
remaining PC2PL/PC3PL fitting differences are retained above, not treated as
parameter parity or evidence of global optimality.

```sh
uv run python scripts/prepare_response_reference.py
Rscript --vanilla scripts/export_response_reference.R
uv run python scripts/check_response_reference.py
# After reviewing outputs/response-models/comparison.json:
cp outputs/response-models/fixture.json tests/fixtures/response_mirt.json
uv run pytest tests/test_noncompensatory.py tests/test_sequential.py tests/test_response_mirt.py
uv run python examples/response_models.py
```

The R library is installed with `scripts/build_mirt_reference.sh`. Generated
reports stay under ignored outputs/; reference repositories remain unchanged.

Methods: Chalmers & Flora (2014), *Maximum-Likelihood Estimation of
Noncompensatory IRT Models With the MH-RM Algorithm*,
[doi:10.1177/0146621614520958](https://doi.org/10.1177/0146621614520958);
Tutz (1990), *Sequential item response models with an ordered response*,
[doi:10.1111/j.2044-8317.1990.tb00925.x](https://doi.org/10.1111/j.2044-8317.1990.tb00925.x).
The PC response mechanism follows that literature; jirt uses quadrature MML,
not the paper's MH-RM estimator.

## GGUM unfolding models

Independent R mirt 1.48.5 (R 4.5.1) fits cover six binary items and five mixed
items with 2/3/4/3/3 categories. Independent NumPy paired-exponential formulas
generate deterministic rounded frequencies (approximately 500,000 persons).
Both packages use fixed N(0,1), 61-point Gauss–Hermite nodes/weights, positive
discriminations, positive first-item location, no item priors, and identical
perturbed starting values. The parameter mapping is `a=a1`, `delta=b1`, `tau=t`.

jirt enforces positive decreasing thresholds. mirt uses positive bounds but
has no corresponding order constraint in this comparison; the export script
requires strictly ordered interior R estimates before accepting the reference.
Thus both fitted solutions satisfy the same restrictions. R uses EM/nlminb,
TOL=1e-9, no acceleration and up to 10,000 cycles; jirt uses SLSQP with mean-loss
tolerance 1e-13. All four fits converged.

| Case | Maximum parameter difference | Maximum probability difference | Total log-likelihood difference |
| --- | ---: | ---: | ---: |
| Binary GGUM | 4.39e-5 | 2.73e-5 | 4.03e-7 |
| Mixed-category GGUM | 3.54e-6 | 2.59e-6 | 1.96e-8 |

Probability comparisons use 41 points in [-4,4]. At identical R parameters,
maximum probability error is 3.00e-15. Regression tests check fixed-parameter
likelihood, EAP and posterior variance as well as independent parameter fits.
Other tests cover expected-score proximity, binary responses, mixed-category
normalization/padding, extreme-distance stability, a smooth gradient at the
ideal point, partial/all-missing persons, and reflection/location/scale
invariance. Synthetic frequency-table recovery checks binary and polytomous
item parameters on the identified metric (absolute tolerance 0.015).

These cases use informative starts near the generating parameters. They verify
matched numerical calibration and local recovery, not convergence from arbitrary
starts or global optimality. The reproducible attitude example uses two different
starting location scales; both converge to the same likelihood in that sample.

```sh
sh scripts/build_mirt_reference.sh
uv run python scripts/prepare_ggum_reference.py
Rscript --vanilla scripts/export_ggum_reference.R
uv run python scripts/check_ggum_reference.py
# After reviewing outputs/ggum/comparison.json:
cp outputs/ggum/fixture.json tests/fixtures/ggum_mirt.json
uv run pytest tests/test_unfolding.py tests/test_ggum_mirt.py
uv run python examples/unfolding.py
```

Generated reports remain under ignored outputs/ggum/; R is not needed for routine
regression tests and reference source trees are unchanged.

Roberts, Donoghue & Laughlin (2000), *A General Item Response Theory Model for
Unfolding Unidimensional Polytomous Responses*, Applied Psychological Measurement
24, 3–32, [doi:10.1177/01466216000241001](https://doi.org/10.1177/01466216000241001).

## Observed groups, latent mixtures and structural zeros

Six independent R mirt 1.48.5 (R 4.5.1) comparisons cover observed groups,
ordinary mixtures and zero-inflated mixtures, each with binary 2PL and GRM
responses. Binary cases have eight items; graded cases have five three-category
items. Independent SciPy response formulas generate rounded frequencies of
approximately 300,000 per observed group or 300,000 per mixture sample.

Known-group cases share the first two anchor items, free remaining item
parameters and estimate the second group's mean/SD. Ordinary mixtures estimate
two class-specific item profiles under fixed N(0,1), plus class weights.
Zero-inflated cases estimate one regular N(0,1) component and a structural class.
JAX and R both fit two distinct starts for every latent mixture. Regression tests
also reverse the second start's regular class labels and align estimated profiles
with a minimum-cost assignment before comparing parameters and recovery.

| Configuration | Maximum parameter difference | Maximum weight difference | Total log-likelihood difference |
| --- | ---: | ---: | ---: |
| Observed-group 2PL | 9.73e-6 | — | 1.46e-7 |
| Observed-group GRM | 7.37e-6 | — | 1.52e-7 |
| Two-class 2PL | 6.68e-5 | 6.48e-6 | 1.34e-6 |
| Two-class GRM | 7.50e-5 | 7.04e-7 | 1.66e-6 |
| Zero-inflated 2PL | 6.12e-7 | 5.97e-9 | 9.08e-9 |
| Zero-inflated GRM | 5.28e-7 | 1.85e-8 | 2.79e-9 |

Parameters include slopes, thresholds and latent means/SDs where estimated.
At identical R parameters, conditional response probabilities agree within
2.3e-15, EAP/variance within 7.2e-15 and class memberships within 3.0e-15.
R curves and EAP moments come from `probtrace`/`fscores`; reference memberships
are independently computed in R from `probtrace` pattern products, quadrature
and the fitted mirt class weights. They are not an export of `fscores(classify)`.
Tests additionally check single-group/class reductions to ordinary Rasch/2PL/GRM,
exact zero-weight prediction limits, structural/missing distinctions, category
normalization and label-invariant recovery of synthetic parameters.

jirt uses 61-point Gauss–Hermite quadrature with nodes transformed by component
mean/SD and SLSQP mean-loss tolerance 1e-13. R uses EM/nlminb, TOL=1e-9, no
acceleration and up to 10,000 cycles. All starts converged. Ordinary mixtures
share the exact GH nodes/weights. For estimated group distributions, mirt uses
a uniform [-10,10] grid with spacing 0.1; the fixed-parameter likelihood check
verifies that its integration difference is below 1e-6 for these cases.

For zero inflation, the R reference follows mirt's documented construction: a
fixed class with mean -100, variance 1e-5, unit slopes and fixed intercepts,
using a grid containing -100 plus the uniform regular grid. This approximates
structural zeros to floating-point precision; jirt uses an exact deterministic
component. The comparison's reference memberships use the exact structural
pattern probability. No extreme person score is used by jirt. Intercepts map
to thresholds via `steps=-d/a`; variance maps to `sigma=sqrt(COV_11)`. R PI
parameters are converted from reference logits to normalized class weights.

These informative synthetic cases establish matched numerical calibration and
local recovery, not general identifiability or global optimality for mixtures.
Item recovery is checked within 0.003 and class weights within 0.001 of the
unrounded generating parameters. No item priors are used.

```sh
sh scripts/build_mirt_reference.sh
uv run python scripts/prepare_group_reference.py
uv run python scripts/fit_group_reference.py
Rscript --vanilla scripts/export_group_reference.R
uv run python scripts/check_group_reference.py
# After reviewing outputs/groups/comparison.json:
cp outputs/groups/fixture.json tests/fixtures/groups_mirt.json
uv run pytest tests/test_groups.py tests/test_groups_mirt.py
uv run python examples/groups.py
```

Generated reports remain ignored under outputs/groups/. The R regression fixture
allows routine tests without R; reference source trees remain unchanged.

References: Chalmers (2012), [doi:10.18637/jss.v048.i06](https://doi.org/10.18637/jss.v048.i06);
Rost (1990), [doi:10.1177/014662169001400305](https://doi.org/10.1177/014662169001400305);
Wall, Park & Moustaki (2015), [doi:10.1177/0146621615588184](https://doi.org/10.1177/0146621615588184).
