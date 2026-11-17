# Python interfaces

All estimators accept NumPy/JAX arrays or nested numeric sequences. Enable
`jax_enable_x64` before fitting. Inputs are validated on the host; iteration
updates run as pure JAX functions. `jit=False` executes the same updates without
compilation. Entire-fit reverse-mode differentiation is not a supported API.

## Common output and controls

`Control(max_iter=500, tolerance=1e-6, jit=True, convergence="correlation")`
controls fitting. Ordinal defaults to 300 iterations, Poisson to 1500, and network
uses absolute convergence by default. A supplied `Control` overrides those defaults.

`FitResult.means` and `.variances` contain JAX arrays. `.trace` stores a NumPy
array of per-update convergence errors; `.iterations` counts outer updates, and
`.converged` indicates whether the criterion was met. Binary performs its
additional final conditional maximization after detecting convergence.

| Model | Starts | Priors | Additional inputs |
| --- | --- | --- | --- |
| Binary | alpha[J], beta[J], x[N] | x_mu scalar, x_variance scalar, beta_mu[2], beta_covariance[2,2] | method="em" or "variational"; optional anchor_subject or anchor_outcomes |
| Ordinal | tau[J], beta[J], DD[J], x[N] | x: mu[1], sigma[1,1]; beta: mu[2], sigma[2,2] | Three response categories; DD is the squared scale |
| Dynamic | alpha[J], beta[J], x[N,T] | x_mu0[N], x_variance0[N], omega2[N], beta_mu[2], beta_covariance[2,2] | start_period[N], end_period[N], item_period[J] |
| Hierarchical | alpha[J], beta[J], gamma[G,D], eta[N], sigma[G] | beta_mu[2], beta_covariance[2,2], gamma_mu[D], gamma_covariance[D,D], sigma_v scalar, sigma_s scalar | subject[L], item[L], groups[N], covariates[N,D]; votes[L] |
| Network | alpha[J], beta[N], w[J], theta[N], gamma scalar | each parameter has mu scalar and sigma scalar | anchor_subject or anchor_item; gamma remains fixed |
| Poisson | alpha[K], psi[J], beta[J], x[I] | each parameter has mu scalar and sigma2 scalar | counts[J,K], optional document_subject[K] |

N = subjects, J = items (terms for Poisson), T = periods, G = groups,
D = covariates, L = recorded observations, K = documents, I = document authors
or other shared latent subjects. All indices are zero-based. Variance parameters
are variances, not standard deviations. Column-vector starts may also be passed.

Binary beta output has shape [J,2], with intercept first. Binary x variance is
scalar and beta covariance is [2,2], reflecting shared conditional covariance.
Ordinal x variance is scalar; beta and tau variances are per item. Dynamic
variances match their mean shapes. Hierarchical covariance arrays are
[G,D,D] and [J,2,2]. Network variance arrays are per subject/item.

## Parametric bootstrap

```python
from jirt import bootstrap

resampled = bootstrap(
    fit,
    votes,
    model="binary",
    key=jax.random.key(9),
    n_trials=50,
    fit_kwargs={"starts": starts, "priors": priors, "control": control},
)
```

Supported model names: binary, ordinal, dynamic, hierarchical. Preserve the
original fit arguments in `fit_kwargs`, including any indexing or anchoring.
`standard_errors` uses sample SD over finite replicates; `estimates` retains all
replicates, `converged` reports each stopping status, and `failures` maps failed
replicate numbers to errors. Original missingness and dynamic tenure masks are
preserved. `pseudo_data` optionally supplies a full stack of predetermined
replicates for independent validation.

## Logistic IRT

`logistic_irt(responses, model="2pl", starts=None, quadrature_points=41,
max_iter=1000, tolerance=1e-8, jit=True)` fits binary **0/1** responses with
**NaN** for missing observations. Zero is an observed failure, unlike the
missing-value convention of `bin_irt`.

The response curve is `c + (u-c) * sigmoid(a * (theta-b))`. Model options are
`rasch` (a fixed at 1), `1pl` (one estimated shared a), `2pl` (item-specific a),
`3pl` (also estimates c), and `4pl` (estimates both c and u). Lower models fix
c=0 and u=1 as appropriate. All discriminations are positive, so higher ability
increases success probability. The latent distribution is fixed N(0,1), which
identifies scale and location; the Rasch option also fixes the slope scale.

Estimation maximizes the Gauss–Hermite approximation to the marginal likelihood,
using JAX derivatives and SciPy SLSQP. There are no item priors. This is not the
probit EM/variational estimator exposed by `bin_irt`. Numerical bounds are
a >= 1e-6, estimated c >= 1e-6, estimated u <= 1-1e-6, and u-c >= 1e-6.
`starts` optionally supplies b, a, c and u only where those parameters are free;
scalar starts broadcast, and 1PL accepts one shared slope. Defaults are b=0,
a=1, c=0.1 and u=0.9. Each item needs an observed response; a fully missing
subject receives the standard-normal prior as its posterior.

`LogisticResult.means` contains a, b, c, u and EAP person scores x.
`variances["x"]` is conditional posterior uncertainty given fitted items; it
is not an item-parameter standard error and does not include calibration
uncertainty. `log_likelihood` is the total marginal log likelihood. `trace`
records this objective at initialization and optimizer callbacks. `converged`,
`iterations` and `message` expose the optimizer termination status; check these
before interpreting estimates. The tolerance applies to SLSQP's mean negative
log-likelihood objective, not parameter correlation.

`result.predict(theta)` returns conditional probabilities at supplied abilities;
omitting theta uses fitted EAP scores. `result.probabilities` contains these EAP
plug-in probabilities, not posterior-integrated predictive probabilities.
`logistic_probabilities(theta, a=..., b=..., c=0, u=1)` exposes the pure JAX
response function for JIT and differentiation, assuming valid parameter inputs.
The fitting wrapper itself is not an end-to-end differentiable API.

Increasing quadrature resolution and checking multiple starts is particularly
important for 3PL/4PL: small samples and weak identification can yield extreme
parameters or competing optima despite optimizer convergence. Run
`uv run python examples/logistic.py` for a reproducible example of all variants.

## Multidimensional 2PL, bifactor and testlet models

`multidimensional_irt(responses, dimensions=2, loading_mask=None,
structure="multidimensional", groups=None, anchors=None, starts=None,
quadrature_points=15, max_nodes=100000, max_iter=1000, tolerance=1e-10,
jit=True)` fits `sigmoid(theta @ a.T + d)` to 0/1 responses, with NaN missing.
Enable JAX x64 before fitting. `a` has shape `(items, dimensions)` and `d`
contains item intercepts; there is no unique scalar item difficulty in multiple
dimensions. Negative non-anchor loadings are permitted. No item priors are used.

All latent factors have fixed mean zero and covariance identity. This fixes
location and scale and assumes orthogonal factors; this release does not estimate
factor correlations. Factor signs and rotations require additional restrictions:

- For `structure="multidimensional"` without a mask, the first D items form a
  lower-triangular loading block with positive diagonal. This chooses an
  exploratory orientation when that block has full rank. Choose suitable anchor
  items by ordering the columns of the response matrix. The latent factors can
  be relabelled/rotated for interpretation after fitting, provided loadings and
  abilities are transformed together.
- A boolean `loading_mask` of shape `(items,D)` fixes every false entry to exactly
  zero. This supports confirmatory designs. `anchors` optionally specifies one
  item index per factor whose allowed loading must be at least 1e-6; otherwise
  the first allowed item is used. Signs alone do not remove arbitrary rotations:
  the supplied zero pattern must identify the intended orientation. An all-true
  mask is not an identified confirmatory model. The implementation checks shapes
  and nonempty factors, not generic/local identifiability of every possible mask.
- For `structure="bifactor"`, supply `groups`, an integer vector of item length.
  Contiguous IDs 0,1,... assign items to specific factors; -1 means general-only.
  Dimension is inferred as `1 + number of groups`. Every item loads on the general
  factor and at most one orthogonal specific factor. Allowed loadings are free
  per item. Each group requires at least three items, but this count alone is not
  sufficient for identification. Pure general items can anchor the general /
  specific decomposition, particularly for designs with only two groups.
- `structure="testlet"` uses the same mask but constrains specific loadings within
  each group to a shared nonnegative scale `tau_g`. With a standard-normal specific
  factor this is a common additive logit effect `t_ig ~ N(0,tau_g**2)`, whose item
  coefficients are one. `means["testlet_variance"]` returns these variances.
  Testlet is therefore a constrained bifactor model, not the subject-group
  regression implemented in `hier_irt`. This is an additive random-intercept
  convention; formulations that multiply testlet effects by item discrimination
  use different constraints. Zero specific scale removes that testlet effect.

`starts` accepts `a` `(J,D)` and `d` `(J,)`, also described by
`jirt.types.MultidimensionalStarts`. Default intercepts are zero, allowed
first-factor loadings are one and other allowed loadings are 0.5. Starts must
respect structural zeros, anchor signs and testlet equality constraints. Use
positive testlet starts: at exactly zero the variance-scale derivative is zero,
which can prevent a gradient optimizer from moving away from that boundary.

Estimation uses tensor Gauss–Hermite quadrature and SLSQP with JAX derivatives.
`quadrature_points` is **per factor**, so total nodes grow as `points**D`.
`max_nodes` rejects a grid above that limit before allocation. This is a
low-dimensional implementation, not Cai's dimension-reduction algorithm; high
factor counts require a different integration strategy. Response patterns are
aggregated, including their missingness pattern, before posterior evaluation.
Memory also scales with distinct patterns times nodes and factor count; the
node limit alone is not a memory guarantee. Increase quadrature resolution and
compare multiple starts when assessing a fit. Finite tensor quadrature is only
approximately rotation invariant; rotate results for interpretation, not as a
claim that arbitrary rotations preserve the finite-grid likelihood exactly.

`MultidimensionalResult.means` contains `a`, `d` and EAP scores `x` `(N,D)`.
`covariances["x"]` contains full conditional posterior covariance matrices
`(N,D,D)`; `variances["x"]` gives their diagonals. These exclude item-calibration
uncertainty. A completely missing subject receives the N(0,I) prior. Items with
no observed responses are rejected. `predict(theta)` accepts `(N,D)` abilities;
without theta it evaluates EAP plug-in probabilities, as does `probabilities`.
`multidimensional_probabilities(theta, a=..., d=...)` is a pure JAX function
supporting JIT and autodifferentiation; fitting is not end-to-end differentiable.

`log_likelihood`, `trace`, `converged`, `iterations` and `message` describe the
optimization. `tolerance` controls mean-loss SLSQP stopping precision. Reaching
the iteration limit returns a result with `converged=False`. See
`uv run python examples/multidimensional.py` for all three structures.

## General polytomous logistic IRT

`polytomous_irt(responses, category_counts=..., model="grm", starts=None,
quadrature_points=41, max_iter=1000, tolerance=1e-10, jit=True)` fits GRM, PCM,
GPCM or RSM. Responses are integer labels `0,...,K_j-1`, with NaN missing.
Declare every item's category count explicitly; unobserved categories are retained,
not recoded or dropped. GRM, PCM and GPCM permit different counts across items;
RSM requires a common count. Each item needs an observed response. Empty
categories can drive thresholds/steps toward boundaries; convergence is not a
certificate of finite, well-identified estimates. No pseudo-counts are added.

For GRM, `P(Y>=k)=sigmoid(a_j*(theta-b_jk))` for k=1,...,K_j-1. Category
probabilities are successive cumulative differences. Slopes are positive and
thresholds strictly increasing, with a numerical minimum gap of 1e-6. Log
probabilities use a stable difference formula rather than subtracting saturated
sigmoid values.

For GPCM, category k has unnormalized log probability
`a_j * sum(theta-b_jh, h=1,...,k)`; category zero has logit zero. PCM fixes
`a_j=1`. Their adjacent-category step difficulties **need not be ordered**.
RSM further constrains steps to `b_j + tau_k`, with unit slopes and
`sum(tau)=0`. Centering shared steps identifies the decomposition into item
locations and category effects; it does not impose threshold ordering.
All four models fix ability to N(0,1) and use no item priors. Positive slopes
fix the direction of increasing ability. These logistic models do not change
the existing three-category probit `ord_irt` API or behavior.

`starts` optionally supplies `steps` of shape `(J,max(K)-1)` and, for GRM/GPCM,
`a` of shape `(J,)`. Valid steps must be finite; padding is ignored and can be
NaN. RSM starts must admit the shared-step decomposition. Defaults are unit
slopes and evenly spaced steps from -1 to 1 (zero for binary items). Numerical
slope bounds are a>=1e-6. Fitting uses SLSQP with JAX derivatives and Gauss–Hermite
marginal likelihood. `tolerance` is ftol on mean negative marginal likelihood.

`PolytomousResult.means` contains `a`, padded `steps` and EAP `x`. RSM also
returns item locations `b` and centered shared `tau`. `variances["x"]` gives
conditional posterior variances, excluding item uncertainty. Fully missing
subjects receive the N(0,1) prior. `category_counts` preserves declared counts.
`converged`, `iterations`, `message`, `trace` and `log_likelihood` expose fitting
status and the total marginal log likelihood; iteration-limit results have
`converged=False`.

`predict(theta)` returns `(N,J,max(K))` category probabilities with exactly zero
padding for nonexistent categories. Omitting theta uses EAP scores, as does the
`probabilities` property; these are plug-in, not posterior-integrated predictions.
`polytomous_probabilities(theta, a=..., steps=..., category_counts=..., model=...)`
is a pure JAX function (make model static for JIT). It assumes valid shapes and
constraints; PCM/RSM force unit slopes. Supply shared RSM steps when using this
low-level function. Run `uv run python examples/polytomous.py` for all four models.
