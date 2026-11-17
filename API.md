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

## Nominal and nested response models

`nominal_irt(responses, category_counts=..., answer_keys=None, model="nominal",
starts=None, quadrature_points=41, max_iter=1000, tolerance=1e-10, jit=True)`
fits unordered categories with fixed N(0,1) ability and no item priors. Counts
are declared integers >=2 per item, response labels start at zero, and NaN
marks missing responses. Unobserved declared categories are retained; they may
lead to poorly identified or boundary estimates. Each item needs observations.

`model="nominal"` uses category logits `slopes[j,k]*theta + intercepts[j,k]`
and a stable softmax. Both coefficients of category zero are fixed to zero for
each item, removing arbitrary common shifts. The item-zero/category-one slope
is constrained >=1e-6 to fix the ability sign. Choose that anchor by reordering
items/categories if necessary; category labels have no inherent order. Other
category slopes are unrestricted. Location and scale are fixed by N(0,1).

For nested models, choose `model="2pl"`, `"3pl"`, `"3plu"` or `"4pl"` and
supply zero-based `answer_keys` of item length. Correct-answer probability is
`q=c+(u-c)*sigmoid(a*(theta-b))`. 2PL fixes c=0,u=1; 3PL estimates c; 3PLu
estimates u only; 4PL estimates both. Slopes satisfy a>=1e-6. Fitted asymptotes
lie in [1e-6,1-1e-6], with u-c>=1e-6 for 4PL. Each incorrect category receives
`(1-q)*softmax(distractor logits)`. Thus summing distractors exactly yields
1-q, while the key receives q. With one distractor the model reduces to the
specified binary response curve. Correct-option nominal coefficients and both
coefficients of the first non-key category are fixed at zero. Distractor slopes
are otherwise unrestricted; the positive correct curve fixes latent direction.

`starts` accepts `slopes` and `intercepts`, padded `(J,max(K))` arrays, plus free
binary parameters `a`, `b`, `c`, `u` as `(J,)` arrays for nested models. Reference,
key and padding entries must be zero. Defaults are free category slopes 0.5,
category intercepts zero, a=1,b=0,c=0.1,u=0.9 where applicable. No ordering
constraints are imposed on unordered categories or distractors.

`NominalResult.means` returns category coefficients, EAP `x`, and the binary
parameters for nested fits. `variances["x"]` gives conditional posterior
variance, excluding item calibration uncertainty. Fully missing subjects retain
the N(0,1) prior. Prediction has shape `(N,J,max(K))`, with zero probability for
padding. `predict(theta)` uses supplied ability values; omitting them, or using
`probabilities`, returns EAP plug-in predictions. `category_counts` and
`answer_keys` preserve the input coding. SLSQP uses JAX derivatives and Gaussian
quadrature; tolerance is ftol on mean negative marginal likelihood. Check
`converged`, `iterations`, `message`, `trace` and total `log_likelihood`.

`nominal_probabilities` and `nested_probabilities` expose pure JAX probability
functions with matching keyword parameters. They assume valid inputs and allow
equivalent common shifts of category coefficients. Padded coefficients can be
NaN in these low-level functions; the fitting interface requires zero padding.
Use multiple starts and quadrature checks for weakly identified fits, particularly
nested 3PL/4PL. See `uv run python examples/nominal.py` for a distractor-analysis
example covering all five configurations with varying keys and category counts.

## Noncompensatory binary models

`noncompensatory_irt(responses, loading_mask=..., model="pc2pl", ...)` fits
`pc1pl`, `pc2pl` or `pc3pl`. Responses are `(N,J)` binary observations with NaN
missing. The required boolean `(J,D)` loading mask defines active dimensions;
each row and column must have an active entry. Success is
`c_j + (1-c_j) * product_h sigmoid(a_jh * theta_h + d_jh)` over active dimensions.
A high ability in one dimension cannot remove a bottleneck in another.
PC1PL fixes active slopes to one. PC2PL estimates positive active slopes.
PC3PL additionally estimates item guessing probabilities in `[1e-6,1-1e-6]`.
The other two models fix guessing to zero. Inactive slopes and intercepts are zero.

The latent distribution is fixed to `N(0,I)`, fixing location and scale, and
positive slopes fix direction. Use substantive mask anchors to label dimensions:
identical mask columns leave permutation ambiguity. A valid mask alone does not
prove statistical identification; each factor needs enough informative items.
The model does not estimate latent correlations or allow negative active slopes.

`starts` accepts `d` `(J,D)`, free `a` `(J,D)`, and `c` `(J,)` for PC3PL.
Inactive entries must be zero. Controls are `quadrature_points=15`,
`max_nodes=100000`, `max_iter=1000`, `tolerance=1e-10`, and `jit=True`.
Tensor quadrature grows exponentially with D and rejects grids over `max_nodes`.
No item priors are used. Response patterns are compressed before optimization.
Enable JAX x64. Check `result.converged` and `result.message`.

`NoncompensatoryResult.means` contains `a`, `d`, `c` and EAP `x` `(N,D)`;
`covariances["x"]` is `(N,D,D)` and `variances["x"]` contains its diagonal.
These are conditional on item estimates. All-missing subjects recover the
quadrature prior; entirely missing items are rejected. `predict(theta)` accepts
`(N,D)` abilities and `probabilities` predicts at EAP scores.
`noncompensatory_probabilities` is a pure differentiable JAX kernel with the same
mask and parameter conventions; callers supply valid shapes and restrictions.

## Sequential ordered responses

`sequential_irt(responses, category_counts=..., model="sequential", ...)` fits
sequential logistic stages; `model="tutz"` fixes slopes to one. Responses are
zero-based categories with NaN missing. Counts `(J,)` are explicit integers >=2,
including unobserved categories; mixed counts are supported.

Stage h succeeds with `q_h = sigmoid(a_j*(theta-b_jh))`. Category k means passing
stages before k and failing stage k; the last category passes every stage.
Stage difficulties need not be ordered. The sequential model estimates one
positive slope per item shared by all stages; Tutz fixes it to one. Ability is
fixed `N(0,1)` and there are no item priors. Binary items reduce to logistic 2PL
or Rasch respectively. Products and failure probabilities use log arithmetic.

`starts` accepts `steps` `(J,max(K)-1)` and, for sequential only, `a` `(J,)`.
Unused padded steps are ignored. Controls match `polytomous_irt`, including
`quadrature_points=41`. Returns `PolytomousResult` with item parameters, EAP `x`,
conditional posterior variances and optimizer diagnostics. Predictions have shape
`(N,J,max(K))` with zero probability for padding. `sequential_probabilities` is
the corresponding pure JAX prediction function. The same two model names are
also accepted by `polytomous_irt` and `polytomous_probabilities`.

## Generalized graded unfolding model

`ggum_irt(responses, category_counts=..., starts=..., anchor_item=0, ...)`
fits unidimensional GGUM for binary or ordered polytomous items. Responses are
`(N,J)` integer categories `0..K_j-1`, where larger categories mean stronger
agreement, and NaN means missing. Declare counts `(J,)` explicitly, including
unobserved categories. Items may have different category counts.

Let `z = a_j*(theta-delta_j)` and `S_k = sum(tau_jh, h=1..k)`, with `S_0=0`.
Category weights are `exp(a_j*S_k)*cosh((K_j-1/2-k)*z)`, normalized across
`k=0..K_j-1`. This is the paired-exponential GGUM formula with a common factor
removed. Implementation uses log-add-exp and log-softmax, remains smooth at
`theta=delta_j`, and avoids exponential overflow at large distances. For binary
items the same formula applies with `K_j=2`. Expected agreement is highest at
the item location and falls as distance increases in either direction.

Parameters and constraints:

- `a` `(J,)`: positive discriminations, bounded below by `1e-6`.
- `delta` `(J,)`: item ideal locations. `delta[anchor_item] >= 1e-6` fixes the
  otherwise arbitrary reflection of the entire scale. Choose an item known to
  lie on its positive side; other locations are unrestricted.
- `tau` `(J,max(K)-1)`: positive, strictly decreasing distance thresholds,
  `tau_1 > ... > tau_(K-1) > 0`, with minimum gap/value `1e-6`. This uses mirt's
  `t` sign convention: negate these to obtain the original GGUM lower-side
  thresholds, whose upper-side counterparts are symmetric. Unused padding is
  ignored on input and zero on output.

`starts` requires all three arrays. Ability is fixed to `N(0,1)`, identifying
location and scale; no item priors are used. A reporting transformation
`theta' = shift + sign*scale*theta` maps `delta` the same way, divides `a` by
positive `scale`, and multiplies `tau` by `scale`. `sign` is either +1 or -1.
Response probabilities are invariant under this transformation. An anchor fixes
direction, but does not guarantee identification for an uninformative design.
Use items spanning the attitude range. The likelihood is nonconvex: try several
substantively reasonable starting locations and compare converged likelihoods.
The API requires explicit starts instead of inferring an item ordering.

Controls are `quadrature_points=61`, `max_iter=2000`, `tolerance=1e-10` and
`jit=True`. Enable JAX x64. Fitting uses compressed response patterns,
Gauss–Hermite marginal maximum likelihood and SLSQP with JAX derivatives.
`tolerance` applies to the mean negative log likelihood. Check
`result.converged`/`message`; convergence alone does not establish a global
optimum. Increase quadrature density when discriminations or distant locations
make the integrand narrow.

`GGUMResult.means` contains `a`, `delta`, `tau` and EAP person locations `x`.
`variances["x"]` is the conditional posterior variance, excluding item estimation
uncertainty. Unfolding posteriors can be bimodal, so EAP alone may not describe a
person well. All-missing persons recover the quadrature prior; entirely missing
items are rejected. `predict(theta)` and `probabilities` return
`(N,J,max(K))` category probabilities with zero padding, the latter evaluated
at EAP scores. `ggum_probabilities` is a differentiable pure JAX function with
the same parameter conventions; callers provide valid shapes and restrictions.
See `examples/unfolding.py` for synthetic attitude responses and two-start fitting.
