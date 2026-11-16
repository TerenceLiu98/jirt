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
