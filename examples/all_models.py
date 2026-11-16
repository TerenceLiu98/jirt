"""Small standalone examples for all six models: uv run python examples/all_models.py."""

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
from jirt import (
    Control,
    bin_irt,
    ord_irt,
    dyn_irt,
    hier_irt,
    network_irt,
    pois_irt,
    get_starts,
    make_priors,
    network_starts,
)

rng = np.random.default_rng(42)
n, m = 12, 6
control = Control(max_iter=10)
y = rng.choice([-1, 1], size=(n, m))
s = get_starts(n, m, key=jax.random.key(1))
binary = bin_irt(y, starts=s, control=control)
ordinal = ord_irt(
    rng.integers(1, 4, size=(n, m)),
    starts=dict(
        x=rng.normal(size=n),
        beta=rng.normal(size=m),
        tau=np.full(m, 0.5),
        DD=np.ones(m),
    ),
    priors=make_priors("ordinal"),
    control=control,
)
dynamic = dyn_irt(
    y,
    starts=dict(alpha=np.zeros(m), beta=rng.normal(size=m), x=np.zeros((n, 3))),
    priors=dict(
        x_mu0=np.zeros(n),
        x_variance0=np.ones(n),
        omega2=np.full(n, 0.1),
        beta_mu=np.zeros(2),
        beta_covariance=25 * np.eye(2),
    ),
    start_period=np.zeros(n, dtype=int),
    end_period=np.full(n, 2),
    item_period=np.repeat(np.arange(3), 2),
    control=control,
)
hierarchical = hier_irt(
    y.reshape(-1),
    subject=np.repeat(np.arange(n), m),
    item=np.tile(np.arange(m), n),
    groups=np.repeat(np.arange(3), 4),
    covariates=np.column_stack([np.ones(n), np.linspace(-1, 1, n)]),
    starts=dict(
        alpha=np.zeros(m),
        beta=rng.normal(size=m),
        gamma=rng.normal(size=(3, 2)),
        eta=np.zeros(n),
        sigma=np.ones(3),
    ),
    priors=dict(
        beta_mu=np.zeros(2),
        beta_covariance=25 * np.eye(2),
        gamma_mu=np.zeros(2),
        gamma_covariance=np.eye(2),
        sigma_v=2.0,
        sigma_s=1.0,
    ),
    control=control,
)
edges = (y == 1).astype(int)
network = network_irt(
    edges,
    starts=network_starts(edges, key=jax.random.key(2)),
    priors=make_priors("network"),
    control=control,
)
poisson = pois_irt(
    rng.poisson(3, size=(m, n)),
    starts=dict(
        alpha=np.ones(n),
        psi=np.zeros(m),
        beta=rng.normal(size=m) * 0.1,
        x=rng.normal(size=n) * 0.1,
    ),
    priors={k: dict(mu=0.0, sigma2=1.0) for k in ("alpha", "psi", "beta", "x")},
    control=control,
)
for name, fit in [
    ("binary", binary),
    ("ordinal", ordinal),
    ("dynamic", dynamic),
    ("hierarchical", hierarchical),
    ("network", network),
    ("poisson", poisson),
]:
    assert all(np.isfinite(v).all() for v in fit.means.values())
    print(name, fit.iterations, fit.converged)
