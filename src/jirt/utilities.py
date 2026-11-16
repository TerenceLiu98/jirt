"""Initialization, priors and explicit vote recoding."""

import jax
import jax.numpy as jnp
import numpy as np


def convert_votes(values, *, yes=(1,), no=(-1,), missing=(0,), absent=(9,)):
    """Map disjoint user codes to 1/-1/0/9; NaN is missing.

    Unrecognized codes raise instead of silently treating observations as missing.
    """
    groups = [tuple(x) for x in (yes, no, missing, absent)]
    flat = [x for g in groups for x in g]
    if len(set(flat)) != len(flat):
        raise ValueError("Vote code sets must be disjoint")
    a = np.asarray(values, dtype=float)
    out = np.zeros(a.shape, dtype=np.int8)
    known = np.isnan(a)
    for codes, target in zip(groups, (1, -1, 0, 9)):
        selected = np.isin(a, codes)
        out[selected] = target
        known |= selected
    if not known.all():
        raise ValueError("Unrecognized vote codes")
    return out


def make_priors(model="binary"):
    """Return independent default prior arrays for binary, ordinal or network.

    Dynamic, hierarchical and Poisson priors are supplied explicitly because
    their scale and dimensions depend on the study design.
    """
    if model == "binary":
        return dict(
            x_mu=0.0,
            x_variance=1.0,
            beta_mu=np.zeros(2),
            beta_covariance=25 * np.eye(2),
        )
    if model == "ordinal":
        return dict(
            x=dict(mu=np.zeros(1), sigma=np.eye(1)),
            beta=dict(mu=np.zeros(2), sigma=25 * np.eye(2)),
        )
    if model == "network":
        return {
            k: dict(
                mu=1.0 if k == "gamma" else 0.0,
                sigma=25.0 if k in ("alpha", "beta") else 1.0,
            )
            for k in ("alpha", "beta", "w", "theta", "gamma")
        }
    raise ValueError("No default priors for this model; supply study-specific priors")


def get_starts(n_subjects, n_items, *, key, method="zeros"):
    """Binary starts: zeros initializes item parameters to zero and x randomly."""
    if (
        not isinstance(n_subjects, int)
        or not isinstance(n_items, int)
        or min(n_subjects, n_items) < 1
    ):
        raise ValueError("Dimensions must be positive integers")
    if method not in ("zeros", "random"):
        raise ValueError("method must be zeros or random")
    ka, kb, kx = jax.random.split(key, 3)
    return dict(
        alpha=jax.random.normal(ka, (n_items,)) * 10
        if method == "random"
        else jnp.zeros(n_items),
        beta=jax.random.normal(kb, (n_items,)) * 10
        if method == "random"
        else jnp.zeros(n_items),
        x=jax.random.normal(kx, (n_subjects,)),
    )


def network_starts(edges, *, key):
    """Initialize network intercepts from observed marginal edge frequencies."""
    from jax.scipy.special import ndtri

    y = np.asarray(edges, dtype=float)
    if y.ndim != 2 or not np.isin(y, [0, 1]).all():
        raise ValueError("Initialization requires a complete 0/1 matrix")
    a = ndtri(jnp.asarray(y.mean(axis=0)))
    b = ndtri(jnp.asarray(y.mean(axis=1)))
    a = jnp.nan_to_num(a, posinf=2.0, neginf=-2.0)
    b = jnp.nan_to_num(b, posinf=2.0, neginf=-2.0)
    kt, kw = jax.random.split(key)
    return dict(
        alpha=a,
        beta=b,
        theta=0.1 * jax.random.normal(kt, (y.shape[0],)),
        w=0.1 * jax.random.normal(kw, (y.shape[1],)),
        gamma=jnp.asarray(1.0),
    )
