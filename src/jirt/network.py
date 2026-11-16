"""Probit distance network scaling with fixed gamma."""

import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr, ndtr
import numpy as np
from .binary import Control, FitResult
from .types import NetworkStarts, NetworkPriors


def network_irt(
    edges,
    *,
    starts: NetworkStarts,
    priors: NetworkPriors,
    control=None,
    anchor_subject=None,
    anchor_item=None,
):
    """Fit subject-by-item 0/1 edges; NaN denotes unobserved edges.

    Priors contain scalar mu and sigma (variance) for alpha, beta, w, theta,
    gamma. Gamma stays fixed at its starting value.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable JAX x64 before fitting")
    c = control or Control(convergence="absolute")
    y = np.asarray(edges, dtype=float)
    if y.ndim != 2 or min(y.shape) < 1 or not (np.isin(y, [0, 1]) | np.isnan(y)).all():
        raise ValueError("edges must contain 0, 1 or NaN")
    n, m = y.shape
    names = ("alpha", "beta", "w", "theta")
    arrays = [np.asarray(starts[k], dtype=float).reshape(-1) for k in names]
    if [a.size for a in arrays] != [m, n, m, n] or not all(
        np.isfinite(a).all() for a in arrays
    ):
        raise ValueError("Invalid starts")
    gamma = float(np.asarray(starts["gamma"]).item())
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be positive and finite")
    mus = {}
    vs = {}
    for k in (*names, "gamma"):
        mus[k] = float(np.asarray(priors[k]["mu"]).item())
        vs[k] = float(np.asarray(priors[k]["sigma"]).item())
        if not np.isfinite(mus[k]) or not np.isfinite(vs[k]) or vs[k] <= 0:
            raise ValueError("Invalid priors")
    if (
        c.max_iter < 1
        or c.tolerance < 0
        or not np.isfinite(c.tolerance)
        or c.convergence not in ("absolute", "correlation")
    ):
        raise ValueError("Invalid controls")
    for index, size in ((anchor_subject, n), (anchor_item, m)):
        if index is not None and (
            not isinstance(index, (int, np.integer)) or not 0 <= index < size
        ):
            raise ValueError("Invalid anchor")
    yy = jnp.asarray(y)
    observed = jnp.isfinite(yy)
    va = 1 / (n + 1 / vs["alpha"])
    vb = 1 / (m + 1 / vs["beta"])

    def step(state):
        a, b, w, t, vw, vt = state
        distance = (t[:, None] - w[None, :]) ** 2
        eta = a[None, :] + b[:, None] - gamma * distance
        sign = jnp.where(yy == 1, 1.0, -1.0)
        z = eta * sign
        latent = eta + jnp.where(
            observed,
            sign * jnp.exp(-0.5 * z * z - 0.5 * jnp.log(2 * jnp.pi) - log_ndtr(z)),
            0.0,
        )
        a = va * (
            mus["alpha"] / vs["alpha"]
            + jnp.sum(latent - b[:, None] + gamma * distance, axis=0)
        )
        b = vb * (
            mus["beta"] / vs["beta"]
            + jnp.sum(latent - a[None, :] + gamma * distance, axis=1)
        )
        residual = latent - a[None, :] - b[:, None]
        d = t[:, None] - w[None, :]
        curvature = 2 / vs["w"] + 4 * jnp.sum(
            gamma * residual + 3 * gamma**2 * d * d, axis=0
        )
        gradient = 2 / vs["w"] * (w - mus["w"]) - 4 * jnp.sum(
            gamma * d * residual + gamma**2 * d**3, axis=0
        )
        vw = 2 / curvature
        w = w - gradient / curvature
        d = t[:, None] - w[None, :]
        curvature = 2 / vs["theta"] + 4 * jnp.sum(
            gamma * residual + 3 * gamma**2 * d * d, axis=1
        )
        gradient = 2 / vs["theta"] * (t - mus["theta"]) + 4 * jnp.sum(
            gamma * d * residual + gamma**2 * d**3, axis=1
        )
        vt = 2 / curvature
        t = t - gradient / curvature
        return a, b, w, t, vw, vt

    update = jax.jit(step) if c.jit else step
    state = (*map(jnp.asarray, arrays), jnp.full(m, vs["w"]), jnp.full(n, vs["theta"]))
    history = []
    converged = False
    for iteration in range(1, c.max_iter + 1):
        old = state
        state = update(state)
        if not all(np.isfinite(np.asarray(a)).all() for a in state):
            raise FloatingPointError(f"Nonfinite update at iteration {iteration}")
        errors = []
        for u, v in zip(old[:4], state[:4]):
            u, v = np.asarray(u), np.asarray(v)
            errors.append(
                float(max(0, 1 - np.corrcoef(u, v)[0, 1]))
                if c.convergence == "correlation" and np.std(u) > 0 and np.std(v) > 0
                else float(np.max(np.abs(u - v)))
            )
        history.append(errors)
        if iteration > 1 and max(errors) < c.tolerance:
            converged = True
            break
    a, b, w, t, vw, vt = state
    flip = (anchor_subject is not None and t[anchor_subject] < jnp.mean(t)) or (
        anchor_subject is None
        and anchor_item is not None
        and w[anchor_item] < jnp.mean(w)
    )
    if flip:
        w, t = -w, -t
    return FitResult(
        dict(alpha=a, beta=b, w=w, theta=t),
        dict(alpha=jnp.full(m, va), beta=jnp.full(n, vb), w=vw, theta=vt),
        converged,
        iteration,
        np.asarray(history),
        ndtr(a[None, :] + b[:, None] - gamma * (t[:, None] - w[None, :]) ** 2),
    )
