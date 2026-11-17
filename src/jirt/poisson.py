"""Variational Poisson text scaling."""

import jax
import jax.numpy as jnp
import numpy as np
from ._iteration import run_explicit
from .binary import Control, FitResult
from .types import PoissonStarts, PoissonPriors


def _poisson_step(state, data):
    yy, ix, mu, precision = data
    pa, pp, pb, px = precision
    ma, mp, mb, mx = mu
    ni = state[3].shape[0]

    def aggregate(v):
        return jnp.zeros(ni).at[ix].add(v)

    a, p, b, x, vb, vx, xi = state
    xi = a[None, :] + p[:, None] + b[:, None] * x[ix][None, :]
    e = jnp.exp(xi)
    a = (
        jnp.sum(
            yy - e - e * (p[:, None] + b[:, None] * x[ix][None, :]) + e * xi, axis=0
        )
        + ma * pa
    ) / (jnp.sum(e, axis=0) + pa)
    vb = 1 / (e @ (x * x + vx)[ix] + pb)
    vx = 1 / (aggregate((b * b + vb) @ e) + px)
    residual = yy - e + e * xi - e * (a[None, :] + p[:, None])
    x = vx * (aggregate(b @ residual) + mx * px)
    b = vb * (residual @ x[ix] + mb * pb)
    p = (
        jnp.sum(
            yy - e + e * xi - e * (a[None, :] + b[:, None] * x[ix][None, :]), axis=1
        )
        + mp * pp
    ) / (jnp.sum(e, axis=1) + pp)
    return a, p, b, x, vb, vx, xi


def pois_irt(
    counts,
    *,
    starts: PoissonStarts,
    priors: PoissonPriors,
    document_subject=None,
    control=None,
):
    """Fit term-by-document counts with optional repeated documents per subject.

    Priors map alpha/psi/beta/x to dictionaries containing mu and sigma2.
    Starts use the same parameter names; alpha is per document, psi/beta per
    term, and x per subject. The result's probabilities field is None.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable JAX x64 before fitting")
    c = control or Control(max_iter=1500)
    y = np.asarray(counts, dtype=float)
    if (
        y.ndim != 2
        or min(y.shape) < 1
        or not np.isfinite(y).all()
        or (y < 0).any()
        or (y != np.floor(y)).any()
    ):
        raise ValueError("counts must be a finite nonnegative integer matrix")
    nj, nk = y.shape
    params = [
        np.asarray(starts[k], dtype=float).reshape(-1)
        for k in ("alpha", "psi", "beta", "x")
    ]
    ni = len(params[3])
    idx = np.arange(nk) if document_subject is None else np.asarray(document_subject)
    if (
        idx.shape != (nk,)
        or not np.issubdtype(idx.dtype, np.integer)
        or (idx < 0).any()
        or (idx >= ni).any()
    ):
        raise ValueError("Invalid document_subject mapping")
    if (
        [len(a) for a in params] != [nk, nj, nj, ni]
        or ni < 1
        or not all(np.isfinite(a).all() for a in params)
    ):
        raise ValueError("Invalid starts")
    mu = []
    precision = []
    for k in ("alpha", "psi", "beta", "x"):
        m = float(np.asarray(priors[k]["mu"]).item())
        v = float(np.asarray(priors[k]["sigma2"]).item())
        if not np.isfinite(m) or not np.isfinite(v) or v <= 0:
            raise ValueError("Priors require finite means and positive variances")
        mu.append(m)
        precision.append(1 / v)
    if (
        c.max_iter < 1
        or c.tolerance < 0
        or not np.isfinite(c.tolerance)
        or c.convergence not in ("absolute", "correlation")
    ):
        raise ValueError("Invalid controls")
    yy = jnp.asarray(y)
    ix = jnp.asarray(idx)

    data = (yy, ix, jnp.asarray(mu), jnp.asarray(precision))

    def step(state):
        return _poisson_step(state, data)

    update = step
    state = (*map(jnp.asarray, params), jnp.ones(nj), jnp.ones(ni), jnp.zeros_like(yy))
    if c.jit:
        state, converged, iteration, history = run_explicit(
            _poisson_step,
            state,
            data,
            count=4,
            max_iter=c.max_iter,
            tolerance=c.tolerance,
            correlation=c.convergence == "correlation",
            min_iter=3,
        )
    else:
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
                    if c.convergence == "correlation"
                    and np.std(u) > 0
                    and np.std(v) > 0
                    else float(np.max(np.abs(u - v)))
                )
            history.append(errors)
            if iteration > 2 and max(errors) < c.tolerance:
                converged = True
                break
    a, p, b, x, vb, vx, xi = state
    return FitResult(
        dict(alpha=a, psi=p, beta=b, x=x, xi=xi),
        dict(beta=vb, x=vx),
        converged,
        iteration,
        np.asarray(history),
        None,
    )
