"""Dynamic probit IRT with Gaussian forward filtering and smoothing."""

import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr, ndtr
import numpy as np
from ._iteration import run_explicit
from .binary import Control, FitResult
from .types import BinaryStarts, DynamicPriors


def _dynamic_select(state, data):
    return state[0], state[1], state[2].reshape(-1)[data[-1]]


def _dynamic_step(state, data):
    mask, ip, yy, precision, pm, mu0, v0, omega, first, last, _ = data
    n, nt = state[2].shape
    a, b, x, vx, vc, ebb, eba = state
    eta = a[None, :] + x[:, ip] * b[None, :]
    sign = jnp.where(yy == -1, -1.0, 1.0)
    z = eta * sign
    latent = jnp.where(
        mask[:, ip],
        eta
        + jnp.where(
            jnp.abs(yy) == 1,
            sign * jnp.exp(-0.5 * z * z - 0.5 * jnp.log(2 * jnp.pi) - log_ndtr(z)),
            0.0,
        ),
        0.0,
    )
    precision_t = jnp.zeros(nt).at[ip].add(ebb)
    cross = jnp.zeros(nt).at[ip].add(eba)
    info = jnp.zeros((n, nt)).at[:, ip].add(latent * b[None, :]) - cross[None, :]

    def forward(carry, t):
        mean, var = carry
        mean = jnp.where(t == first, mu0, mean)
        pred = jnp.where(t == first, v0, var) + omega
        var = pred / (1 + pred * precision_t[t])
        mean = var * (mean / pred + info[:, t])
        return (mean, var), (mean, var, pred)

    _, (fm, fv, pred) = jax.lax.scan(forward, (mu0, v0), jnp.arange(nt))

    def backward(carry, t):
        next_mean, next_var = carry
        gain = fv[t] / (fv[t] + omega)
        mean = jnp.where(t == last, fm[t], fm[t] + gain * (next_mean - fm[t]))
        var = jnp.where(
            t == last, fv[t], fv[t] + gain * gain * (next_var - fv[t] - omega)
        )
        mean = jnp.where(mask[:, t], mean, 0.0)
        var = jnp.where(mask[:, t], var, 0.0)
        return (mean, var), (mean, var)

    _, (sx, sv) = jax.lax.scan(
        backward, (jnp.zeros(n), jnp.zeros(n)), jnp.arange(nt - 1, -1, -1)
    )
    x = sx[::-1].T
    vx = sv[::-1].T
    sx = jnp.sum(x, axis=0)
    moment = (
        jnp.zeros((nt, 2, 2))
        .at[:, 0, 0]
        .set(jnp.sum(mask, axis=0))
        .at[:, 0, 1]
        .set(sx)
        .at[:, 1, 0]
        .set(sx)
        .at[:, 1, 1]
        .set(jnp.sum(x * x + vx, axis=0))
    )
    vc = jnp.linalg.inv(precision[None, :, :] + moment)
    rhs = (
        jnp.stack([jnp.sum(latent, axis=0), jnp.sum(latent * x[:, ip], axis=0)], axis=1)
        + pm
    )
    ab = jnp.einsum("jab,jb->ja", vc[ip], rhs)
    a, b = ab[:, 0], ab[:, 1]
    return a, b, x, vx, vc, b * b + vc[ip, 1, 1], a * b + vc[ip, 0, 1]


def dyn_irt(
    votes,
    *,
    starts: BinaryStarts,
    priors: DynamicPriors,
    start_period,
    end_period,
    item_period,
    control=None,
):
    """Fit respondent-by-item votes (-1/0/1) with inclusive active periods.

    Priors: x_mu0, x_variance0, omega2 per subject; beta_mu (2,) and
    beta_covariance (2,2). Starts: alpha, beta, x (subjects by periods).
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable JAX x64 before fitting")
    c = control or Control()
    y = np.asarray(votes, dtype=float)
    x = np.asarray(starts["x"], dtype=float)
    if (
        y.ndim != 2
        or x.ndim != 2
        or y.shape[0] != x.shape[0]
        or min(*y.shape, *x.shape) < 1
        or not np.isin(y, [-1, 0, 1]).all()
    ):
        raise ValueError("Invalid votes or starts")
    n, m = y.shape
    nt = x.shape[1]

    def indices(v, size):
        a = np.asarray(v).reshape(-1)
        if (
            a.size != size
            or not np.isfinite(a).all()
            or (a != np.floor(a)).any()
            or (a < 0).any()
            or (a >= nt).any()
        ):
            raise ValueError("Invalid period indices")
        return a.astype(int)

    first = indices(start_period, n)
    last = indices(end_period, n)
    period = indices(item_period, m)
    if (first > last).any():
        raise ValueError("start_period must not exceed end_period")
    a = np.asarray(starts["alpha"], dtype=float).reshape(-1)
    b = np.asarray(starts["beta"], dtype=float).reshape(-1)
    if a.size != m or b.size != m or not all(np.isfinite(v).all() for v in (a, b, x)):
        raise ValueError("Invalid starts")
    p = {
        k: np.asarray(priors[k], dtype=float).reshape(-1)
        for k in ("x_mu0", "x_variance0", "omega2")
    }
    if (
        any(v.size != n or not np.isfinite(v).all() for v in p.values())
        or (p["x_variance0"] <= 0).any()
        or (p["omega2"] < 0).any()
    ):
        raise ValueError("Invalid trajectory priors")
    bm = np.asarray(priors["beta_mu"], dtype=float).reshape(-1)
    bc = np.asarray(priors["beta_covariance"], dtype=float)
    if (
        bm.shape != (2,)
        or bc.shape != (2, 2)
        or not np.isfinite(bm).all()
        or not np.isfinite(bc).all()
        or not np.allclose(bc, bc.T)
        or np.linalg.eigvalsh(bc).min() <= 0
    ):
        raise ValueError("Invalid item priors")
    if (
        c.max_iter < 1
        or c.tolerance < 0
        or not np.isfinite(c.tolerance)
        or c.convergence not in ("absolute", "correlation")
    ):
        raise ValueError("Invalid controls")
    active = (np.arange(nt)[None, :] >= first[:, None]) & (
        np.arange(nt)[None, :] <= last[:, None]
    )
    mask = jnp.asarray(active)
    ip = jnp.asarray(period)
    yy = jnp.asarray(y)
    precision = jnp.linalg.inv(jnp.asarray(bc))
    pm = precision @ bm
    mu0 = jnp.asarray(p["x_mu0"])
    v0 = jnp.asarray(p["x_variance0"])
    omega = jnp.asarray(p["omega2"])

    active_indices = jnp.asarray(np.flatnonzero(active))
    data = (
        mask,
        ip,
        yy,
        precision,
        pm,
        mu0,
        v0,
        omega,
        jnp.asarray(first),
        jnp.asarray(last),
        active_indices,
    )

    def update(state):
        return _dynamic_step(state, data)

    state = (
        jnp.asarray(a),
        jnp.asarray(b),
        jnp.where(mask, x, 0.0),
        jnp.zeros_like(x),
        jnp.zeros((nt, 2, 2)),
        jnp.asarray(b * b),
        jnp.asarray(a * b),
    )
    if c.jit:
        state, converged, iteration, history = run_explicit(
            _dynamic_step,
            state,
            data,
            count=3,
            select=_dynamic_select,
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
            if not all(np.isfinite(np.asarray(v)).all() for v in state):
                raise FloatingPointError(f"Nonfinite update at iteration {iteration}")
            errors = []
            for k, (u, v) in enumerate(zip(old[:3], state[:3])):
                u, v = np.asarray(u), np.asarray(v)
                if k == 2:
                    u, v = u[active], v[active]
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
    a, b, x, vx, vc, _, _ = state
    probabilities = jnp.where(
        mask[:, ip], ndtr(a[None, :] + x[:, ip] * b[None, :]), jnp.nan
    )
    return FitResult(
        dict(alpha=a, beta=b, x=x),
        dict(alpha=vc[ip, 0, 0], beta=vc[ip, 1, 1], x=vx),
        converged,
        iteration,
        np.asarray(history),
        probabilities,
    )
