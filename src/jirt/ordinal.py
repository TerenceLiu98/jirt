"""Three-category ordinal probit ECM estimation."""

import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr, ndtr, gammaln
import numpy as np
from .binary import Control, FitResult
from .types import OrdinalStarts, OrdinalPriors


def _moments(eta, dd, y):
    sd = 1 / jnp.sqrt(dd)
    lo = -eta / sd
    hi = (1 - eta) / sd
    # Reflect positive-tail intervals to avoid subtracting CDFs near one.
    reflect = lo > 0
    a = jnp.where(reflect, -hi, lo)
    b = jnp.where(reflect, -lo, hi)
    lb = log_ndtr(b, series_order=10)
    la = log_ndtr(a, series_order=10)
    logz = lb + jnp.log(-jnp.expm1(la - lb))
    ra = jnp.exp(-0.5 * a * a - 0.5 * jnp.log(2 * jnp.pi) - logz)
    rb = jnp.exp(-0.5 * b * b - 0.5 * jnp.log(2 * jnp.pi) - logz)
    meanmid = (ra - rb) * jnp.where(reflect, -1.0, 1.0)
    varmid = 1 + a * ra - b * rb - (ra - rb) ** 2
    left = jnp.exp(
        -0.5 * lo * lo - 0.5 * jnp.log(2 * jnp.pi) - log_ndtr(lo, series_order=10)
    )
    right = jnp.exp(
        -0.5 * hi * hi - 0.5 * jnp.log(2 * jnp.pi) - log_ndtr(-hi, series_order=10)
    )
    shift = jnp.where(
        y == 1, -left, jnp.where(y == 3, right, jnp.where(y == 2, meanmid, 0.0))
    )
    var = jnp.where(
        y == 1,
        1 - lo * left - left**2,
        jnp.where(y == 3, 1 + hi * right - right**2, jnp.where(y == 2, varmid, 1.0)),
    )
    return eta + sd * shift, sd**2 * var


def ord_irt(responses, *, starts: OrdinalStarts, priors: OrdinalPriors, control=None):
    """Fit categories 1/2/3 with 0 for missing responses.

    starts: tau, beta, DD (per item), x (per respondent).
    priors: x and beta, each with mu and sigma (covariance).
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable JAX x64 before fitting")
    c = control or Control(max_iter=300)
    y = np.asarray(responses, dtype=float)
    if y.ndim != 2 or min(y.shape) < 1 or not np.isin(y, [0, 1, 2, 3]).all():
        raise ValueError("responses must have codes 0,1,2,3")
    n, m = y.shape
    arrays = [
        np.asarray(starts[k], dtype=float).reshape(-1)
        for k in ("tau", "beta", "x", "DD")
    ]
    if (
        [a.size for a in arrays] != [m, m, n, m]
        or not all(np.isfinite(a).all() for a in arrays)
        or (arrays[3] <= 0).any()
    ):
        raise ValueError("Invalid starts")
    xm = float(np.asarray(priors["x"]["mu"]).item())
    xv = float(np.asarray(priors["x"]["sigma"]).item())
    bm = np.asarray(priors["beta"]["mu"], dtype=float).reshape(-1)
    bc = np.asarray(priors["beta"]["sigma"], dtype=float)
    if (
        not np.isfinite(xm)
        or not np.isfinite(xv)
        or xv <= 0
        or bm.shape != (2,)
        or bc.shape != (2, 2)
        or not np.isfinite(bm).all()
        or not np.isfinite(bc).all()
        or not np.allclose(bc, bc.T)
        or np.linalg.eigvalsh(bc).min() <= 0
    ):
        raise ValueError("Invalid priors")
    if (
        c.max_iter < 1
        or c.tolerance < 0
        or not np.isfinite(c.tolerance)
        or c.convergence not in ("absolute", "correlation")
    ):
        raise ValueError("Invalid controls")
    precision = jnp.linalg.inv(jnp.asarray(bc))
    pm = precision @ bm
    yy = jnp.asarray(y)
    # Preserve the integer division in the reference Gamma shape.
    shape = 1 + n // 2

    def step(state):
        t, b, x, dd, vx, vc, rate = state
        ez, _ = _moments(t[None, :] + x[:, None] * b[None, :], dd, yy)
        ebb = b * b
        vx = 1 / (1 / xv + ebb @ dd)
        x = vx * (xm / xv + ez @ (b * dd) - jnp.sum(b * t * dd))
        design = jnp.stack([jnp.ones(n), x], axis=1)
        moment = (design.T @ design).at[1, 1].add(n * vx)
        vc = jnp.linalg.inv(precision[None, :, :] + dd[:, None, None] * moment)
        ab = jnp.einsum("jab,jb->ja", vc, pm[None, :] + dd[:, None] * (ez.T @ design))
        t, b = ab[:, 0], ab[:, 1]
        _, vz = _moments(t[None, :] + x[:, None] * b[None, :], dd, yy)
        rate = 0.5 * jnp.sum(
            ez * ez
            + vz
            + t[None, :] ** 2
            + (x * x + vx)[:, None] * ebb[None, :]
            - 2 * ez * t[None, :]
            - 2 * ez * x[:, None] * b[None, :]
            + 2 * t[None, :] * x[:, None] * b[None, :],
            axis=0,
        )
        dd = shape / rate
        return t, b, x, dd, vx, vc, rate

    update = jax.jit(step) if c.jit else step
    state = (
        *map(jnp.asarray, arrays),
        jnp.asarray(xv),
        jnp.zeros((m, 2, 2)),
        jnp.ones(m),
    )
    history = []
    converged = False
    for iteration in range(1, c.max_iter + 1):
        old = state
        state = update(state)
        if (
            not all(np.isfinite(np.asarray(a)).all() for a in state)
            or (np.asarray(state[3]) <= 0).any()
        ):
            raise FloatingPointError(f"Invalid update at iteration {iteration}")
        errors = []
        for u, v in zip(old[:4], state[:4]):
            u, v = np.asarray(u), np.asarray(v)
            errors.append(
                float(max(0, 1 - np.corrcoef(u, v)[0, 1]))
                if c.convergence == "correlation" and np.std(u) > 0 and np.std(v) > 0
                else float(np.max(np.abs(u - v)))
            )
        history.append(errors)
        if iteration > 2 and max(errors) < c.tolerance:
            converged = True
            break
    t, b, x, dd, vx, vc, rate = state
    delta = jnp.exp(gammaln(shape + 0.5) - gammaln(shape)) / jnp.sqrt(rate)
    eta = t[None, :] + x[:, None] * b[None, :]
    p1 = ndtr(-eta * delta)
    p2 = ndtr((1 - eta) * delta)
    return FitResult(
        dict(x=x, beta=b, tau=t, Delta_sq=dd, Delta=delta),
        dict(x=vx, beta=vc[:, 1, 1], tau=vc[:, 0, 0]),
        converged,
        iteration,
        np.asarray(history),
        jnp.stack([p1, p2 - p1, 1 - p2], axis=-1),
    )
