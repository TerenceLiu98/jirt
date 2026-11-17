"""Hierarchical probit IRT using indexed observations and group regressions."""

import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr, ndtr
import numpy as np
from ._iteration import run_explicit
from .binary import Control, FitResult
from .types import HierarchicalStarts, HierarchicalPriors


def _aggregate(v, ix, count):
    return jnp.zeros((count,) + v.shape[1:]).at[ix].add(v)


def _hierarchical_select(state, data):
    return (state[0], state[1], *(state[2][:, k] for k in range(state[2].shape[1])))


def _hierarchical_step(state, data):
    ii, jj, gg, zz, yy, bp, bm, gp, gm, outer, ss, sv, count = data
    ni, nj, ng = state[3].shape[0], state[0].shape[0], state[2].shape[0]
    a, b, ga, eta, sigma, ve, vg, vb, ebb, eba = state
    regression = jnp.sum(ga[gg] * zz, axis=1)
    linear = a[jj] + b[jj] * (regression[ii] + eta[ii])
    sign = jnp.where(yy == -1, -1.0, 1.0)
    q = sign * linear
    latent = linear + jnp.where(
        jnp.abs(yy) == 1,
        sign * jnp.exp(-0.5 * q * q - 0.5 * jnp.log(2 * jnp.pi) - log_ndtr(q)),
        0.0,
    )
    ve = 1 / (1 / sigma[gg] + _aggregate(ebb[jj], ii, ni))
    eta = ve * _aggregate(latent * b[jj] - eba[jj] - ebb[jj] * regression[ii], ii, ni)
    x = regression + eta
    xx = x * x + ve + jnp.einsum("nd,ndk,nk->n", zz, vg[gg], zz)
    moment = (
        jnp.ones((ni, 2, 2)).at[:, 0, 1].set(x).at[:, 1, 0].set(x).at[:, 1, 1].set(xx)
    )
    vb = jnp.linalg.inv(bp[None, :, :] + _aggregate(moment[ii], jj, nj))
    design = jnp.stack([jnp.ones(ni), x], axis=1)
    ab = jnp.einsum(
        "jab,jb->ja", vb, bm + _aggregate(latent[:, None] * design[ii], jj, nj)
    )
    a, b = ab[:, 0], ab[:, 1]
    ebb = b * b + vb[:, 1, 1]
    eba = a * b + vb[:, 0, 1]
    # Aggregate over subjects first to avoid an observations-by-D-by-D tensor.
    weight = _aggregate(ebb[jj], ii, ni)
    vg = jnp.linalg.inv(
        gp[None, :, :] + _aggregate(outer * weight[:, None, None], gg, ng)
    )
    rhs = _aggregate(b[jj] * (latent - a[jj]) - ebb[jj] * eta[ii], ii, ni)
    ga = jnp.einsum("gab,gb->ga", vg, gm + _aggregate(zz * rhs[:, None], gg, ng))
    sigma = (ss + _aggregate(eta * eta + ve, gg, ng)) / (sv + count)
    return a, b, ga, eta, sigma, ve, vg, vb, ebb, eba


def hier_irt(
    votes,
    *,
    subject,
    item,
    groups,
    covariates,
    starts: HierarchicalStarts,
    priors: HierarchicalPriors,
    control: Control | None = None,
):
    """Fit indexed observations; priors use beta_mu/beta_covariance,
    gamma_mu/gamma_covariance, sigma_v and sigma_s. Covariance arrays in the
    result put item/group index first. Sigma denotes the residual variance.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable JAX x64 before fitting")
    c = control or Control()
    y = np.asarray(votes, dtype=float).reshape(-1)
    z = np.asarray(covariates, dtype=float)
    ga = np.asarray(starts["gamma"], dtype=float)
    if (
        z.ndim != 2
        or ga.ndim != 2
        or z.shape[1] != ga.shape[1]
        or min(*z.shape, *ga.shape) < 1
        or not np.isfinite(z).all()
        or not np.isfinite(ga).all()
    ):
        raise ValueError("Invalid covariates or gamma")
    ni, nd = z.shape
    ng = ga.shape[0]
    a, b, eta, sigma = [
        np.asarray(starts[k], dtype=float).reshape(-1)
        for k in ("alpha", "beta", "eta", "sigma")
    ]
    nj = a.size
    if (
        nj < 1
        or b.size != nj
        or eta.size != ni
        or sigma.size != ng
        or (sigma <= 0).any()
        or not all(np.isfinite(v).all() for v in (a, b, eta, sigma))
    ):
        raise ValueError("Invalid starts")
    if y.size < 1 or not np.isin(y, [-1, 0, 1]).all():
        raise ValueError("Invalid votes")

    def index(v, size, upper):
        v = np.asarray(v).reshape(-1)
        if (
            v.size != size
            or not np.isfinite(v).all()
            or (v != np.floor(v)).any()
            or (v < 0).any()
            or (v >= upper).any()
        ):
            raise ValueError("Invalid indices")
        return jnp.asarray(v, dtype=int)

    ii = index(subject, y.size, ni)
    jj = index(item, y.size, nj)
    gg = index(groups, ni, ng)

    def prior(prefix, size):
        mu = np.asarray(priors[prefix + "_mu"], dtype=float).reshape(-1)
        cov = np.asarray(priors[prefix + "_covariance"], dtype=float)
        if (
            mu.shape != (size,)
            or cov.shape != (size, size)
            or not np.isfinite(mu).all()
            or not np.isfinite(cov).all()
            or not np.allclose(cov, cov.T)
            or np.linalg.eigvalsh(cov).min() <= 0
        ):
            raise ValueError("Invalid priors")
        prec = jnp.linalg.inv(jnp.asarray(cov))
        return prec, prec @ mu

    bp, bm = prior("beta", 2)
    gp, gm = prior("gamma", nd)
    sv = float(np.asarray(priors["sigma_v"]).item())
    ss = float(np.asarray(priors["sigma_s"]).item())
    if not np.isfinite(sv) or not np.isfinite(ss) or sv < 0 or ss < 0:
        raise ValueError("Invalid sigma prior")
    if (
        c.max_iter < 1
        or c.tolerance < 0
        or not np.isfinite(c.tolerance)
        or c.convergence not in ("absolute", "correlation")
    ):
        raise ValueError("Invalid controls")
    zz = jnp.asarray(z)
    yy = jnp.asarray(y)

    def aggregate(v, ix, count):
        return jnp.zeros((count,) + v.shape[1:]).at[ix].add(v)

    count = aggregate(jnp.ones(ni), gg, ng)
    if np.any(np.asarray(count) + sv <= 0):
        raise ValueError("Empty group requires positive sigma_v")
    outer = zz[:, :, None] * zz[:, None, :]

    data = (ii, jj, gg, zz, yy, bp, bm, gp, gm, outer, ss, sv, count)

    def update(state):
        return _hierarchical_step(state, data)

    state = (
        *map(jnp.asarray, (a, b, ga, eta, sigma)),
        jnp.zeros(ni),
        jnp.zeros((ng, nd, nd)),
        jnp.zeros((nj, 2, 2)),
        jnp.asarray(b * b),
        jnp.asarray(a * b),
    )
    if c.jit:
        state, converged, iteration, history = run_explicit(
            _hierarchical_step,
            state,
            data,
            count=2 + nd,
            select=_hierarchical_select,
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
            pairs = [(old[0], state[0]), (old[1], state[1])] + [
                (old[2][:, k], state[2][:, k]) for k in range(nd)
            ]
            for u, v in pairs:
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
    a, b, ga, eta, sigma, ve, vg, vb, _, _ = state
    x = jnp.sum(ga[gg] * zz, axis=1) + eta
    return FitResult(
        dict(alpha=a, beta=b, gamma=ga, eta=eta, sigma=sigma, x_implied=x),
        dict(eta=ve, gamma=vg, beta2=vb),
        converged,
        iteration,
        np.asarray(history),
        ndtr(a[jj] + b[jj] * x[ii]),
    )
