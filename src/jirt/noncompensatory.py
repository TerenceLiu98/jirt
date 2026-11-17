"""Partially compensatory binary response probabilities."""

from ._optimize import device_minimize

import jax
import jax.numpy as jnp


def _log_responses(theta, a, d, c, loading_mask):
    mask = jnp.asarray(loading_mask, dtype=bool)
    a = jnp.where(mask, jnp.asarray(a), 0.0)
    d = jnp.where(mask, jnp.asarray(d), 0.0)
    z = jnp.asarray(theta)[:, None, :] * a[None, :, :] + d[None, :, :]
    passed = jnp.where(mask[None, :, :], jax.nn.log_sigmoid(z), 0.0)
    # Sum disjoint events: fail this dimension after passing all previous ones.
    # This avoids cancellation in log(1-prod(sigmoid(z))) at high ability.
    prefix = jnp.concatenate(
        [jnp.zeros_like(passed[:, :, :1]), jnp.cumsum(passed, axis=-1)[:, :, :-1]],
        axis=-1,
    )
    failed = jnp.where(mask[None, :, :], prefix + jax.nn.log_sigmoid(-z), -jnp.inf)
    log_fail = jax.scipy.special.logsumexp(failed, axis=-1)
    log_no = jnp.log1p(-c) + log_fail
    log_yes = jnp.logaddexp(jnp.log(c), jnp.log1p(-c) + passed.sum(axis=-1))
    return log_yes, log_no


def noncompensatory_probabilities(theta, *, a, d, loading_mask, c=None, model="pc2pl"):
    """Return success probabilities c+(1-c)*prod(sigmoid(a*theta+d)).

    theta is (N,D); a, d and the explicit boolean loading_mask are (J,D).
    Inactive dimensions are excluded from the product. Every item needs at
    least one active dimension. Active slopes are positive. PC1PL fixes them
    to one; PC1PL/PC2PL fix guessing to zero. PC3PL accepts c of shape (J,).
    This pure JAX kernel assumes valid input shapes and parameter restrictions.
    """
    if model not in ("pc1pl", "pc2pl", "pc3pl"):
        raise ValueError("model must be pc1pl, pc2pl or pc3pl")
    a = jnp.asarray(a)
    if model == "pc1pl":
        a = jnp.ones_like(a)
    if c is None or model != "pc3pl":
        c = jnp.zeros(a.shape[0])
    return jnp.exp(_log_responses(theta, a, d, jnp.asarray(c), loading_mask)[0])


from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

from .multidimensional import _quadrature


@dataclass
class NoncompensatoryResult:
    model: str
    means: dict
    covariances: dict
    loading_mask: np.ndarray
    converged: bool
    iterations: int
    log_likelihood: float
    trace: np.ndarray
    message: str

    @property
    def variances(self):
        return {"x": jnp.diagonal(self.covariances["x"], axis1=-2, axis2=-1)}

    def predict(self, theta=None):
        return noncompensatory_probabilities(
            self.means["x"] if theta is None else theta,
            a=self.means["a"],
            d=self.means["d"],
            c=self.means["c"],
            loading_mask=self.loading_mask,
            model=self.model,
        )

    @property
    def probabilities(self):
        return self.predict()


def _unpack(raw, design, model):
    n = design.shape[-1]
    d = jnp.einsum("jdk,k->jd", design, raw[:n])
    a = (
        design.sum(axis=-1)
        if model == "pc1pl"
        else jnp.einsum("jdk,k->jd", design, raw[n : 2 * n])
    )
    c = raw[2 * n :] if model == "pc3pl" else jnp.zeros(design.shape[0])
    return a, d, c


def _posterior(raw, patterns, nodes, lw, design, model):
    a, d, c = _unpack(raw, design, model)
    yes, no = _log_responses(nodes, a, d, c, design.sum(axis=-1) > 0)
    joint = (patterns == 1) @ yes.T + (patterns == 0) @ no.T + lw
    return jax.scipy.special.logsumexp(joint, axis=1), jax.nn.softmax(joint, axis=1)


def _loss(raw, patterns, frequency, nodes, lw, design, model):
    ll, _ = _posterior(raw, patterns, nodes, lw, design, model)
    return -(ll @ frequency) / frequency.sum()


_value_grad = jax.jit(jax.value_and_grad(_loss), static_argnames=("model",))
_score = jax.jit(_posterior, static_argnames=("model",))


def noncompensatory_irt(
    responses,
    *,
    loading_mask,
    model="pc2pl",
    starts=None,
    quadrature_points=15,
    max_nodes=100000,
    max_iter=1000,
    tolerance=1e-10,
    jit=True,
    optimizer="scipy",
):
    """Fit PC1PL/PC2PL/PC3PL using tensor Gauss-Hermite marginal likelihood.

    Responses are binary with NaN missing. The required boolean (J,D) mask
    excludes inactive dimensions from each item's product. Abilities have
    fixed independent N(0,1) distributions, all active slopes are positive,
    and PC1PL slopes equal one. Each active dimension has its own intercept.
    PC3PL estimates one guessing probability per item. No item priors apply.

    Distinct mask columns label factors; identical columns leave a permutation
    ambiguity. Supply substantive anchors in the mask and adequate items per
    factor for identification; the mask is not a certificate of identification.
    Starts accepts d and free a of shape (J,D), and c (J,) for PC3PL.
    Inactive entries must be zero. Posterior covariance is conditional on the
    estimated item parameters and does not include calibration uncertainty.
    optimizer="scipy" retains SLSQP. optimizer="jax" requires jit=True and
    uses device-side spectral projected gradient with the same constraints.
    For JAX, tolerance bounds the unit-step projected-gradient infinity norm;
    trace contains accepted iterates and may be nonmonotone. See PERFORMANCE.md.
    """
    if optimizer not in ("scipy", "jax") or (optimizer == "jax" and not jit):
        raise ValueError("optimizer must be scipy or jax; jax requires jit=True")
    if not jax.config.x64_enabled:
        raise ValueError("Enable JAX x64 before fitting")
    if model not in ("pc1pl", "pc2pl", "pc3pl"):
        raise ValueError("model must be pc1pl, pc2pl or pc3pl")
    for v, name, minimum in (
        (quadrature_points, "quadrature_points", 3),
        (max_nodes, "max_nodes", 1),
        (max_iter, "max_iter", 1),
    ):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not isinstance(jit, bool):
        raise ValueError("jit must be boolean")
    if not np.isscalar(tolerance) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    y = np.asarray(responses, dtype=float)
    if y.ndim != 2 or min(y.shape) < 2:
        raise ValueError("responses must be a matrix with at least two subjects/items")
    if not (np.isnan(y) | (y == 0) | (y == 1)).all():
        raise ValueError("responses must be binary or NaN")
    if np.any(np.all(np.isnan(y), axis=0)):
        raise ValueError("Each item needs an observed response")
    mask = np.asarray(loading_mask)
    if (
        mask.ndim != 2
        or mask.shape[0] != y.shape[1]
        or mask.shape[1] < 1
        or mask.dtype.kind != "b"
        or not mask.any(axis=1).all()
        or not mask.any(axis=0).all()
    ):
        raise ValueError(
            "loading_mask must be boolean (items,dimensions) with active rows/columns"
        )
    nodes, lw = _quadrature(mask.shape[1], quadrature_points, max_nodes)
    n = int(mask.sum())
    design = np.zeros((*mask.shape, n))
    design[mask] = np.eye(n)
    start = dict(starts or {})
    allowed = (
        {"d"}
        | ({"a"} if model != "pc1pl" else set())
        | ({"c"} if model == "pc3pl" else set())
    )
    if set(start) - allowed:
        raise ValueError("Unexpected starts")
    d = np.asarray(start.get("d", np.zeros(mask.shape)), dtype=float)
    if d.shape != mask.shape or not np.isfinite(d).all() or np.any(d[~mask] != 0):
        raise ValueError("Invalid intercept starts")
    pars = d[mask]
    lower, upper = np.full(n, -np.inf), np.full(n, np.inf)
    if model != "pc1pl":
        a = np.asarray(start.get("a", mask.astype(float)), dtype=float)
        if (
            a.shape != mask.shape
            or not np.isfinite(a).all()
            or np.any(a[mask] < 1e-6)
            or np.any(a[~mask] != 0)
        ):
            raise ValueError("Invalid positive slope starts")
        pars = np.r_[pars, a[mask]]
        lower, upper = np.r_[lower, np.full(n, 1e-6)], np.r_[upper, np.full(n, np.inf)]
    if model == "pc3pl":
        c = np.asarray(start.get("c", np.full(y.shape[1], 0.1)), dtype=float)
        if (
            c.shape != (y.shape[1],)
            or not np.isfinite(c).all()
            or np.any((c < 1e-6) | (c > 1 - 1e-6))
        ):
            raise ValueError("Invalid guessing starts")
        pars = np.r_[pars, c]
        lower, upper = (
            np.r_[lower, np.full(len(c), 1e-6)],
            np.r_[upper, np.full(len(c), 1 - 1e-6)],
        )
    patterns, inverse, frequency = np.unique(
        np.where(np.isnan(y), -1, y).astype(int),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    args = tuple(jnp.asarray(v) for v in (patterns, frequency, nodes, lw, design))
    evaluate = _value_grad if jit else jax.value_and_grad(_loss)

    def objective(raw):
        value, grad = evaluate(jnp.asarray(raw), *args, model=model)
        if not np.isfinite(value) or not np.isfinite(grad).all():
            raise FloatingPointError("Nonfinite noncompensatory objective")
        return float(value), np.asarray(grad)

    trace = [-objective(pars)[0] * len(y)]
    if optimizer == "jax":
        opt = device_minimize(
            _loss,
            pars,
            args,
            lower=lower,
            upper=upper,
            constraints=(),
            config=(("model", model),),
            max_iter=max_iter,
            tolerance=tolerance,
        )
        trace = -opt.trace * len(y)
    else:
        opt = minimize(
            objective,
            pars,
            jac=True,
            method="SLSQP",
            bounds=list(zip(lower, upper)),
            callback=lambda raw: trace.append(-objective(raw)[0] * len(y)),
            options={"maxiter": max_iter, "ftol": tolerance},
        )
    ll, post = (_score if jit else _posterior)(
        jnp.asarray(opt.x), args[0], args[2], args[3], args[4], model=model
    )
    a, d, c = _unpack(jnp.asarray(opt.x), args[4], model)
    x = post @ args[2]
    centered = args[2][None, :, :] - x[:, None, :]
    covariance = jnp.einsum("pq,pqd,pqe->pde", post, centered, centered)
    return NoncompensatoryResult(
        model,
        dict(a=a, d=d, c=c, x=x[inverse]),
        {"x": covariance[inverse]},
        mask.copy(),
        bool(opt.success),
        int(opt.nit),
        float(ll @ args[1]),
        np.asarray(trace),
        str(opt.message),
    )
