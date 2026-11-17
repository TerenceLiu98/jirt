"""Generalized graded unfolding responses for unidimensional ideal points."""

import jax
import jax.numpy as jnp


def _log_probabilities(theta, a, delta, tau, category_counts):
    theta = jnp.atleast_1d(jnp.asarray(theta))
    a, delta, tau = map(jnp.asarray, (a, delta, tau))
    counts = jnp.asarray(category_counts)
    valid_steps = jnp.arange(tau.shape[1])[None, :] < counts[:, None] - 1
    tau = jnp.where(valid_steps, tau, 0.0)
    cumulative = jnp.concatenate(
        [jnp.zeros_like(tau[:, :1]), jnp.cumsum(tau, axis=1)], axis=1
    )
    k = jnp.arange(tau.shape[1] + 1)
    z = a[None, :, None] * (theta[:, None, None] - delta[None, :, None])
    v = (counts[None, :, None] - 0.5 - k[None, None, :]) * z
    # Center the paired exponentials by their common M*z/2 term. The signed
    # form is smooth at theta=delta and symmetric without differentiating abs.
    logits = a[None, :, None] * cumulative[None, :, :] + jnp.logaddexp(v, -v)
    valid = k[None, None, :] < counts[None, :, None]
    return jax.nn.log_softmax(jnp.where(valid, logits, -jnp.inf), axis=-1)


def ggum_probabilities(theta, *, a, delta, tau, category_counts):
    """GGUM category probabilities, padded with zeros to max_categories.

    theta is (N,), a and delta are (J,), tau is (J,max(K)-1). Counts are
    explicit integers >=2. Categories increase with agreement. Positive a
    controls discrimination; delta is the ideal location. tau uses mirt's
    positive distance convention (the negative of the lower-side thresholds
    in the original GGUM). Valid thresholds satisfy tau_1 > ... > tau_C > 0;
    padded entries are ignored. This JAX kernel assumes valid inputs.
    """
    return jnp.exp(_log_probabilities(theta, a, delta, tau, category_counts))


from dataclasses import dataclass
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize, LinearConstraint


@dataclass
class GGUMResult:
    means: dict
    variances: dict
    category_counts: np.ndarray
    anchor_item: int
    converged: bool
    iterations: int
    log_likelihood: float
    trace: np.ndarray
    message: str

    def predict(self, theta=None):
        return ggum_probabilities(
            self.means["x"] if theta is None else theta,
            a=self.means["a"],
            delta=self.means["delta"],
            tau=self.means["tau"],
            category_counts=self.category_counts,
        )

    @property
    def probabilities(self):
        return self.predict()


def _unpack(raw, design):
    j = design.shape[0]
    return raw[:j], raw[j : 2 * j], jnp.einsum("jsk,k->js", design, raw[2 * j :])


def _posterior(raw, patterns, nodes, lw, design, counts):
    a, delta, tau = _unpack(raw, design)
    lp = _log_probabilities(nodes, a, delta, tau, counts)
    selected = lp[:, jnp.arange(patterns.shape[1])[None, :], jnp.maximum(patterns, 0)]
    joint = jnp.sum(jnp.where(patterns[None, :, :] >= 0, selected, 0.0), axis=2).T + lw
    return jax.scipy.special.logsumexp(joint, axis=1), jax.nn.softmax(joint, axis=1)


def _loss(raw, patterns, frequency, nodes, lw, design, counts):
    ll, _ = _posterior(raw, patterns, nodes, lw, design, counts)
    return -(ll @ frequency) / frequency.sum()


_value_grad = jax.jit(jax.value_and_grad(_loss))
_score = jax.jit(_posterior)


def ggum_irt(
    responses,
    *,
    category_counts,
    starts,
    anchor_item=0,
    quadrature_points=61,
    max_iter=2000,
    tolerance=1e-10,
    jit=True,
):
    """Fit unidimensional binary/polytomous GGUM by quadrature MML.

    Responses are integer categories 0..K_j-1, with NaN missing. Abilities
    have fixed N(0,1) distribution and no item priors apply. Discriminations
    are positive; thresholds use tau_1 > ... > tau_C > 0 (mirt convention).
    anchor_item's delta is positive to fix reflection. Other deltas are free.
    Supply starts with a (J,), delta (J,), tau (J,max(K)-1). Nonconvex ideal
    point likelihoods require substantive starting locations and preferably
    multiple starts; convergence does not establish a global optimum.
    Returns conditional EAP scores, which can conceal bimodal posteriors.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable JAX x64 before fitting")
    for value, name, minimum in (
        (quadrature_points, "quadrature_points", 3),
        (max_iter, "max_iter", 1),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not isinstance(jit, bool):
        raise ValueError("jit must be boolean")
    if not np.isscalar(tolerance) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    y = np.asarray(responses, dtype=float)
    if y.ndim != 2 or min(y.shape) < 2:
        raise ValueError("responses must be a matrix with at least two subjects/items")
    j = y.shape[1]
    counts = np.asarray(category_counts)
    if counts.shape != (j,) or counts.dtype.kind not in "iu" or np.any(counts < 2):
        raise ValueError("category_counts must contain an integer >=2 per item")
    if not (np.isnan(y) | ((y >= 0) & (y < counts) & (y == np.floor(y)))).all():
        raise ValueError("responses must be declared integer categories or NaN")
    if np.any(np.all(np.isnan(y), axis=0)):
        raise ValueError("Each item needs an observed response")
    if (
        isinstance(anchor_item, bool)
        or not isinstance(anchor_item, (int, np.integer))
        or not 0 <= anchor_item < j
    ):
        raise ValueError("anchor_item must index an item")
    start = dict(starts)
    if set(start) != {"a", "delta", "tau"}:
        raise ValueError("starts requires a, delta and tau")
    a, delta, tau = (np.asarray(start[k], dtype=float) for k in ("a", "delta", "tau"))
    if a.shape != (j,) or not np.isfinite(a).all() or np.any(a < 1e-6):
        raise ValueError("Invalid positive slope starts")
    if delta.shape != (j,) or not np.isfinite(delta).all() or delta[anchor_item] < 1e-6:
        raise ValueError("Invalid locations: anchor delta must be positive")
    mask = np.arange(counts.max() - 1)[None, :] < counts[:, None] - 1
    if (
        tau.shape != mask.shape
        or not np.isfinite(tau[mask]).all()
        or np.any(tau[mask] < 1e-6)
    ):
        raise ValueError("Invalid positive threshold starts")
    n = int(mask.sum())
    design = np.zeros((*mask.shape, n))
    design[mask] = np.eye(n)
    raw = np.r_[a, delta, tau[mask]]
    lower = np.r_[np.full(j, 1e-6), np.full(j, -np.inf), np.full(n, 1e-6)]
    lower[j + anchor_item] = 1e-6
    rows = []
    for item, k in enumerate(counts - 1):
        for step in range(k - 1):
            rows.append(
                np.r_[np.zeros(2 * j), design[item, step] - design[item, step + 1]]
            )
    constraints = []
    if rows:
        order = np.asarray(rows)
        if np.any(order @ raw < 1e-6):
            raise ValueError("Thresholds must be strictly decreasing")
        constraints = [LinearConstraint(order, 1e-6, np.inf)]
    patterns, inverse, frequency = np.unique(
        np.where(np.isnan(y), -1, y).astype(int),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    nodes, w = hermgauss(quadrature_points)
    nodes *= np.sqrt(2)
    lw = np.log(w) - 0.5 * np.log(np.pi)
    args = tuple(
        jnp.asarray(v) for v in (patterns, frequency, nodes, lw, design, counts)
    )
    evaluate = _value_grad if jit else jax.value_and_grad(_loss)

    def objective(raw):
        value, grad = evaluate(jnp.asarray(raw), *args)
        if not np.isfinite(value) or not np.isfinite(grad).all():
            raise FloatingPointError("Nonfinite GGUM objective")
        return float(value), np.asarray(grad)

    trace = [-objective(raw)[0] * len(y)]
    opt = minimize(
        objective,
        raw,
        jac=True,
        method="SLSQP",
        bounds=list(zip(lower, np.full(len(raw), np.inf))),
        constraints=constraints,
        callback=lambda raw: trace.append(-objective(raw)[0] * len(y)),
        options={"maxiter": max_iter, "ftol": tolerance},
    )
    ll, post = (_score if jit else _posterior)(
        jnp.asarray(opt.x), args[0], args[2], args[3], args[4], args[5]
    )
    a, delta, tau = _unpack(jnp.asarray(opt.x), args[4])
    x = post @ args[2]
    var = jnp.sum(post * (args[2][None, :] - x[:, None]) ** 2, axis=1)
    return GGUMResult(
        dict(a=a, delta=delta, tau=tau, x=x[inverse]),
        {"x": var[inverse]},
        counts.copy(),
        int(anchor_item),
        bool(opt.success),
        int(opt.nit),
        float(ll @ args[1]),
        np.asarray(trace),
        str(opt.message),
    )
