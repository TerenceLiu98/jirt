"""Logistic ordered-response models with explicitly declared categories."""

import jax
import jax.numpy as jnp


def _log_probabilities(theta, a, steps, category_counts, model):
    """Padded subject/item/category log probabilities; unused categories are -inf."""
    theta = jnp.atleast_1d(jnp.asarray(theta))
    a = jnp.asarray(a)
    steps = jnp.asarray(steps)
    counts = jnp.asarray(category_counts)
    valid_steps = jnp.arange(steps.shape[1])[None, :] < counts[:, None] - 1
    # Padding must not enter arithmetic, even when supplied as NaN.
    steps = jnp.where(valid_steps, steps, 0.0)
    z = a[None, :, None] * (theta[:, None, None] - steps[None, :, :])
    if model == "grm":
        first = jax.nn.log_sigmoid(-z[:, :, :1])
        last = jax.nn.log_sigmoid(z)
        # sigmoid(x)-sigmoid(y), x>y, evaluated without cancellation.
        gap = a[:, None] * (steps[:, 1:] - steps[:, :-1])
        interior = (
            jax.nn.log_sigmoid(z[:, :, :-1])
            + jax.nn.log_sigmoid(-z[:, :, 1:])
            + jnp.log(-jnp.expm1(-jnp.where(gap > 0, gap, 1.0)))[None, :, :]
        )
        middle = jnp.concatenate([interior, jnp.zeros_like(first)], axis=2)
        index = jnp.arange(steps.shape[1])[None, None, :]
        rest = jnp.where(index == counts[None, :, None] - 2, last, middle)
        logp = jnp.concatenate([first, rest], axis=2)
    else:
        increments = jnp.where(valid_steps[None, :, :], z, 0.0)
        logits = jnp.concatenate(
            [jnp.zeros_like(z[:, :, :1]), jnp.cumsum(increments, axis=2)], axis=2
        )
        valid = jnp.arange(logits.shape[2])[None, None, :] < counts[None, :, None]
        logp = jax.nn.log_softmax(jnp.where(valid, logits, -jnp.inf), axis=2)
    valid = jnp.arange(logp.shape[2])[None, None, :] < counts[None, :, None]
    return jnp.where(valid, logp, -jnp.inf)


def polytomous_probabilities(theta, *, a, steps, category_counts, model="grm"):
    """Return (subjects, items, max_categories) probabilities, padded with zero.

    GRM steps are strictly increasing thresholds. GPCM steps are adjacent-category
    difficulties and need not be ordered. PCM uses unit slopes; RSM additionally
    uses steps b_j + tau_k with shared centered tau. Supply those restrictions in
    the inputs when using this pure JAX prediction function. Categories start at
    zero. Inputs must have valid shapes, positive slopes and declared counts >=2.
    """
    if model not in ("grm", "pcm", "gpcm", "rsm"):
        raise ValueError("Unknown polytomous model")
    if model in ("pcm", "rsm"):
        a = jnp.ones_like(jnp.asarray(a))
    return jnp.exp(_log_probabilities(theta, a, steps, category_counts, model))


from dataclasses import dataclass
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize, LinearConstraint


@dataclass
class PolytomousResult:
    model: str
    means: dict
    variances: dict
    category_counts: np.ndarray
    converged: bool
    iterations: int
    log_likelihood: float
    trace: np.ndarray
    message: str

    def predict(self, theta=None):
        return polytomous_probabilities(
            self.means["x"] if theta is None else theta,
            a=self.means["a"],
            steps=self.means["steps"],
            category_counts=self.category_counts,
            model=self.model,
        )

    @property
    def probabilities(self):
        return self.predict()


def _unpack(raw, design, counts, model):
    j = design.shape[0]
    n = design.shape[2]
    steps = jnp.einsum("jsk,k->js", design, raw[:n])
    a = raw[n:] if model in ("grm", "gpcm") else jnp.ones(j)
    return a, steps


def _posterior(raw, y, nodes, lw, design, counts, model):
    a, steps = _unpack(raw, design, counts, model)
    lp = _log_probabilities(nodes, a, steps, counts, model)
    # Gather categories rather than multiply zeros by padded -inf values.
    selected = lp[:, jnp.arange(y.shape[1])[None, :], jnp.maximum(y, 0)]
    joint = jnp.sum(jnp.where(y[None, :, :] >= 0, selected, 0.0), axis=2).T + lw
    return jax.scipy.special.logsumexp(joint, axis=1), jax.nn.softmax(joint, axis=1)


def _loss(raw, y, frequency, nodes, lw, design, counts, model):
    ll, _ = _posterior(raw, y, nodes, lw, design, counts, model)
    return -jnp.sum(ll * frequency) / jnp.sum(frequency)


_value_grad = jax.jit(jax.value_and_grad(_loss), static_argnames=("model",))
_score = jax.jit(_posterior, static_argnames=("model",))


def polytomous_irt(
    responses,
    *,
    category_counts,
    model="grm",
    starts=None,
    quadrature_points=41,
    max_iter=1000,
    tolerance=1e-10,
    jit=True,
):
    """Fit GRM, PCM, GPCM or RSM with fixed N(0,1) ability and no item priors.

    Responses use integer categories 0..K_j-1 and NaN missing. Counts must be
    declared even for unobserved categories. RSM requires equal category counts
    and estimates steps b_j + tau_k with sum(tau)=0. PCM/RSM fix slopes to one;
    GRM/GPCM estimate positive slopes. Only GRM requires increasing steps.
    starts accepts padded steps (J,max(K)-1) and free slopes a (J,).
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable JAX x64 before fitting")
    if model not in ("grm", "pcm", "gpcm", "rsm"):
        raise ValueError("Unknown polytomous model")
    for v, name, minimum in (
        (quadrature_points, "quadrature_points", 3),
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
    counts = np.asarray(category_counts)
    if (
        counts.shape != (y.shape[1],)
        or counts.dtype.kind not in "iu"
        or np.any(counts < 2)
    ):
        raise ValueError("category_counts must contain an integer >=2 per item")
    valid = np.isnan(y) | ((y >= 0) & (y < counts) & (y == np.floor(y)))
    if not valid.all():
        raise ValueError("responses must be declared integer categories or NaN")
    if np.any(np.all(np.isnan(y), axis=0)):
        raise ValueError("Each item needs an observed response")
    if model == "rsm" and np.any(counts != counts[0]):
        raise ValueError("RSM requires equal category counts")
    j = len(counts)
    smax = int(counts.max() - 1)
    if model == "rsm":
        design = np.zeros((j, smax, j + smax - 1))
        for item in range(j):
            design[item, :, item] = 1
            for k in range(smax - 1):
                design[item, k, j + k] = 1
                design[item, -1, j + k] = -1
    else:
        design = np.zeros((j, smax, int(np.sum(counts - 1))))
        offset = 0
        for item, k in enumerate(counts - 1):
            for step in range(k):
                design[item, step, offset + step] = 1
            offset += k
    initial_steps = np.zeros((j, smax))
    for item, k in enumerate(counts - 1):
        initial_steps[item, :k] = np.linspace(-1, 1, k) if k > 1 else 0
    start = dict(starts or {})
    if set(start) - ({"steps", "a"} if model in ("grm", "gpcm") else {"steps"}):
        raise ValueError("Unexpected starts")
    steps = np.asarray(start.get("steps", initial_steps), dtype=float)
    mask = np.arange(smax)[None, :] < counts[:, None] - 1
    if steps.shape != (j, smax) or not np.isfinite(steps[mask]).all():
        raise ValueError("Invalid step starts")
    steps = np.where(mask, steps, 0)
    matrix = design.reshape(j * smax, -1)
    pars = np.linalg.lstsq(matrix, steps.ravel(), rcond=None)[0]
    if not np.allclose(matrix @ pars, steps.ravel(), atol=1e-10, rtol=0):
        raise ValueError("RSM starts must have shared steps plus item locations")
    n = len(pars)
    lower = np.full(n, -np.inf)
    if model in ("grm", "gpcm"):
        a = np.asarray(start.get("a", np.ones(j)), dtype=float)
        if a.shape != (j,) or not np.isfinite(a).all() or np.any(a < 1e-6):
            raise ValueError("Invalid positive slope starts")
        pars = np.r_[pars, a]
        lower = np.r_[lower, np.full(j, 1e-6)]
    constraints = []
    if model == "grm":
        rows = []
        for item, k in enumerate(counts - 1):
            for step in range(k - 1):
                rows.append(
                    np.r_[design[item, step + 1] - design[item, step], np.zeros(j)]
                )
        if rows:
            order = np.asarray(rows)
            if np.any(order @ pars < 1e-6):
                raise ValueError("GRM thresholds must be strictly increasing")
            constraints = [LinearConstraint(order, 1e-6, np.inf)]
    patterns, inverse, freq = np.unique(
        np.where(np.isnan(y), -1, y).astype(int),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    nodes, w = hermgauss(quadrature_points)
    nodes *= np.sqrt(2)
    lw = np.log(w) - 0.5 * np.log(np.pi)
    args = tuple(jnp.asarray(v) for v in (patterns, freq, nodes, lw, design, counts))
    evaluator = _value_grad if jit else jax.value_and_grad(_loss)

    def objective(raw):
        value, grad = evaluator(jnp.asarray(raw), *args, model=model)
        if not np.isfinite(value) or not np.isfinite(grad).all():
            raise FloatingPointError("Nonfinite polytomous objective")
        return float(value), np.asarray(grad)

    trace = [-objective(pars)[0] * len(y)]
    opt = minimize(
        objective,
        pars,
        jac=True,
        method="SLSQP",
        bounds=list(zip(lower, np.full(len(lower), np.inf))),
        constraints=constraints,
        callback=lambda raw: trace.append(-objective(raw)[0] * len(y)),
        options={"maxiter": max_iter, "ftol": tolerance},
    )
    ll, post = (_score if jit else _posterior)(
        jnp.asarray(opt.x), args[0], args[2], args[3], args[4], args[5], model=model
    )
    a, steps = _unpack(jnp.asarray(opt.x), args[4], args[5], model)
    x = post @ args[2]
    var = jnp.sum(post * (args[2][None, :] - x[:, None]) ** 2, axis=1)
    means = dict(a=a, steps=steps, x=x[inverse])
    if model == "rsm":
        means.update(
            b=jnp.asarray(opt.x[:j]), tau=jnp.r_[opt.x[j:n], -np.sum(opt.x[j:n])]
        )
    return PolytomousResult(
        model,
        means,
        {"x": var[inverse]},
        counts.copy(),
        bool(opt.success),
        int(opt.nit),
        float(ll @ args[1]),
        np.asarray(trace),
        str(opt.message),
    )
