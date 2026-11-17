"""Nominal response and nested logistic distractor models."""

import jax
import jax.numpy as jnp


def _log_probabilities(
    theta,
    slopes,
    intercepts,
    category_counts,
    answer_keys=None,
    a=None,
    b=None,
    c=None,
    u=None,
):
    theta = jnp.atleast_1d(jnp.asarray(theta))
    slopes = jnp.asarray(slopes)
    intercepts = jnp.asarray(intercepts)
    counts = jnp.asarray(category_counts)
    categories = jnp.arange(slopes.shape[1])[None, :]
    valid = categories < counts[:, None]
    # Padding is excluded before arithmetic, allowing NaN padded inputs.
    logits = (
        theta[:, None, None] * jnp.where(valid, slopes, 0)[None, :, :]
        + jnp.where(valid, intercepts, 0)[None, :, :]
    )
    if answer_keys is None:
        return jax.nn.log_softmax(
            jnp.where(valid[None, :, :], logits, -jnp.inf), axis=-1
        )
    keys = jnp.asarray(answer_keys)
    correct = categories == keys[:, None]
    distractor = valid & ~correct
    conditional = jax.nn.log_softmax(
        jnp.where(distractor[None, :, :], logits, -jnp.inf), axis=-1
    )
    z = jnp.asarray(a)[None, :] * (theta[:, None] - jnp.asarray(b)[None, :])
    log_success = jnp.logaddexp(
        jnp.log(jnp.asarray(c))[None, :] + jax.nn.log_sigmoid(-z),
        jnp.log(jnp.asarray(u))[None, :] + jax.nn.log_sigmoid(z),
    )
    log_failure = jnp.logaddexp(
        jnp.log1p(-jnp.asarray(c))[None, :] + jax.nn.log_sigmoid(-z),
        jnp.log1p(-jnp.asarray(u))[None, :] + jax.nn.log_sigmoid(z),
    )
    result = jnp.where(
        correct[None, :, :],
        log_success[:, :, None],
        log_failure[:, :, None] + conditional,
    )
    return jnp.where(valid[None, :, :], result, -jnp.inf)


def nominal_probabilities(theta, *, slopes, intercepts, category_counts):
    """Return (N,J,max_categories) probabilities; absent categories have zero mass.

    Slopes/intercepts have shape (J,max_categories). Category zero is the
    reference in fitted models (both coefficients zero). This pure JAX function
    also accepts equivalent common shifts. Counts must be >=2 and inputs valid.
    """
    return jnp.exp(_log_probabilities(theta, slopes, intercepts, category_counts))


def nested_probabilities(
    theta, *, slopes, intercepts, category_counts, answer_keys, a, b, c=None, u=None
):
    """Correct mass c+(u-c)*sigmoid(a*(theta-b)); remaining mass splits over distractors.

    Keys are zero-based category indices. Correct-option nominal coefficients
    are ignored. Each item requires at least one distractor. c/u default to 0/1;
    supply c>0 for 3PL, u<1 for 3PLu, or both for full 4PL.
    """
    a = jnp.asarray(a)
    c = jnp.zeros_like(a) if c is None else jnp.broadcast_to(jnp.asarray(c), a.shape)
    u = jnp.ones_like(a) if u is None else jnp.broadcast_to(jnp.asarray(u), a.shape)
    return jnp.exp(
        _log_probabilities(
            theta, slopes, intercepts, category_counts, answer_keys, a, b, c, u
        )
    )


from dataclasses import dataclass
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize, LinearConstraint


@dataclass
class NominalResult:
    model: str
    means: dict
    variances: dict
    category_counts: np.ndarray
    answer_keys: np.ndarray | None
    converged: bool
    iterations: int
    log_likelihood: float
    trace: np.ndarray
    message: str

    def predict(self, theta=None):
        theta = self.means["x"] if theta is None else theta
        args = {k: self.means[k] for k in ("slopes", "intercepts")}
        args["category_counts"] = self.category_counts
        if self.answer_keys is None:
            return nominal_probabilities(theta, **args)
        return nested_probabilities(
            theta,
            **args,
            answer_keys=self.answer_keys,
            **{k: self.means[k] for k in ("a", "b", "c", "u")},
        )

    @property
    def probabilities(self):
        return self.predict()


def _unpack(raw, design, model):
    n = design.shape[-1]
    j = design.shape[0]
    slopes = jnp.einsum("jck,k->jc", design, raw[:n])
    intercepts = jnp.einsum("jck,k->jc", design, raw[n : 2 * n])
    if model == "nominal":
        return slopes, intercepts, None, None, None, None
    offset = 2 * n
    a = raw[offset : offset + j]
    b = raw[offset + j : offset + 2 * j]
    offset += 2 * j
    c = raw[offset : offset + j] if model in ("3pl", "4pl") else jnp.zeros(j)
    if model in ("3pl", "4pl"):
        offset += j
    u = raw[offset : offset + j] if model in ("3plu", "4pl") else jnp.ones(j)
    return slopes, intercepts, a, b, c, u


def _posterior(raw, y, nodes, lw, design, counts, keys, model):
    slopes, intercepts, a, b, c, u = _unpack(raw, design, model)
    lp = _log_probabilities(
        nodes,
        slopes,
        intercepts,
        counts,
        None if model == "nominal" else keys,
        a,
        b,
        c,
        u,
    )
    selected = lp[:, jnp.arange(y.shape[1])[None, :], jnp.maximum(y, 0)]
    joint = jnp.sum(jnp.where(y[None, :, :] >= 0, selected, 0.0), axis=2).T + lw
    return jax.scipy.special.logsumexp(joint, axis=1), jax.nn.softmax(joint, axis=1)


def _loss(raw, y, frequency, nodes, lw, design, counts, keys, model):
    ll, _ = _posterior(raw, y, nodes, lw, design, counts, keys, model)
    return -jnp.sum(ll * frequency) / jnp.sum(frequency)


_value_grad = jax.jit(jax.value_and_grad(_loss), static_argnames=("model",))
_score = jax.jit(_posterior, static_argnames=("model",))


def nominal_irt(
    responses,
    *,
    category_counts,
    answer_keys=None,
    model="nominal",
    starts=None,
    quadrature_points=41,
    max_iter=1000,
    tolerance=1e-10,
    jit=True,
):
    """Fit nominal or nested 2PL/3PL/3PLu/4PL with N(0,1) marginal likelihood.

    Category zero is the nominal reference. The first non-key category is the
    nested distractor reference; correct-option nominal coefficients are zero.
    For nominal fits the item-0/category-1 slope is positive to fix latent sign;
    nested correct-answer slopes are all positive. No item priors are applied.
    Responses use declared zero-based categories and NaN for missing values.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable JAX x64 before fitting")
    if model not in ("nominal", "2pl", "3pl", "3plu", "4pl"):
        raise ValueError("Unknown nominal/nested model")
    for v, name, minimum in (
        (quadrature_points, "quadrature_points", 3),
        (max_iter, "max_iter", 1),
    ):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not isinstance(jit, bool):
        raise ValueError("jit must be boolean")
    if not np.isscalar(tolerance) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Invalid tolerance")
    y = np.asarray(responses, dtype=float)
    counts = np.asarray(category_counts)
    if y.ndim != 2 or min(y.shape) < 2:
        raise ValueError("responses must be a matrix with at least two subjects/items")
    j = y.shape[1]
    if counts.shape != (j,) or counts.dtype.kind not in "iu" or np.any(counts < 2):
        raise ValueError("Invalid category_counts")
    if not (np.isnan(y) | ((y >= 0) & (y < counts) & (y == np.floor(y)))).all():
        raise ValueError("Invalid response category")
    if np.any(np.all(np.isnan(y), axis=0)):
        raise ValueError("Each item requires an observed response")
    keys = np.full(j, -1, dtype=int) if answer_keys is None else np.asarray(answer_keys)
    if model == "nominal":
        if answer_keys is not None:
            raise ValueError("answer_keys applies only to nested models")
    elif (
        keys.shape != (j,)
        or keys.dtype.kind not in "iu"
        or np.any((keys < 0) | (keys >= counts))
    ):
        raise ValueError("Nested models require valid answer_keys")
    free = []
    for item, k in enumerate(counts):
        options = [c for c in range(k) if c != keys[item]]
        free.extend((item, c) for c in options[1:])
    design = np.zeros((j, int(counts.max()), len(free)))
    for ix, (item, c) in enumerate(free):
        design[item, c, ix] = 1
    n = len(free)
    s = dict(starts or {})
    allowed = {"slopes", "intercepts"}
    if model != "nominal":
        allowed.update(("a", "b"))
    if model in ("3pl", "4pl"):
        allowed.add("c")
    if model in ("3plu", "4pl"):
        allowed.add("u")
    if set(s) - allowed:
        raise ValueError("Unexpected starts")
    raw = []
    for name, default in (("slopes", 0.5), ("intercepts", 0.0)):
        matrix = np.asarray(s.get(name, np.sum(design, axis=-1) * default), dtype=float)
        if matrix.shape != design.shape[:2] or not np.isfinite(matrix).all():
            raise ValueError("Invalid category coefficient starts")
        values = np.array([matrix[item, c] for item, c in free])
        if not np.allclose(
            matrix, np.einsum("jck,k->jc", design, values), atol=1e-12, rtol=0
        ):
            raise ValueError("Starts violate category reference or padding constraints")
        raw.extend(values)
    lower = [-np.inf] * (2 * n)
    upper = [np.inf] * (2 * n)
    if model == "nominal":
        lower[0] = 1e-6
    else:
        for name, default, lo, hi in (
            [("a", 1, 1e-6, np.inf), ("b", 0, -np.inf, np.inf)]
            + ([("c", 0.1, 1e-6, 1 - 1e-6)] if "c" in allowed else [])
            + ([("u", 0.9, 1e-6, 1 - 1e-6)] if "u" in allowed else [])
        ):
            value = np.asarray(s.get(name, np.full(j, default)), dtype=float)
            if value.shape != (j,) or not np.isfinite(value).all():
                raise ValueError("Invalid binary curve starts")
            raw.extend(value)
            lower.extend([lo] * j)
            upper.extend([hi] * j)
    raw = np.asarray(raw)
    if np.any(raw < lower) or np.any(raw > upper):
        raise ValueError("Starts violate parameter bounds")
    constraints = []
    if model == "4pl":
        matrix = np.zeros((j, len(raw)))
        matrix[:, 2 * n + 2 * j : 2 * n + 3 * j] = -np.eye(j)
        matrix[:, 2 * n + 3 * j :] = np.eye(j)
        if np.any(matrix @ raw < 1e-6):
            raise ValueError("4PL requires u-c >= 1e-6")
        constraints = [LinearConstraint(matrix, 1e-6, np.inf)]
    patterns, inverse, frequency = np.unique(
        np.where(np.isnan(y), -1, y).astype(int),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    nodes, w = hermgauss(quadrature_points)
    nodes *= np.sqrt(2)
    args = tuple(
        jnp.asarray(v)
        for v in (
            patterns,
            frequency,
            nodes,
            np.log(w) - 0.5 * np.log(np.pi),
            design,
            counts,
            keys,
        )
    )
    evaluator = _value_grad if jit else jax.value_and_grad(_loss)

    def objective(raw):
        value, grad = evaluator(jnp.asarray(raw), *args, model=model)
        if not np.isfinite(value) or not np.isfinite(grad).all():
            raise FloatingPointError("Nonfinite nominal objective")
        return float(value), np.asarray(grad)

    trace = [-objective(raw)[0] * len(y)]
    opt = minimize(
        objective,
        raw,
        jac=True,
        method="SLSQP",
        bounds=list(zip(lower, upper)),
        constraints=constraints,
        callback=lambda p: trace.append(-objective(p)[0] * len(y)),
        options={"maxiter": max_iter, "ftol": tolerance},
    )
    ll, post = (_score if jit else _posterior)(
        jnp.asarray(opt.x),
        args[0],
        args[2],
        args[3],
        args[4],
        args[5],
        args[6],
        model=model,
    )
    slopes, intercepts, a, b, c, u = _unpack(jnp.asarray(opt.x), args[4], model)
    x = post @ args[2]
    var = jnp.sum(post * (args[2][None, :] - x[:, None]) ** 2, axis=1)
    means = dict(slopes=slopes, intercepts=intercepts, x=x[inverse])
    if model != "nominal":
        means.update(a=a, b=b, c=c, u=u)
    return NominalResult(
        model,
        means,
        {"x": var[inverse]},
        counts.copy(),
        None if model == "nominal" else keys.copy(),
        bool(opt.success),
        int(opt.nit),
        float(ll @ args[1]),
        np.asarray(trace),
        str(opt.message),
    )
