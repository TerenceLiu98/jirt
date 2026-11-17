"""Content-only item calibration with a shared logistic 1PL encoder."""

from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from jax.flatten_util import ravel_pytree
from jax.scipy.optimize import minimize as jax_minimize
from functools import partial


def _encode(params, features):
    if "hidden" in params:
        features = jnp.tanh(features @ params["hidden"] + params["hidden_bias"])
    return features @ params["output"] + params["bias"]


def _posterior(difficulty, patterns, nodes, lw):
    z = nodes[:, None] - difficulty[None, :]
    joint = (
        (patterns == 1) @ jax.nn.log_sigmoid(z).T
        + (patterns == 0) @ jax.nn.log_sigmoid(-z).T
        + lw
    )
    return jax.scipy.special.logsumexp(joint, axis=1), jax.nn.softmax(joint, axis=1)


def _unpack(raw, layout):
    params = {}
    offset = 0
    for name, shape in layout:
        size = int(np.prod(shape))
        params[name] = raw[offset : offset + size].reshape(shape)
        offset += size
    return params


def _loss(raw, data, layout):
    features, patterns, freq, nodes, lw, l2 = data
    params = _unpack(raw, layout)
    ll, _ = _posterior(_encode(params, features), patterns, nodes, lw)
    penalty = sum(jnp.sum(v * v) for k, v in params.items() if "bias" not in k)
    return -(ll @ freq) / freq.sum() + 0.5 * l2 * penalty


_value_grad = jax.jit(jax.value_and_grad(_loss), static_argnames=("layout",))


@partial(jax.jit, static_argnames=("layout", "max_iter"))
def _fit_bfgs(raw, data, layout, max_iter, tolerance):
    return jax_minimize(
        lambda x: _loss(x, data, layout),
        raw,
        method="BFGS",
        options=dict(maxiter=max_iter, gtol=tolerance),
    )


@dataclass
class AmortizedResult:
    params: dict
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    item_ids: tuple
    difficulty: jax.Array
    ability: jax.Array
    ability_variance: jax.Array
    converged: bool
    iterations: int
    log_likelihood: float
    objective: float
    trace: np.ndarray
    message: str

    def infer_difficulty(self, features):
        """Infer content-only difficulty using the frozen training transform."""
        x = np.asarray(features, dtype=float)
        if (
            x.ndim != 2
            or x.shape[1] != len(self.feature_mean)
            or not np.isfinite(x).all()
        ):
            raise ValueError("features must be finite (items, training_feature_count)")
        return _encode(
            self.params, jnp.asarray((x - self.feature_mean) / self.feature_scale)
        )

    def predict(self, features, *, theta=None):
        """Predict (subjects,new_items) at supplied abilities or training EAPs."""
        b = self.infer_difficulty(features)
        theta = self.ability if theta is None else jnp.atleast_1d(jnp.asarray(theta))
        return jax.nn.sigmoid(theta[:, None] - b[None, :])


def amortized_irt(
    responses,
    *,
    item_features,
    item_ids,
    feature_item_ids,
    key,
    hidden_size=16,
    l2=1e-4,
    quadrature_points=41,
    max_iter=1000,
    tolerance=1e-10,
    jit=True,
    optimizer="scipy",
):
    """Fit a shared content encoder by regularized normal-marginal 1PL MLE.

    Responses are (N,J) binary with NaN missing. Feature rows are explicitly
    reordered from feature_item_ids to response-column item_ids; both must
    contain the same unique string identifiers. Only training items may enter
    this call. The stored mean/scale never refits on prediction features.

    hidden_size>0 uses one tanh hidden layer; zero uses an affine encoder.
    Unit discrimination and fixed N(0,1) person distribution identify the scale.
    No item-specific offsets or variational posteriors are used. key controls
    initialization. optimizer="scipy" uses full-batch L-BFGS-B; "jax" uses
    device-side BFGS with a gradient-infinity-norm stopping tolerance. The JAX
    path requires jit=True and records only initial/final objectives in trace.
    It reports line-search failure and iteration exhaustion as nonconvergence.
    The loss
    is mean negative person marginal log likelihood plus l2/2 times the sum of
    squared encoder weights (biases excluded). Explicit key is always required.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable JAX x64 before fitting")
    for value, name, minimum in (
        (hidden_size, "hidden_size", 0),
        (quadrature_points, "quadrature_points", 3),
        (max_iter, "max_iter", 1),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for value, name, minimum in ((l2, "l2", 0), (tolerance, "tolerance", 0)):
        if (
            not np.isscalar(value)
            or not np.isfinite(value)
            or value < minimum
            or (name == "tolerance" and value == 0)
        ):
            raise ValueError(f"Invalid {name}")
    if not isinstance(jit, bool):
        raise ValueError("jit must be boolean")
    if optimizer not in ("scipy", "jax") or (optimizer == "jax" and not jit):
        raise ValueError("optimizer must be scipy or jax; jax requires jit=True")
    # Validate typed and legacy JAX keys before doing any optimization work.
    try:
        k1, k2 = jax.random.split(key)
    except (TypeError, ValueError) as exc:
        raise ValueError("key must be a JAX random key") from exc
    y = np.asarray(responses, dtype=float)
    x = np.asarray(item_features, dtype=float)
    if y.ndim != 2 or min(y.shape) < 2:
        raise ValueError("responses must have at least two subjects/items")
    if not (np.isnan(y) | (y == 0) | (y == 1)).all():
        raise ValueError("responses must be binary or NaN")
    if np.any(np.all(np.isnan(y), axis=0)):
        raise ValueError("Every training item needs observed responses")
    ids = tuple(item_ids)
    fids = tuple(feature_item_ids)
    if (
        not all(isinstance(v, str) for v in ids + fids)
        or len(ids) != y.shape[1]
        or len(set(ids)) != len(ids)
        or len(set(fids)) != len(fids)
        or set(ids) != set(fids)
    ):
        raise ValueError("Item IDs must be unique strings with matching sets")
    if (
        x.ndim != 2
        or x.shape[0] != len(fids)
        or x.shape[1] < 1
        or not np.isfinite(x).all()
    ):
        raise ValueError("item_features must be a finite aligned matrix")
    lookup = {v: i for i, v in enumerate(fids)}
    x = x[[lookup[v] for v in ids]]
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    features = jnp.asarray((x - mean) / scale)
    F = x.shape[1]
    if hidden_size:
        params = dict(
            hidden=jax.random.normal(k1, (F, hidden_size), dtype=jnp.float64)
            / np.sqrt(F),
            hidden_bias=jnp.zeros(hidden_size),
            output=0.1
            * jax.random.normal(k2, (hidden_size,), dtype=jnp.float64)
            / np.sqrt(hidden_size),
            bias=jnp.array(0.0),
        )
    else:
        params = dict(
            output=0.01 * jax.random.normal(k1, (F,), dtype=jnp.float64),
            bias=jnp.array(0.0),
        )
    raw, _ = ravel_pytree(params)
    layout = tuple((name, tuple(params[name].shape)) for name in sorted(params))
    patterns, inverse, freq = np.unique(
        np.where(np.isnan(y), -1, y).astype(int),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    patterns = jnp.asarray(patterns)
    freq = jnp.asarray(freq)
    nodes, w = hermgauss(quadrature_points)
    nodes = jnp.asarray(nodes * np.sqrt(2))
    lw = jnp.asarray(np.log(w) - 0.5 * np.log(np.pi))

    data = (features, patterns, freq, nodes, lw, jnp.asarray(l2))
    evaluate = _value_grad if jit else jax.value_and_grad(_loss)

    def objective(raw):
        value, grad = evaluate(jnp.asarray(raw), data, layout)
        if not np.isfinite(value) or not np.isfinite(grad).all():
            raise FloatingPointError("Nonfinite amortized objective")
        return float(value), np.asarray(grad)

    trace = [objective(raw)[0]]
    if optimizer == "jax":
        opt = _fit_bfgs(raw, data, layout, max_iter, jnp.asarray(tolerance))
        if not np.isfinite(opt.fun) or not np.isfinite(opt.x).all():
            raise FloatingPointError("Nonfinite amortized optimization result")
        trace.append(float(opt.fun))
        message = {
            0: "Converged (gradient norm)",
            1: "Iteration limit reached",
            3: "Line search zoom failed",
            4: "Line search failed (saddle point)",
            5: "Line search iteration limit reached",
        }.get(int(opt.status), f"BFGS status {int(opt.status)}")
    else:
        opt = minimize(
            objective,
            np.asarray(raw),
            jac=True,
            method="L-BFGS-B",
            callback=lambda r: trace.append(objective(r)[0]),
            options=dict(maxiter=max_iter, ftol=tolerance, gtol=tolerance),
        )
        message = str(opt.message)
    params = _unpack(jnp.asarray(opt.x), layout)
    b = _encode(params, features)
    ll, post = _posterior(b, patterns, nodes, lw)
    ability = post @ nodes
    variance = jnp.sum(post * (nodes[None, :] - ability[:, None]) ** 2, axis=1)
    return AmortizedResult(
        params,
        mean,
        scale,
        ids,
        b,
        ability[inverse],
        variance[inverse],
        bool(opt.success),
        int(opt.nit),
        float(ll @ freq),
        float(opt.fun),
        np.asarray(trace),
        message,
    )
