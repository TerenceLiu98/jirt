"""Logistic IRT calibration by Gaussian-quadrature marginal likelihood.

Item parameters are estimated by maximum marginal likelihood, with theta ~ N(0,1).
Person scores are EAP estimates conditional on the calibrated item parameters.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import Bounds, LinearConstraint, minimize

from .types import LogisticStarts


@dataclass
class LogisticResult:
    """Calibration and conditional person scoring (not item standard errors)."""

    model: str
    means: dict[str, jax.Array]
    variances: dict[str, jax.Array]
    converged: bool
    iterations: int
    log_likelihood: float
    trace: np.ndarray
    probabilities: jax.Array
    message: str

    def predict(self, theta=None):
        """Conditional response probabilities at supplied or EAP person scores."""
        return logistic_probabilities(
            self.means["x"] if theta is None else theta,
            a=self.means["a"],
            b=self.means["b"],
            c=self.means["c"],
            u=self.means["u"],
        )


def logistic_probabilities(theta, *, a, b, c=0.0, u=1.0):
    """Return subject-by-item probabilities c + (u-c)*sigmoid(a*(theta-b)).

    This pure JAX function supports jit and automatic differentiation. Inputs
    must satisfy a > 0 and 0 <= c < u <= 1; fitting validates these constraints.
    Scalars or one-dimensional arrays are accepted; item arrays broadcast.
    """
    theta = jnp.atleast_1d(jnp.asarray(theta))
    z = (theta[:, None] - jnp.atleast_1d(jnp.asarray(b))) * jnp.atleast_1d(
        jnp.asarray(a)
    )
    return jnp.asarray(c) + (jnp.asarray(u) - jnp.asarray(c)) * jax.nn.sigmoid(z)


def _unpack(params, model, items):
    b = params[:items]
    offset = items
    if model == "rasch":
        a = jnp.ones(items)
    elif model == "1pl":
        a = jnp.full(items, params[offset])
        offset += 1
    else:
        a = params[offset : offset + items]
        offset += items
    c = params[offset : offset + items] if model in ("3pl", "4pl") else jnp.zeros(items)
    if model in ("3pl", "4pl"):
        offset += items
    u = params[offset : offset + items] if model == "4pl" else jnp.ones(items)
    return a, b, c, u


def _log_response(params, nodes, model, items):
    a, b, c, u = _unpack(params, model, items)
    z = a * (nodes[:, None] - b)
    yes, no = jax.nn.log_sigmoid(z), jax.nn.log_sigmoid(-z)
    if model in ("3pl", "4pl"):
        yes = jnp.logaddexp(jnp.log(c) + no, jnp.log(u) + yes)
        no = (
            jnp.logaddexp(jnp.log1p(-c) + no, jnp.log1p(-u) + jax.nn.log_sigmoid(z))
            if model == "4pl"
            else jnp.log1p(-c) + no
        )
    return yes, no


def _posterior(params, successes, failures, nodes, log_weights, model):
    yes, no = _log_response(params, nodes, model, successes.shape[1])
    joint = successes @ yes.T + failures @ no.T + log_weights
    normalizer = jax.scipy.special.logsumexp(joint, axis=1)
    return normalizer, jax.nn.softmax(joint, axis=1)


def _loss(params, successes, failures, nodes, log_weights, model):
    log_likelihood, _ = _posterior(
        params, successes, failures, nodes, log_weights, model
    )
    return -jnp.mean(log_likelihood)


_value_grad = jax.jit(jax.value_and_grad(_loss), static_argnames=("model",))
_score = jax.jit(_posterior, static_argnames=("model",))


def logistic_irt(
    responses,
    *,
    model="2pl",
    starts: LogisticStarts | None = None,
    quadrature_points=41,
    max_iter=1000,
    tolerance=1e-8,
    jit=True,
):
    """Fit Rasch, shared-slope 1PL, 2PL, 3PL or full 4PL to 0/1 responses.

    NaN denotes missing responses. The ability distribution is fixed N(0,1),
    slopes are positive, and no item priors are used. ``rasch`` fixes slopes
    to one; ``1pl`` estimates one common slope. Starts, when supplied, contain
    per-item b and a (scalar a for 1PL), plus c for 3PL/4PL and u for 4PL.

    SLSQP maximizes quadrature marginal likelihood using JAX derivatives.
    Numerical constraints are a >= 1e-6; fitted c >= 1e-6; fitted u <= 1-1e-6;
    and u-c >= 1e-6. Fixed asymptotes remain exactly 0/1.
    ``tolerance`` controls SLSQP's stopping accuracy on mean negative log
    likelihood, not the correlation criterion used by the probit estimators.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable JAX x64 before fitting")
    if model not in ("rasch", "1pl", "2pl", "3pl", "4pl"):
        raise ValueError("Unknown logistic model")
    for value, name, minimum in (
        (max_iter, "max_iter", 1),
        (quadrature_points, "quadrature_points", 3),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not np.isscalar(tolerance) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if not isinstance(jit, bool):
        raise ValueError("jit must be boolean")
    y = np.asarray(responses, dtype=float)
    if y.ndim != 2 or min(y.shape) < 2 or not (np.isin(y, [0, 1]) | np.isnan(y)).all():
        raise ValueError(
            "responses must be a matrix with at least two subjects/items, containing 0/1/NaN"
        )
    if np.any(np.sum(np.isfinite(y), axis=0) == 0):
        raise ValueError("Each item requires at least one observed response")
    items = y.shape[1]
    eps = 1e-6
    s = dict(starts or {})
    allowed = {"b"}
    if model != "rasch":
        allowed.add("a")
    if model in ("3pl", "4pl"):
        allowed.add("c")
    if model == "4pl":
        allowed.add("u")
    if set(s) - allowed:
        raise ValueError(f"Unexpected starts for {model}: {set(s) - allowed}")

    def start(name, default, size):
        value = np.asarray(s.get(name, default), dtype=float)
        if value.ndim == 0:
            value = np.full(size, value.item())
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid start for {name}")
        return value

    initial = [start("b", 0.0, items)]
    lower = [-np.inf] * items
    upper = [np.inf] * items
    if model != "rasch":
        count = 1 if model == "1pl" else items
        initial.append(start("a", 1.0, count))
        lower.extend([eps] * count)
        upper.extend([np.inf] * count)
    if model in ("3pl", "4pl"):
        initial.append(start("c", 0.1, items))
        lower.extend([eps] * items)
        upper.extend([1 - eps] * items)
    if model == "4pl":
        initial.append(start("u", 0.9, items))
        lower.extend([eps] * items)
        upper.extend([1 - eps] * items)
    initial = np.concatenate(initial)
    if np.any(initial < lower) or np.any(initial > upper):
        raise ValueError("Starts violate numerical parameter bounds")
    constraints = []
    if model == "4pl":
        matrix = np.zeros((items, len(initial)))
        matrix[:, 2 * items : 3 * items] = -np.eye(items)
        matrix[:, 3 * items :] = np.eye(items)
        if np.any(matrix @ initial < eps):
            raise ValueError("4PL requires u-c >= 1e-6")
        constraints = [LinearConstraint(matrix, eps, np.inf)]
    nodes, weights = hermgauss(quadrature_points)
    arguments = (
        jnp.asarray(y == 1, dtype=float),
        jnp.asarray(y == 0, dtype=float),
        jnp.asarray(nodes * np.sqrt(2)),
        jnp.asarray(np.log(weights) - 0.5 * np.log(np.pi)),
    )
    evaluator = _value_grad if jit else jax.value_and_grad(_loss)

    def objective(params):
        value, gradient = evaluator(jnp.asarray(params), *arguments, model=model)
        value, gradient = float(value), np.asarray(gradient)
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite logistic likelihood or gradient")
        return value, gradient

    history = [-objective(initial)[0] * y.shape[0]]

    def callback(params):
        history.append(-objective(params)[0] * y.shape[0])

    optimum = minimize(
        objective,
        initial,
        method="SLSQP",
        jac=True,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        callback=callback,
        options={"maxiter": max_iter, "ftol": tolerance},
    )
    params = jnp.asarray(optimum.x)
    log_likelihood, posterior = (_score if jit else _posterior)(
        params, *arguments, model=model
    )
    nodes = arguments[2]
    x = posterior @ nodes
    vx = jnp.sum(posterior * (nodes[None, :] - x[:, None]) ** 2, axis=1)
    a, b, c, u = _unpack(params, model, items)
    return LogisticResult(
        model=model,
        means=dict(a=a, b=b, c=c, u=u, x=x),
        variances=dict(x=vx),
        converged=bool(optimum.success),
        iterations=int(optimum.nit),
        log_likelihood=float(jnp.sum(log_likelihood)),
        trace=np.asarray(history),
        probabilities=logistic_probabilities(x, a=a, b=b, c=c, u=u),
        message=str(optimum.message),
    )
