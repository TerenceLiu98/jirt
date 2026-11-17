"""Compensatory logistic MIRT with standard-normal marginal calibration."""

from dataclasses import dataclass
from itertools import product

from ._optimize import device_minimize

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize

from .types import MultidimensionalStarts


@dataclass
class MultidimensionalResult:
    """Item estimates and conditional EAP scores; covariance excludes item uncertainty."""

    structure: str
    means: dict[str, jax.Array]
    covariances: dict[str, jax.Array]
    converged: bool
    iterations: int
    log_likelihood: float
    trace: np.ndarray
    message: str
    loading_mask: np.ndarray

    @property
    def variances(self):
        return {"x": jnp.diagonal(self.covariances["x"], axis1=-2, axis2=-1)}

    @property
    def probabilities(self):
        return self.predict()

    def predict(self, theta=None):
        return multidimensional_probabilities(
            self.means["x"] if theta is None else theta,
            a=self.means["a"],
            d=self.means["d"],
        )


def multidimensional_probabilities(theta, *, a, d):
    """Subject-by-item sigmoid(theta @ a.T + d); theta has shape (N,D)."""
    return jax.nn.sigmoid(jnp.asarray(theta) @ jnp.asarray(a).T + jnp.asarray(d))


def _quadrature(dimensions, points, max_nodes):
    if points**dimensions > max_nodes:
        raise ValueError(
            "Tensor quadrature exceeds max_nodes; reduce dimensions or quadrature_points"
        )
    x, w = hermgauss(points)
    index = np.array(list(product(range(points), repeat=dimensions)))
    return x[index] * np.sqrt(2), np.sum(np.log(w[index]) - 0.5 * np.log(np.pi), axis=1)


def _posterior(raw, yes, no, nodes, log_weights, design):
    items = design.shape[0]
    a = jnp.einsum("jdk,k->jd", design, raw[items:])
    z = nodes @ a.T + raw[:items]
    joint = yes @ jax.nn.log_sigmoid(z).T + no @ jax.nn.log_sigmoid(-z).T + log_weights
    return jax.scipy.special.logsumexp(joint, axis=1), jax.nn.softmax(joint, axis=1)


def _loss(raw, yes, no, counts, nodes, log_weights, design):
    ll, _ = _posterior(raw, yes, no, nodes, log_weights, design)
    return -jnp.sum(ll * counts) / jnp.sum(counts)


_value_grad = jax.jit(jax.value_and_grad(_loss))
_score = jax.jit(_posterior)


def multidimensional_irt(
    responses,
    *,
    dimensions=2,
    loading_mask=None,
    structure="multidimensional",
    groups=None,
    anchors=None,
    starts: MultidimensionalStarts | None = None,
    quadrature_points=15,
    max_nodes=100000,
    max_iter=1000,
    tolerance=1e-10,
    jit=True,
    optimizer="scipy",
):
    """Fit compensatory 2PL, bifactor or random-intercept testlet IRT.

    Responses are 0/1 with NaN missing. All latent factors are independent N(0,1);
    no item priors are used. ``a`` is item-by-factor and ``d`` is an intercept,
    not univariate difficulty. Tensor Gauss-Hermite MML uses JAX derivatives.

    Without a loading_mask, the first D items form a lower-triangular anchor
    block with positive diagonal (exploratory orientation). A supplied boolean
    mask fixes prohibited loadings to exactly zero; one positive anchor loading
    per factor fixes signs. Masks must independently remove any remaining
    rotations; arbitrary masks are not guaranteed identifiable.

    For bifactor/testlet, groups is a contiguous zero-based item-to-group vector;
    -1 marks a general-only item. Dimensions are inferred as 1 + group count.
    Bifactor allows free specific loadings; testlet ties them within each group
    to a nonnegative scale tau, equivalent to a unit-loading random intercept
    with variance tau**2. Group factors are orthogonal to the general factor.
    Each specific group needs at least three items. These count checks alone
    do not prove identification of a particular design.

    starts accepts a (J,D) and d (J,). anchors is a D-vector of item indices;
    defaults are diagonal for exploratory models, otherwise first allowed item.
    tolerance is SLSQP ftol on mean negative marginal log likelihood.
    optimizer="scipy" retains SLSQP. optimizer="jax" requires jit=True and
    uses device-side spectral projected gradient with the same constraints.
    For JAX, tolerance bounds the unit-step projected-gradient infinity norm;
    trace contains accepted iterates and may be nonmonotone. See PERFORMANCE.md.
    """
    if optimizer not in ("scipy", "jax") or (optimizer == "jax" and not jit):
        raise ValueError("optimizer must be scipy or jax; jax requires jit=True")
    if not jax.config.x64_enabled:
        raise ValueError("Enable JAX x64 before fitting")
    for value, name, minimum in (
        (dimensions, "dimensions", 1),
        (quadrature_points, "quadrature_points", 3),
        (max_nodes, "max_nodes", 1),
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
    if structure not in ("multidimensional", "bifactor", "testlet"):
        raise ValueError("Unknown structure")
    y = np.asarray(responses, dtype=float)
    if y.ndim != 2 or min(y.shape) < 2 or not (np.isin(y, [0, 1]) | np.isnan(y)).all():
        raise ValueError(
            "responses must be a 0/1/NaN matrix with at least two subjects/items"
        )
    if np.any(np.sum(np.isfinite(y), axis=0) == 0):
        raise ValueError("Each item requires an observed response")
    items = y.shape[1]
    exploratory = structure == "multidimensional" and loading_mask is None
    if structure == "multidimensional":
        if groups is not None:
            raise ValueError("groups applies only to bifactor/testlet")
        if dimensions > items:
            raise ValueError("dimensions cannot exceed item count")
        if loading_mask is None:
            mask = np.ones((items, dimensions), dtype=bool)
            mask[:dimensions] = np.tril(np.ones((dimensions, dimensions), dtype=bool))
        else:
            mask = np.asarray(loading_mask)
            if mask.dtype != bool or mask.shape != (items, dimensions):
                raise ValueError(
                    "loading_mask must be boolean with shape (items, dimensions)"
                )
    else:
        if loading_mask is not None:
            raise ValueError("Structured models derive their loading mask from groups")
        group = np.asarray(groups)
        if (
            group.shape != (items,)
            or group.dtype.kind not in "iu"
            or np.any(group < -1)
        ):
            raise ValueError(
                "groups must be an integer item vector with -1 or zero-based group IDs"
            )
        labels = np.unique(group[group >= 0])
        if len(labels) == 0 or not np.array_equal(labels, np.arange(len(labels))):
            raise ValueError("groups must contain contiguous IDs starting at zero")
        if any(np.sum(group == g) < 3 for g in labels):
            raise ValueError("Each specific group requires at least three items")
        dimensions = len(labels) + 1
        mask = np.zeros((items, dimensions), dtype=bool)
        mask[:, 0] = True
        for g in labels:
            mask[group == g, g + 1] = True
    if not mask.any(axis=0).all() or not mask.any(axis=1).all():
        raise ValueError("Every factor and item requires an allowed loading")
    if anchors is None:
        anchor = np.arange(dimensions) if exploratory else mask.argmax(axis=0)
    else:
        anchor = np.asarray(anchors)
        if (
            anchor.shape != (dimensions,)
            or anchor.dtype.kind not in "iu"
            or np.any((anchor < 0) | (anchor >= items))
        ):
            raise ValueError("anchors must contain one valid item index per factor")
    if not mask[anchor, np.arange(dimensions)].all():
        raise ValueError("Anchor loadings must be allowed by the loading mask")
    # Each design column represents a free loading or an equality-constrained group scale.
    slots = []
    for j, k in zip(*np.where(mask), strict=True):
        slot = ("group", k) if structure == "testlet" and k > 0 else (j, k)
        if slot not in slots:
            slots.append(slot)
    design = np.zeros((items, dimensions, len(slots)))
    for j, k in zip(*np.where(mask), strict=True):
        slot = ("group", k) if structure == "testlet" and k > 0 else (j, k)
        design[j, k, slots.index(slot)] = 1
    lower = np.full(items + len(slots), -np.inf)
    for j, k in zip(anchor, range(dimensions), strict=True):
        ix = np.flatnonzero(design[j, k])[0]
        lower[items + ix] = 0 if structure == "testlet" and k > 0 else 1e-6
    s = dict(starts or {})
    if set(s) - {"a", "d"}:
        raise ValueError("starts accepts only a and d")
    a = np.asarray(
        s.get("a", mask * np.where(np.arange(dimensions) == 0, 1.0, 0.5)), dtype=float
    )
    d = np.asarray(s.get("d", np.zeros(items)), dtype=float)
    if (
        a.shape != mask.shape
        or d.shape != (items,)
        or not np.isfinite(a).all()
        or not np.isfinite(d).all()
    ):
        raise ValueError("starts require finite a (items, dimensions) and d (items,)")
    if np.any(a[~mask] != 0):
        raise ValueError("Starts must respect prohibited cross-loadings")
    loading = np.array(
        [a[design[:, :, i].astype(bool)].mean() for i in range(len(slots))]
    )
    if not np.allclose(a, np.einsum("jdk,k->jd", design, loading), atol=1e-12, rtol=0):
        raise ValueError("Testlet specific loadings must be equal within each group")
    initial = np.r_[d, loading]
    if np.any(initial < lower):
        raise ValueError(
            "Starts violate positive anchor or nonnegative testlet constraints"
        )
    nodes, log_weights = _quadrature(dimensions, quadrature_points, max_nodes)
    # Aggregate response patterns, including missingness, before allocating posteriors.
    patterns, inverse, counts = np.unique(
        np.where(np.isnan(y), -1, y), axis=0, return_inverse=True, return_counts=True
    )
    args = tuple(
        jnp.asarray(v)
        for v in (patterns == 1, patterns == 0, counts, nodes, log_weights, design)
    )
    evaluator = _value_grad if jit else jax.value_and_grad(_loss)

    def objective(raw):
        value, grad = evaluator(jnp.asarray(raw), *args)
        if not np.isfinite(value) or not np.isfinite(grad).all():
            raise FloatingPointError("Nonfinite MIRT objective or gradient")
        return float(value), np.asarray(grad)

    history = [-objective(initial)[0] * len(y)]
    if optimizer == "jax":
        optimum = device_minimize(
            _loss,
            initial,
            args,
            lower=lower,
            upper=np.full(len(lower), np.inf),
            constraints=(),
            config=(),
            max_iter=max_iter,
            tolerance=tolerance,
        )
        history = -optimum.trace * len(y)
    else:
        optimum = minimize(
            objective,
            initial,
            jac=True,
            method="SLSQP",
            bounds=list(zip(lower, np.full(len(lower), np.inf))),
            callback=lambda raw: history.append(-objective(raw)[0] * len(y)),
            options={"maxiter": max_iter, "ftol": tolerance},
        )
    raw = jnp.asarray(optimum.x)
    ll, posterior = (_score if jit else _posterior)(
        raw, args[0], args[1], args[3], args[4], args[5]
    )
    x = posterior @ args[3]
    centered = args[3][None, :, :] - x[:, None, :]
    cov = jnp.einsum("pq,pqd,pqe->pde", posterior, centered, centered)
    a = jnp.einsum("jdk,k->jd", args[5], raw[items:])
    means = dict(a=a, d=raw[:items], x=x[inverse])
    if structure == "testlet":
        means["testlet_variance"] = jnp.array(
            [a[np.flatnonzero(group == g)[0], g + 1] ** 2 for g in labels]
        )
    return MultidimensionalResult(
        structure,
        means,
        {"x": cov[inverse]},
        bool(optimum.success),
        int(optimum.nit),
        float(ll @ args[2]),
        np.asarray(history),
        str(optimum.message),
        mask.copy(),
    )
