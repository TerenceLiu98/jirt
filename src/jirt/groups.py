"""Observed-group and latent-class logistic/graded response mechanisms."""

import jax
import jax.numpy as jnp

from .polytomous import _log_probabilities


def _component_logs(theta, a, steps, counts, model):
    # theta is class-by-person; parameters are class-by-item(-by-threshold).
    if model == "rasch":
        a = jnp.ones_like(a)
    return jax.vmap(
        lambda t, slopes, b: _log_probabilities(t, slopes, b, counts, "grm")
    )(theta, a, steps)


def group_probabilities(theta, *, groups, a, steps, category_counts, model="2pl"):
    """Return (N,J,max(K)) probabilities for observed integer group labels.

    a is (G,J), steps is (G,J,max(K)-1), groups is (N,), theta is (N,).
    Models are 2pl, rasch (unit slopes), or grm. Binary models require K=2.
    This pure JAX function assumes valid parameter shapes and restrictions.
    """
    if model not in ("2pl", "rasch", "grm"):
        raise ValueError("Unknown response model")
    theta = jnp.atleast_1d(jnp.asarray(theta))
    a = jnp.asarray(a)
    lp = _component_logs(
        jnp.broadcast_to(theta, (a.shape[0], len(theta))),
        a,
        jnp.asarray(steps),
        jnp.asarray(category_counts),
        model,
    )
    return jnp.exp(lp[jnp.asarray(groups), jnp.arange(len(theta))])


def mixture_probabilities(
    theta, *, a, steps, category_counts, weights, model="2pl", zero_inflated=False
):
    """Prior-weighted item probabilities for common abilities theta (N,).

    a/steps hold C regular classes. weights has length C (or C+1 when
    zero_inflated). The final structural component always emits category zero
    on every item. It has no ability or item parameters. Zero weights are
    allowed. Mixture item marginals do not factorize into pattern likelihoods.
    """
    if model not in ("2pl", "rasch", "grm"):
        raise ValueError("Unknown response model")
    theta = jnp.atleast_1d(jnp.asarray(theta))
    a = jnp.asarray(a)
    p = jnp.exp(
        _component_logs(
            jnp.broadcast_to(theta, (a.shape[0], len(theta))),
            a,
            jnp.asarray(steps),
            jnp.asarray(category_counts),
            model,
        )
    )
    if zero_inflated:
        structural = jnp.zeros_like(p[:1]).at[:, :, :, 0].set(1.0)
        p = jnp.concatenate([p, structural], axis=0)
    return jnp.einsum("c,cnjk->njk", jnp.asarray(weights), p)


def _component_posterior(patterns, nodes, lw, a, steps, mu, sigma, counts, model):
    theta = mu[:, None] + sigma[:, None] * nodes[None, :]
    lp = _component_logs(theta, a, steps, counts, model)
    selected = lp[
        :, :, jnp.arange(patterns.shape[1])[None, :], jnp.maximum(patterns, 0)
    ]
    joint = jnp.sum(jnp.where(patterns[None, None, :, :] >= 0, selected, 0.0), axis=-1)
    joint = jnp.transpose(joint, (2, 0, 1)) + lw[None, None, :]
    return (
        jax.scipy.special.logsumexp(joint, axis=-1),
        jax.nn.softmax(joint, axis=-1),
        theta,
    )


def _mix_posterior(component_ll, patterns, weights, zero_inflated):
    if zero_inflated:
        # Missing responses place no restriction on the structural component.
        # Any observed nonzero response rules it out exactly.
        structural = jnp.where(jnp.all(patterns <= 0, axis=1), 0.0, -jnp.inf)
        component_ll = jnp.column_stack([component_ll, structural])
    joint = component_ll + jnp.log(weights)[None, :]
    return jax.scipy.special.logsumexp(joint, axis=1), jax.nn.softmax(joint, axis=1)


from dataclasses import dataclass
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize, LinearConstraint


@dataclass
class GroupIRTResult:
    kind: str
    model: str
    means: dict
    variances: dict
    category_counts: np.ndarray
    weights: np.ndarray | None
    memberships: jax.Array | None
    groups: np.ndarray | None
    zero_inflated: bool
    converged: bool
    iterations: int
    log_likelihood: float
    trace: np.ndarray
    message: str
    start_log_likelihoods: np.ndarray

    def predict(self, theta, *, groups=None):
        pars = {k: self.means[k] for k in ("a", "steps")}
        if self.kind == "group":
            if groups is None:
                raise ValueError("Supply group labels for prediction")
            return group_probabilities(
                theta,
                groups=groups,
                category_counts=self.category_counts,
                model=self.model,
                **pars,
            )
        return mixture_probabilities(
            theta,
            category_counts=self.category_counts,
            model=self.model,
            weights=self.weights,
            zero_inflated=self.zero_inflated,
            **pars,
        )


def _fit(
    responses,
    *,
    groups,
    classes,
    model,
    category_counts,
    starts,
    shared_a,
    shared_steps,
    free_distribution,
    zero_inflated,
    quadrature_points,
    max_iter,
    tolerance,
    jit,
):
    if not jax.config.x64_enabled:
        raise ValueError("Enable JAX x64 before fitting")
    if model not in ("rasch", "2pl", "grm"):
        raise ValueError("Unknown response model")
    for v, name, minimum in (
        (classes, "classes", 1),
        (quadrature_points, "quadrature_points", 3),
        (max_iter, "max_iter", 1),
    ):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not all(isinstance(v, bool) for v in (free_distribution, zero_inflated, jit)):
        raise ValueError("Flags must be boolean")
    if not np.isscalar(tolerance) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    y = np.asarray(responses, dtype=float)
    if y.ndim != 2 or min(y.shape) < 2:
        raise ValueError("responses must be a matrix with at least two subjects/items")
    J = y.shape[1]
    C = classes
    counts = (
        np.full(J, 2)
        if category_counts is None and model != "grm"
        else np.asarray(category_counts)
    )
    if counts.shape != (J,) or counts.dtype.kind not in "iu" or np.any(counts < 2):
        raise ValueError("Declare integer category_counts >=2 per item")
    if model != "grm" and np.any(counts != 2):
        raise ValueError("Binary models require two categories")
    if not (np.isnan(y) | ((y >= 0) & (y < counts) & (y == np.floor(y)))).all():
        raise ValueError("responses must be declared integer categories or NaN")
    if np.any(np.all(np.isnan(y), axis=0)):
        raise ValueError("Each item needs an observed response")
    if groups is not None:
        groups = np.asarray(groups)
        if (
            groups.shape != (len(y),)
            or groups.dtype.kind not in "iu"
            or not np.array_equal(np.unique(groups), np.arange(C))
        ):
            raise ValueError(
                "groups must be contiguous integer labels starting at zero"
            )
        if zero_inflated:
            raise ValueError("Structural component belongs to latent mixtures")

    def shared(value, name):
        if value is None:
            return np.zeros(J, dtype=bool)
        v = np.asarray(value)
        if v.shape != (J,) or v.dtype.kind != "b":
            raise ValueError(f"{name} must be a boolean per item")
        return v.copy()

    share_a = shared(shared_a, "shared_a")
    share_b = shared(shared_steps, "shared_steps")
    if model == "rasch":
        share_a[:] = True
    if free_distribution and C > 1 and np.sum(share_a & share_b) < 2:
        raise ValueError(
            "Free latent distributions require at least two fully shared anchor items"
        )
    if groups is not None:
        for c in range(C):
            missing = np.all(np.isnan(y[groups == c]), axis=0)
            if np.any(missing & ~(share_a & share_b)):
                raise ValueError(
                    "Every group-specific item needs observations in its group"
                )
    start = dict(starts)
    if set(start) - {"a", "steps", "mu", "sigma", "weights"}:
        raise ValueError("Unexpected starts")
    a = np.asarray(start.get("a", np.ones((C, J))), dtype=float)
    shape = (C, J, int(counts.max() - 1))
    if "steps" not in start:
        raise ValueError("starts requires steps for every group/class")
    b = np.asarray(start["steps"], dtype=float)
    valid = np.broadcast_to(
        np.arange(shape[-1])[None, None, :] < counts[None, :, None] - 1, shape
    )
    if a.shape != (C, J) or not np.isfinite(a).all() or np.any(a < 1e-6):
        raise ValueError("Invalid slope starts")
    if model == "rasch" and not np.all(a == 1):
        raise ValueError("Rasch slopes must equal one")
    if b.shape != shape or not np.isfinite(b[valid]).all():
        raise ValueError("Invalid step starts")
    b = np.where(valid, b, 0.0)
    if not np.allclose(
        a[:, share_a], a[0, share_a], rtol=0, atol=1e-12
    ) or not np.allclose(b[:, share_b], b[0, share_b], rtol=0, atol=1e-12):
        raise ValueError("Shared parameters need equal starts")
    mu = np.asarray(start.get("mu", np.zeros(C)), dtype=float)
    sigma = np.asarray(start.get("sigma", np.ones(C)), dtype=float)
    if (
        mu.shape != (C,)
        or sigma.shape != (C,)
        or not np.isfinite(mu).all()
        or not np.isfinite(sigma).all()
        or np.any(sigma < 1e-3)
    ):
        raise ValueError("Invalid latent distribution starts")
    if (
        mu[0] != 0
        or sigma[0] != 1
        or (not free_distribution and (np.any(mu != 0) or np.any(sigma != 1)))
    ):
        raise ValueError(
            "Reference distribution must be N(0,1); fixed distributions must all be N(0,1)"
        )
    raw = []
    lower = []
    upper = []

    def add(v, lo=-np.inf, hi=np.inf):
        index = len(raw)
        raw.append(v)
        lower.append(lo)
        upper.append(hi)
        return index

    ia = np.zeros((C, J), int)
    ib = np.zeros(shape, int)
    for c in range(C):
        for j in range(J):
            if model != "rasch":
                ia[c, j] = ia[0, j] if c and share_a[j] else add(a[c, j], 1e-6)
            for k in range(counts[j] - 1):
                ib[c, j, k] = ib[0, j, k] if c and share_b[j] else add(b[c, j, k])
    distribution_offset = len(raw)
    if free_distribution:
        for c in range(1, C):
            add(mu[c])
        for c in range(1, C):
            add(sigma[c], 1e-3)
    weight_offset = len(raw)
    components = C + int(zero_inflated)
    if groups is None:
        weights = np.asarray(
            start.get("weights", np.full(components, 1 / components)), dtype=float
        )
        if (
            weights.shape != (components,)
            or not np.isfinite(weights).all()
            or np.any(weights <= 0)
            or not np.isclose(weights.sum(), 1, atol=1e-10, rtol=0)
        ):
            raise ValueError("Weight starts must be positive and sum to one")
        logits = np.log(weights[1:] / weights[0])
        if np.any(abs(logits) > 20):
            raise ValueError("Initial weight log ratios must be within [-20,20]")
        for v in logits:
            add(v, -20, 20)
    elif "weights" in start:
        raise ValueError("Observed groups do not estimate class weights")
    order = []
    for c in range(C):
        for j in range(J):
            for k in range(counts[j] - 2):
                row = np.zeros(len(raw))
                row[ib[c, j, k + 1]] = 1
                row[ib[c, j, k]] = -1
                order.append(row)
    constraints = []
    if order:
        matrix = np.asarray(order)
        if np.any(matrix @ raw < 1e-6):
            raise ValueError("Graded thresholds must be strictly increasing")
        constraints = [LinearConstraint(matrix, 1e-6, np.inf)]
    coded = np.where(np.isnan(y), -1, y).astype(int)
    if groups is not None:
        coded = np.column_stack([coded, groups])
    unique, inverse, frequency = np.unique(
        coded, axis=0, return_inverse=True, return_counts=True
    )
    labels = unique[:, -1] if groups is not None else None
    patterns = jnp.asarray(unique[:, :J])
    freq = jnp.asarray(frequency)
    nodes, w = hermgauss(quadrature_points)
    nodes = jnp.asarray(nodes * np.sqrt(2))
    lw = jnp.asarray(np.log(w) - 0.5 * np.log(np.pi))
    ia, ib = jnp.asarray(ia), jnp.asarray(ib)
    valid_jax = jnp.asarray(valid)
    counts_jax = jnp.asarray(counts)

    def unpack(raw):
        slopes = jnp.ones((C, J)) if model == "rasch" else raw[ia]
        steps = jnp.where(valid_jax, raw[ib], 0.0)
        mean = (
            jnp.r_[0.0, raw[distribution_offset : distribution_offset + C - 1]]
            if free_distribution
            else jnp.zeros(C)
        )
        sd = (
            jnp.r_[
                1.0,
                raw[distribution_offset + C - 1 : distribution_offset + 2 * (C - 1)],
            ]
            if free_distribution
            else jnp.ones(C)
        )
        weight = (
            jax.nn.softmax(jnp.r_[0.0, raw[weight_offset:]]) if groups is None else None
        )
        return slopes, steps, mean, sd, weight

    def posterior(raw):
        slopes, steps, mean, sd, weight = unpack(raw)
        cll, post, theta = _component_posterior(
            patterns, nodes, lw, slopes, steps, mean, sd, counts_jax, model
        )
        if groups is None:
            ll, members = _mix_posterior(cll, patterns, weight, zero_inflated)
        else:
            ll = cll[jnp.arange(len(labels)), labels]
            members = jax.nn.one_hot(labels, C)
        return ll, members, post, theta

    def loss(raw):
        return -(posterior(raw)[0] @ freq) / freq.sum()

    evaluate = jax.value_and_grad(loss)
    if jit:
        evaluate = jax.jit(evaluate)

    def objective(raw):
        value, grad = evaluate(jnp.asarray(raw))
        if not np.isfinite(value) or not np.isfinite(grad).all():
            raise FloatingPointError("Nonfinite group/mixture objective")
        return float(value), np.asarray(grad)

    trace = [-objective(np.asarray(raw))[0] * len(y)]
    opt = minimize(
        objective,
        np.asarray(raw),
        jac=True,
        method="SLSQP",
        bounds=list(zip(lower, upper)),
        constraints=constraints,
        callback=lambda raw: trace.append(-objective(raw)[0] * len(y)),
        options={"maxiter": max_iter, "ftol": tolerance},
    )
    ll, members, post, theta = (jax.jit(posterior) if jit else posterior)(
        jnp.asarray(opt.x)
    )
    slopes, steps, mean, sd, weights = unpack(jnp.asarray(opt.x))
    cx = jnp.sum(post * theta[None, :, :], axis=-1)
    cv = jnp.sum(post * (theta[None, :, :] - cx[:, :, None]) ** 2, axis=-1)
    regular = members[:, :C]
    regular = regular / regular.sum(axis=1, keepdims=True)
    x = jnp.sum(regular * cx, axis=1)
    var = jnp.sum(regular * (cv + (cx - x[:, None]) ** 2), axis=1)
    value = float(ll @ freq)
    return GroupIRTResult(
        "mixture" if groups is None else "group",
        model,
        dict(
            a=slopes, steps=steps, mu=mean, sigma=sd, x=x[inverse], class_x=cx[inverse]
        ),
        dict(x=var[inverse], class_x=cv[inverse]),
        counts.copy(),
        None if weights is None else np.asarray(weights),
        members[inverse] if groups is None else None,
        None if groups is None else groups.copy(),
        zero_inflated,
        bool(opt.success),
        int(opt.nit),
        value,
        np.asarray(trace),
        str(opt.message),
        np.array([value]),
    )


def group_irt(
    responses,
    *,
    groups,
    starts,
    model="2pl",
    category_counts=None,
    shared_a=None,
    shared_steps=None,
    free_distribution=False,
    quadrature_points=41,
    max_iter=2000,
    tolerance=1e-10,
    jit=True,
):
    """Fit observed groups; integer labels 0..G-1 are known, not inferred.

    Models: rasch, 2pl and grm. Shared parameter flags are boolean (J,) arrays;
    defaults free all items. Free distributions estimate mu/sigma outside group
    zero, requiring at least two fully shared anchor items. starts supplies
    steps (G,J,max(K)-1), optionally a (G,J), mu/sigma (G,).
    """
    labels = np.asarray(groups)
    if labels.ndim != 1 or labels.dtype.kind not in "iu" or not len(labels):
        raise ValueError("groups must be integer labels")
    return _fit(
        responses,
        groups=labels,
        classes=len(np.unique(labels)),
        model=model,
        category_counts=category_counts,
        starts=starts,
        shared_a=shared_a,
        shared_steps=shared_steps,
        free_distribution=free_distribution,
        zero_inflated=False,
        quadrature_points=quadrature_points,
        max_iter=max_iter,
        tolerance=tolerance,
        jit=jit,
    )


def mixture_irt(
    responses,
    *,
    classes,
    starts,
    model="2pl",
    category_counts=None,
    shared_a=None,
    shared_steps=None,
    free_distribution=False,
    zero_inflated=False,
    quadrature_points=41,
    max_iter=2000,
    tolerance=1e-10,
    jit=True,
):
    """Fit latent regular classes plus an optional structural all-lowest class.

    starts is a parameter dict or a nonempty list of dicts for multiple starts.
    Return the converged fit with highest likelihood; if none converges, return
    the highest-likelihood failed fit with converged=False. Class labels are
    arbitrary. x and its variance condition on membership in regular classes;
    the structural class has no ability. Its posterior is the last column of
    memberships. Missing observations impose no restriction on structural membership.
    """
    candidates = starts if isinstance(starts, (list, tuple)) else [starts]
    if not candidates:
        raise ValueError("Supply at least one start")
    fits = [
        _fit(
            responses,
            groups=None,
            classes=classes,
            model=model,
            category_counts=category_counts,
            starts=s,
            shared_a=shared_a,
            shared_steps=shared_steps,
            free_distribution=free_distribution,
            zero_inflated=zero_inflated,
            quadrature_points=quadrature_points,
            max_iter=max_iter,
            tolerance=tolerance,
            jit=jit,
        )
        for s in candidates
    ]
    successful = [fit for fit in fits if fit.converged]
    best = max(successful or fits, key=lambda fit: fit.log_likelihood)
    best.start_log_likelihoods = np.array([fit.log_likelihood for fit in fits])
    return best
