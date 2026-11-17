"""Device-native spectral projected gradient for box/linear constrained fits.

The optional solver keeps the original closed feasible set. SciPy remains the
reference/default; this path uses a projected-gradient stopping criterion.
"""

from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np


def _project(x, lower, upper, matrix, rhs, pairs=()):
    """Dykstra projection onto a box and linear half-spaces A x >= b."""
    rows = matrix.shape[0]
    if rows == 0:
        return jnp.clip(x, lower, upper), jnp.array(True)
    if pairs:
        point = jnp.clip(x, lower, upper)
        for row, (left, right) in enumerate(pairs):
            gap = rhs[row]
            boundary = jnp.clip(
                (x[left] + x[right] - gap) / 2,
                jnp.maximum(lower[left], lower[right] - gap),
                jnp.minimum(upper[left], upper[right] - gap),
            )
            violates = point[right] - point[left] < gap
            point = point.at[left].set(jnp.where(violates, boundary, point[left]))
            point = point.at[right].set(
                jnp.where(violates, boundary + gap, point[right])
            )
        return point, jnp.array(True)
    norms = jnp.sum(matrix * matrix, axis=1)

    def cycle(carry):
        i, previous, box_correction, corrections, _ = carry
        previous_box, previous_corrections = box_correction, corrections
        z = previous + box_correction
        point = jnp.clip(z, lower, upper)
        box_correction = z - point

        def row(k, state):
            point, corrections = state
            z = point + corrections[k] * matrix[k]
            shift = jnp.maximum(rhs[k] - matrix[k] @ z, 0.0) / norms[k]
            return z + shift * matrix[k], corrections.at[k].set(-shift)

        point, corrections = jax.lax.fori_loop(0, rows, row, (point, corrections))
        violation = jnp.maximum(
            jnp.max(jnp.maximum(lower - point, point - upper)),
            jnp.max(rhs - matrix @ point),
        )
        done = (
            (jnp.max(jnp.abs(point - previous)) <= 1e-10)
            & (jnp.max(jnp.abs(box_correction - previous_box)) <= 1e-10)
            & (jnp.max(jnp.abs(corrections - previous_corrections)) <= 1e-10)
            & (violation <= 1e-10)
        )
        return i + 1, point, box_correction, corrections, done

    result = jax.lax.while_loop(
        lambda c: (c[0] < 1000) & ~c[-1],
        cycle,
        (jnp.array(0), x, jnp.zeros_like(x), jnp.zeros(rows), jnp.array(False)),
    )
    return result[1], result[-1]


@partial(jax.jit, static_argnames=("loss", "config", "max_iter", "pairs"))
def _solve(
    loss, x, args, lower, upper, matrix, rhs, tolerance, *, config, max_iter, pairs=()
):
    evaluate = jax.value_and_grad(loss)

    def objective(x):
        return evaluate(x, *args, **dict(config))

    value, gradient = objective(x)
    trace = jnp.full(max_iter + 1, jnp.nan).at[0].set(value)
    recent = jnp.full(10, value)
    finite = jnp.isfinite(value) & jnp.all(jnp.isfinite(gradient))
    # Status: 0 running, 1 converged, 2 line search, 3 projection, 4 nonfinite.
    initial = (
        jnp.array(0),
        x,
        value,
        gradient,
        jnp.array(1.0),
        jnp.where(finite, 0, 4),
        trace,
        recent,
    )

    def body(state):
        i, x, value, gradient, scale, _, trace, recent = state
        projected, ok = _project(x - gradient, lower, upper, matrix, rhs, pairs)
        converged = jnp.max(jnp.abs(projected - x)) <= tolerance
        target, target_ok = _project(
            x - scale * gradient, lower, upper, matrix, rhs, pairs
        )
        direction = target - x
        slope = gradient @ direction
        ceiling = jnp.max(recent)

        def trial(alpha):
            point = x + alpha * direction
            v, g = objective(point)
            finite = jnp.isfinite(v) & jnp.all(jnp.isfinite(g))
            accepted = finite & (v <= ceiling + 1e-4 * alpha * slope)
            return point, v, g, accepted

        point, v, g, accepted = trial(jnp.array(1.0))
        search = jax.lax.while_loop(
            lambda s: (s[0] < 40) & ~s[-1],
            lambda s: (s[0] + 1, s[1] * 0.5, *trial(s[1] * 0.5)),
            (jnp.array(0), jnp.array(1.0), point, v, g, accepted),
        )
        _, _, point, v, g, accepted = search
        delta, dg = point - x, g - gradient
        curvature = delta @ dg
        next_scale = jnp.where(curvature > 0, (delta @ delta) / curvature, 1.0)
        next_scale = jnp.clip(next_scale, 1e-6, 1e3)
        status = jnp.where(
            ~(ok & target_ok), 3, jnp.where(converged, 1, jnp.where(accepted, 0, 2))
        )
        advance = status == 0
        point = jnp.where(advance, point, x)
        v = jnp.where(advance, v, value)
        g = jnp.where(advance, g, gradient)
        return (
            i + advance.astype(int),
            point,
            v,
            g,
            next_scale,
            status,
            jax.lax.cond(advance, lambda t: t.at[i + 1].set(v), lambda t: t, trace),
            jax.lax.cond(advance, lambda t: t.at[i % 10].set(v), lambda t: t, recent),
        )

    result = jax.lax.while_loop(
        lambda s: (s[0] < max_iter) & (s[5] == 0), body, initial
    )
    i, x, value, gradient, _, status, trace, _ = result
    projected, ok = _project(x - gradient, lower, upper, matrix, rhs, pairs)
    status = jnp.where(
        (status == 0) & ok & (jnp.max(jnp.abs(projected - x)) <= tolerance), 1, status
    )
    return x, value, i, status, trace


def device_minimize(
    loss, initial, args, *, lower, upper, constraints=(), config=(), max_iter, tolerance
):
    """Host boundary around one compiled solve; no iteration callbacks."""
    x = np.asarray(initial, dtype=float)
    matrices, limits = [], []
    for constraint in constraints:
        a = np.asarray(constraint.A, dtype=float)
        lo = np.broadcast_to(constraint.lb, a.shape[0])
        hi = np.broadcast_to(constraint.ub, a.shape[0])
        matrices.extend((a[np.isfinite(lo)], -a[np.isfinite(hi)]))
        limits.extend((lo[np.isfinite(lo)], -hi[np.isfinite(hi)]))
    matrix = np.concatenate(matrices) if matrices else np.zeros((0, len(x)))
    rhs = np.concatenate(limits) if limits else np.zeros(0)
    if np.any(np.sum(matrix * matrix, axis=1) == 0):
        raise ValueError("Constraint rows must be nonzero")
    pairs = []
    used = set()
    for row in matrix:
        left, right = np.flatnonzero(row == -1), np.flatnonzero(row == 1)
        if (
            len(left) != 1
            or len(right) != 1
            or np.count_nonzero(row) != 2
            or int(left[0]) in used
            or int(right[0]) in used
        ):
            pairs = []
            break
        pair = (int(left[0]), int(right[0]))
        pairs.append(pair)
        used.update(pair)
    x, value, iterations, status, trace = _solve(
        loss,
        jnp.asarray(x),
        args,
        jnp.asarray(lower),
        jnp.asarray(upper),
        jnp.asarray(matrix),
        jnp.asarray(rhs),
        jnp.asarray(tolerance),
        config=config,
        max_iter=max_iter,
        pairs=tuple(pairs),
    )
    iterations, status = int(iterations), int(status)
    if status == 4:
        raise FloatingPointError("Nonfinite device objective or gradient")
    messages = {
        0: "Iteration limit reached",
        1: "Converged (projected gradient)",
        2: "Line search failed",
        3: "Constraint projection failed",
    }
    return SimpleNamespace(
        x=x,
        fun=value,
        nit=iterations,
        success=status == 1,
        message=messages[status],
        trace=np.asarray(trace)[: iterations + 1],
    )
