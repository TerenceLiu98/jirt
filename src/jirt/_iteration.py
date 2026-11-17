"""Device-side convergence and control flow shared by iterative estimators."""

import jax
import jax.numpy as jnp
import numpy as np


def change(before, after, correlation):
    u, v = jnp.ravel(before), jnp.ravel(after)
    absolute = jnp.max(jnp.abs(v - u))
    if not correlation:
        return absolute
    uc, vc = u - jnp.mean(u), v - jnp.mean(v)
    uu, vv = jnp.sum(uc * uc), jnp.sum(vc * vc)
    denominator = jnp.sqrt(uu * vv)
    corr = jnp.clip(
        jnp.sum(uc * vc) / jnp.where(denominator > 0, denominator, 1.0), -1.0, 1.0
    )
    return jnp.where((uu > 0) & (vv > 0), jnp.maximum(0.0, 1 - corr), absolute)


def finite_state(state):
    return jnp.all(
        jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(state)])
    )


def compiled_iterations(
    step,
    state,
    data,
    tolerance,
    *,
    max_iter,
    correlation,
    count,
    min_iter=3,
    select=None,
    valid=None,
    finalize=None,
):
    """Reusable module-level executor for explicit-data update functions."""
    history = jnp.full((max_iter, count), jnp.nan)
    initial = (jnp.array(0), state, jnp.array(False), jnp.array(True), history)

    def condition(carry):
        i, _, converged, ok, _ = carry
        return (i < max_iter) & ~converged & ok

    def body(carry):
        i, old, _, _, history = carry
        new = step(old, data)
        errors = jnp.stack(
            [
                change(u, v, correlation)
                for u, v in zip(
                    old[:count] if select is None else select(old, data),
                    new[:count] if select is None else select(new, data),
                )
            ]
        )
        ok = finite_state(new) if valid is None else valid(new, data)
        converged = ok & (i + 1 >= min_iter) & (jnp.max(errors) < tolerance)
        if finalize is not None:
            new = jax.lax.cond(
                converged, lambda _: finalize(new, old, data), lambda _: new, None
            )
            ok = ok & (finite_state(new) if valid is None else valid(new, data))
        return i + 1, new, converged, ok, history.at[i].set(errors)

    return jax.lax.while_loop(condition, body, initial)


_cached_iterations = jax.jit(
    compiled_iterations,
    static_argnames=(
        "step",
        "max_iter",
        "correlation",
        "count",
        "min_iter",
        "select",
        "valid",
        "finalize",
    ),
)


def run_explicit(
    step,
    state,
    data,
    *,
    max_iter,
    tolerance,
    correlation,
    count,
    min_iter=3,
    select=None,
    valid=None,
    finalize=None,
):
    i, state, converged, ok, history = _cached_iterations(
        step,
        state,
        data,
        jnp.asarray(tolerance),
        max_iter=max_iter,
        correlation=correlation,
        count=count,
        min_iter=min_iter,
        select=select,
        valid=valid,
        finalize=finalize,
    )
    i = int(i)
    if not bool(ok):
        raise FloatingPointError(f"Nonfinite update at iteration {i}")
    return state, bool(converged), i, np.asarray(history)[:i]
