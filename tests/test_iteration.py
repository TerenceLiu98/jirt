"""Compiled iteration reuse, stopping and invalid-state behavior."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jirt._iteration import run_explicit, _cached_iterations, change

jax.config.update("jax_enable_x64", True)


def _step(state, data):
    return (state[0] + data,)


def test_reuses_compilation_without_stale_data():
    _cached_iterations.clear_cache()
    options = dict(max_iter=3, tolerance=0.0, correlation=False, count=1)
    a = run_explicit(_step, (jnp.zeros(2),), jnp.ones(2), **options)
    size = _cached_iterations._cache_size()
    b = run_explicit(
        _step, (jnp.zeros(2),), jnp.full(2, 2.0, dtype=jnp.float64), **options
    )
    assert _cached_iterations._cache_size() == size
    np.testing.assert_array_equal(a[0][0], [3, 3])
    np.testing.assert_array_equal(b[0][0], [6, 6])
    assert not a[1] and a[2] == 3


def test_minimum_iterations_and_nonfinite():
    options = dict(max_iter=10, tolerance=1e-8, correlation=True, count=1, min_iter=3)
    fit = run_explicit(_step, (jnp.zeros(2),), jnp.zeros(2), **options)
    assert fit[1] and fit[2] == 3 and fit[3].shape == (3, 1)
    with pytest.raises(FloatingPointError, match="iteration 1"):
        run_explicit(_step, (jnp.zeros(2),), jnp.array([jnp.inf, 0.0]), **options)


def test_correlation_and_constant_fallback():
    np.testing.assert_allclose(
        change(jnp.array([1.0, 2.0, 4.0]), jnp.array([5.0, 4.0, 3.0]), True),
        1 - np.corrcoef([1, 2, 4], [5, 4, 3])[0, 1],
        atol=1e-14,
    )
    assert change(jnp.ones(3), jnp.full(3, 2.0), True) == 1


@pytest.mark.parametrize(
    "model", ["network", "ordinal", "hierarchical", "poisson", "binary_em", "dynamic"]
)
def test_model_reuses_kernel_with_changed_observations(model):
    import json
    from pathlib import Path
    from jirt import Control, network_irt, ord_irt, hier_irt, pois_irt, bin_irt, dyn_irt

    functions = dict(
        network=network_irt,
        ordinal=ord_irt,
        hierarchical=hier_irt,
        poisson=pois_irt,
        binary_em=bin_irt,
        dynamic=dyn_irt,
    )
    case = json.loads(
        (Path(__file__).parent / "fixtures" / f"{model}.json").read_text()
    )
    kwargs = case["kwargs"]
    control = Control(max_iter=5, tolerance=0, convergence="absolute")
    _cached_iterations.clear_cache()
    first = functions[model](**kwargs, control=control)
    size = _cached_iterations._cache_size()
    key = {
        "network": "edges",
        "ordinal": "responses",
        "hierarchical": "votes",
        "poisson": "counts",
        "binary_em": "votes",
        "dynamic": "votes",
    }[model]
    values = np.asarray(kwargs[key], dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    flat = values.ravel()
    if model == "network":
        flat[finite] = 1 - flat[finite]
    elif model in ("hierarchical", "binary_em", "dynamic"):
        flat[finite] = -flat[finite]
    elif model == "ordinal":
        flat[finite] = np.where(
            flat[finite] == 1, 3, np.where(flat[finite] == 3, 1, flat[finite])
        )
    else:
        flat[finite] += 1
    kwargs[key] = values
    second = functions[model](**kwargs, control=control)
    assert _cached_iterations._cache_size() == size
    eager = functions[model](
        **kwargs,
        control=Control(max_iter=5, tolerance=0, convergence="absolute", jit=False),
    )
    for key in second.means:
        np.testing.assert_allclose(
            second.means[key], eager.means[key], atol=1e-8, rtol=1e-6
        )
    assert any(
        not np.allclose(first.means[key], second.means[key]) for key in first.means
    )


def test_diagnostic_kernel_reuses_changed_frequencies():
    from itertools import product
    from jirt.diagnostic import diagnostic_irt, _fit_compiled

    patterns = np.array(list(product([0.0, 1.0], repeat=4)))
    first_data = np.vstack([patterns, np.zeros((10, 4))])
    second_data = np.vstack([patterns, np.ones((10, 4))])
    options = dict(q_matrix=[[1, 0], [0, 1], [1, 1], [1, 0]], max_iter=20)
    _fit_compiled.clear_cache()
    first = diagnostic_irt(first_data, **options)
    size = _fit_compiled._cache_size()
    second = diagnostic_irt(second_data, **options)
    assert _fit_compiled._cache_size() == size
    eager = diagnostic_irt(second_data, jit=False, **options)
    np.testing.assert_allclose(second.log_likelihood, eager.log_likelihood, atol=1e-10)
    assert not np.allclose(first.slip, second.slip)
