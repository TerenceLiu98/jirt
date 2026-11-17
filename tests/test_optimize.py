"""Constrained device optimization: boundaries, failures and compilation reuse."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import LinearConstraint
from jirt._optimize import device_minimize, _solve

jax.config.update("jax_enable_x64", True)


def _quadratic(x, target):
    return 0.5 * jnp.sum((x - target) ** 2)


def test_closed_boundaries_and_order_constraint():
    fit = device_minimize(
        _quadratic,
        [0.2, 0.8],
        (jnp.array([2.0, -1.0]),),
        lower=[0.0, 0.0],
        upper=[1.0, 1.0],
        constraints=[LinearConstraint([[-1.0, 1.0]], 0.2, np.inf)],
        max_iter=100,
        tolerance=1e-8,
    )
    assert fit.success, fit.message
    np.testing.assert_allclose(fit.x, [0.4, 0.6], atol=1e-8)
    boundary = device_minimize(
        _quadratic,
        [0.2, 0.8],
        (jnp.array([-2.0, 3.0]),),
        lower=[0.0, 0.0],
        upper=[1.0, 1.0],
        max_iter=100,
        tolerance=1e-8,
    )
    np.testing.assert_array_equal(boundary.x, [0.0, 1.0])
    assert boundary.success


def test_explicit_data_reuses_compilation():
    _solve.clear_cache()
    options = dict(lower=[-np.inf] * 2, upper=[np.inf] * 2, max_iter=10, tolerance=1e-8)
    a = device_minimize(_quadratic, [0.0, 0.0], (jnp.array([1.0, 2.0]),), **options)
    size = _solve._cache_size()
    b = device_minimize(_quadratic, [0.0, 0.0], (jnp.array([3.0, 4.0]),), **options)
    assert _solve._cache_size() == size
    np.testing.assert_allclose(a.x, [1.0, 2.0])
    np.testing.assert_allclose(b.x, [3.0, 4.0])


def _rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def test_limit_and_invalid_objective():
    import pytest

    fit = device_minimize(
        _rosenbrock,
        [-1.0, 2.0],
        (),
        lower=[-np.inf] * 2,
        upper=[np.inf] * 2,
        max_iter=1,
        tolerance=1e-12,
    )
    assert not fit.success and fit.nit == 1
    assert fit.message == "Iteration limit reached"
    assert fit.trace.shape == (2,)
    with pytest.raises(FloatingPointError):
        device_minimize(
            _quadratic,
            [0.0, 0.0],
            (jnp.array([jnp.nan, 1.0]),),
            lower=[-np.inf] * 2,
            upper=[np.inf] * 2,
            max_iter=10,
            tolerance=1e-6,
        )


def test_overlapping_order_constraints():
    fit = device_minimize(
        _quadratic,
        [0.0, 1.0, 2.0],
        (jnp.array([3.0, 2.0, 1.0]),),
        lower=[0.0] * 3,
        upper=[4.0] * 3,
        constraints=[
            LinearConstraint([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]], 0.1, np.inf)
        ],
        max_iter=100,
        tolerance=1e-7,
    )
    assert fit.success, fit.message
    np.testing.assert_allclose(fit.x, [1.9, 2.0, 2.1], atol=1e-7)


def test_dykstra_does_not_stop_on_a_stalled_feasible_point():
    from jirt._optimize import _project

    point, ok = jax.jit(_project)(
        jnp.array([10.0, -10.0, -10.0]),
        jnp.zeros(3),
        jnp.ones(3),
        jnp.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]]),
        jnp.full(2, 0.1),
    )
    assert ok
    np.testing.assert_allclose(point, [0.0, 0.1, 0.2], atol=1e-8)


def test_projection_exhaustion_is_not_reported_as_success():
    from jirt._optimize import _project

    _, ok = jax.jit(_project)(
        jnp.array([100.0, -100.0, -100.0]),
        jnp.zeros(3),
        jnp.ones(3),
        jnp.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]]),
        jnp.full(2, 0.1),
    )
    assert not ok
