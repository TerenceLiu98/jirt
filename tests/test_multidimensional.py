"""Numerical and structural checks for compensatory MIRT."""

import itertools

import jax
import numpy as np
import pytest
from scipy.special import expit
from jirt.logistic import logistic_irt
from jirt.multidimensional import (
    multidimensional_irt,
    multidimensional_probabilities,
    _quadrature,
)

jax.config.update("jax_enable_x64", True)


def frequency_data(a, d, points=15, n=50000):
    nodes, lw = _quadrature(a.shape[1], points, 100000)
    p = expit(nodes @ a.T + d)
    patterns = np.array(list(itertools.product([0, 1], repeat=len(d))))
    freq = np.exp(patterns @ np.log(p).T + (1 - patterns) @ np.log1p(-p).T) @ np.exp(lw)
    return np.repeat(patterns, np.maximum(1, np.round(n * freq).astype(int)), axis=0)


def test_formula_rotation_and_jit():
    rng = np.random.default_rng(21)
    a = rng.normal(size=(7, 2))
    theta = rng.normal(size=(10, 2))
    d = rng.normal(size=7)
    q, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    expected = expit(theta @ a.T + d)
    np.testing.assert_allclose(
        multidimensional_probabilities(theta @ q, a=a @ q, d=d), expected, atol=1e-14
    )
    np.testing.assert_allclose(
        jax.jit(multidimensional_probabilities)(theta, a=a, d=d), expected, atol=1e-14
    )


def test_unidimensional_reduction():
    a = np.linspace(0.8, 1.4, 6)[:, None]
    d = np.linspace(-1, 1, 6)
    y = frequency_data(a, d, points=31)
    one = logistic_irt(y, model="2pl", quadrature_points=31, tolerance=1e-13)
    multi = multidimensional_irt(y, dimensions=1, quadrature_points=31, tolerance=1e-13)
    assert one.converged and multi.converged
    np.testing.assert_allclose(multi.means["a"][:, 0], one.means["a"], atol=2e-5)
    np.testing.assert_allclose(
        multi.means["d"], -one.means["a"] * one.means["b"], atol=2e-5
    )
    np.testing.assert_allclose(multi.means["x"][:, 0], one.means["x"], atol=2e-5)
    np.testing.assert_allclose(
        multi.variances["x"][:, 0], one.variances["x"], atol=2e-5
    )
    assert abs(multi.log_likelihood - one.log_likelihood) < 1e-6


@pytest.mark.parametrize("structure", ["multidimensional", "bifactor", "testlet"])
def test_recovery_and_structural_zeros(structure):
    if structure == "multidimensional":
        a = np.array([[1, 0], [0.8, 0], [1.3, 0], [0, 1.1], [0, 0.9], [0, 1.4]])
        kwargs = dict(dimensions=2, loading_mask=a != 0)
    else:
        a = np.array(
            [
                [1, 0.7, 0],
                [0.8, 0.7, 0],
                [1.3, 0.7, 0],
                [1.1, 0, 0.6],
                [0.9, 0, 0.6],
                [1.4, 0, 0.6],
            ]
        )
        # Pure general-factor items anchor the general/specific decomposition.
        a = np.vstack([a, [[0.9, 0, 0], [1.2, 0, 0], [1.5, 0, 0]]])
        kwargs = dict(groups=[0, 0, 0, 1, 1, 1, -1, -1, -1], structure=structure)
    d = np.linspace(-1, 1, len(a))
    y = frequency_data(a, d, n=200000)
    fit = multidimensional_irt(y, **kwargs, tolerance=1e-12, max_iter=2000)
    assert fit.converged, fit.message
    np.testing.assert_array_equal(np.asarray(fit.means["a"])[a == 0], 0)
    np.testing.assert_allclose(fit.means["a"], a, atol=0.06, rtol=0)
    np.testing.assert_allclose(fit.means["d"], d, atol=0.01, rtol=0)
    if structure == "testlet":
        np.testing.assert_allclose(
            fit.means["testlet_variance"], [0.49, 0.36], atol=0.04
        )
        np.testing.assert_array_equal(
            np.asarray(fit.means["a"])[:3, 1], np.repeat(fit.means["a"][0, 1], 3)
        )


def test_missing_and_eager():
    rng = np.random.default_rng(2)
    y = rng.binomial(1, 0.5, size=(100, 6)).astype(float)
    y[0] = np.nan
    y[1, 2] = np.nan
    eager = multidimensional_irt(y, max_iter=2, jit=False, quadrature_points=7)
    compiled = multidimensional_irt(y, max_iter=2, quadrature_points=7)
    assert not compiled.converged
    np.testing.assert_allclose(eager.means["a"], compiled.means["a"], atol=1e-12)
    np.testing.assert_allclose(compiled.means["x"][0], 0, atol=1e-14)
    np.testing.assert_allclose(compiled.covariances["x"][0], np.eye(2), atol=1e-14)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dimensions": 0},
        {"dimensions": True},
        {"quadrature_points": 2},
        {"dimensions": 3, "max_nodes": 10},
        {"structure": "bad"},
        {"loading_mask": np.ones((6, 2))},
        {"anchors": [8, 0]},
        {"structure": "testlet", "groups": [0, 0, 1, 1, 1, 1]},
        {"structure": "bifactor", "groups": [1] * 6},
        {"starts": {"b": np.zeros(6)}},
        {"tolerance": float("nan")},
    ],
)
def test_invalid_specifications(kwargs):
    with pytest.raises(ValueError):
        multidimensional_irt(np.tile([0, 1, 0, 1, 0, 1], (10, 1)), **kwargs)


def test_exploratory_recovery_after_rotation_alignment():
    from scipy.linalg import orthogonal_procrustes

    a = np.array(
        [[1.0, 0], [0.3, 1.2], [1.3, 0.2], [-0.2, 1.1], [0.8, 0.5], [0.4, 1.4]]
    )
    d = np.linspace(-1, 1, 6)
    y = frequency_data(a, d, n=200000)
    fit = multidimensional_irt(y, dimensions=2, tolerance=1e-13, max_iter=2000)
    assert fit.converged
    # Compare to an equivalent rotated generating orientation.
    angle = 0.63
    q = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    truth = a @ q
    rotation, _ = orthogonal_procrustes(np.asarray(fit.means["a"]), truth)
    np.testing.assert_allclose(
        np.asarray(fit.means["a"]) @ rotation, truth, atol=0.01, rtol=0
    )
    np.testing.assert_allclose(fit.means["d"], d, atol=0.005, rtol=0)
    assert fit.means["a"][0, 1] == 0
    assert fit.means["a"][0, 0] > 0 and fit.means["a"][1, 1] > 0


def test_testlet_zero_variance_reduces_to_unidimensional_probability():
    theta = np.array([[0.2, -10, 30], [-0.5, 20, -2]])
    a = np.array([[1.0, 0, 0], [0.8, 0, 0]])
    d = np.array([-0.4, 0.6])
    np.testing.assert_allclose(
        multidimensional_probabilities(theta, a=a, d=d),
        expit(theta[:, :1] @ a[:, :1].T + d),
    )


def test_structured_start_constraints_and_general_only_items():
    y = np.tile([0, 1, 0, 1, 0, 1, 0], (10, 1))
    a = np.array([[1, 0.5, 0]] * 3 + [[1, 0, 0.5]] * 3 + [[1, 0, 0]], dtype=float)
    a[0, 1] = 0.8
    with pytest.raises(ValueError, match="equal"):
        multidimensional_irt(
            y, structure="testlet", groups=[0, 0, 0, 1, 1, 1, -1], starts={"a": a}
        )
    a[0, 1] = 0.5
    a[0, 2] = 0.1
    with pytest.raises(ValueError, match="cross-loadings"):
        multidimensional_irt(
            y, structure="bifactor", groups=[0, 0, 0, 1, 1, 1, -1], starts={"a": a}
        )
