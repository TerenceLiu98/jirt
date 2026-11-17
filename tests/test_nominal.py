"""Independent probability identities for nominal and nested models."""

import jax
import numpy as np
import pytest
from scipy.special import expit, softmax
from jirt.nominal import nominal_probabilities, nested_probabilities

jax.config.update("jax_enable_x64", True)


def test_nominal_softmax_and_identification():
    theta = np.linspace(-4, 4, 17)
    a = np.array([[0, 0.5, -0.3], [0, 1, np.nan]])
    d = np.array([[0, -0.2, 0.4], [0, 0.3, np.nan]])
    p = np.asarray(
        jax.jit(nominal_probabilities)(
            theta, slopes=a, intercepts=d, category_counts=np.array([3, 2])
        )
    )
    np.testing.assert_allclose(
        p[:, 0], softmax(theta[:, None] * a[0] + d[0], axis=1), atol=1e-14
    )
    np.testing.assert_allclose(p.sum(axis=-1), 1, atol=1e-14)
    np.testing.assert_array_equal(p[:, 1, 2], 0)
    np.testing.assert_allclose(
        p,
        nominal_probabilities(
            theta, slopes=a + 2, intercepts=d - 3, category_counts=[3, 2]
        ),
        atol=1e-14,
    )


@pytest.mark.parametrize("c,u", [(0, 1), (0.15, 1), (0, 0.9), (0.15, 0.9)])
def test_nested_collapse_and_extremes(c, u):
    theta = np.array([-1e6, -3, 0, 2, 1e6])
    a = np.array([0.8, 1.2])
    b = np.array([-0.2, 0.4])
    p = np.asarray(
        jax.jit(nested_probabilities)(
            theta,
            slopes=np.array([[0, 0.7, -0.4], [0.2, 0, np.nan]]),
            intercepts=np.zeros((2, 3)),
            category_counts=np.array([3, 2]),
            answer_keys=np.array([1, 0]),
            a=a,
            b=b,
            c=c,
            u=u,
        )
    )
    expected = c + (u - c) * expit(a * (theta[:, None] - b))
    np.testing.assert_allclose(p[:, 0, 1], expected[:, 0], atol=1e-14)
    np.testing.assert_allclose(p[:, 1, 0], expected[:, 1], atol=1e-14)
    np.testing.assert_allclose(p[:, 0, 0] + p[:, 0, 2], 1 - expected[:, 0], atol=1e-14)
    np.testing.assert_allclose(p.sum(axis=-1), 1, atol=1e-14)
    np.testing.assert_array_equal(p[:, 1, 2], 0)


from jirt.nominal import nominal_irt
from jirt.logistic import logistic_irt


@pytest.mark.parametrize("model", ["nominal", "2pl", "3pl", "3plu", "4pl"])
def test_fitting_missing_categories_and_jit(model):
    y = np.array(
        [
            [0, 0, 1],
            [2, 1, 2],
            [0, 2, 0],
            [2, 0, 2],
            [np.nan, np.nan, np.nan],
            [0, np.nan, 1],
        ]
    )
    kw = dict(
        category_counts=[3] * 3,
        model=model,
        answer_keys=None if model == "nominal" else [2, 1, 0],
        max_iter=3,
    )
    fit = nominal_irt(y, **kw)
    eager = nominal_irt(y, **kw, jit=False)
    np.testing.assert_allclose(fit.means["slopes"], eager.means["slopes"], atol=1e-11)
    np.testing.assert_allclose(fit.predict().sum(axis=-1), 1, atol=1e-13)
    assert abs(float(fit.means["x"][4])) < 1e-13
    assert abs(float(fit.variances["x"][4]) - 1) < 1e-13
    assert np.isfinite(fit.log_likelihood)
    for j in range(3):
        reference = 0 if model == "nominal" or kw["answer_keys"][j] != 0 else 1
        assert fit.means["slopes"][j, reference] == 0
        assert fit.means["intercepts"][j, reference] == 0


def test_binary_nested_reduction():
    rng = np.random.default_rng(18)
    theta = rng.normal(size=1500)
    y = rng.binomial(
        1, expit(theta[:, None] * np.linspace(0.8, 1.5, 5) - np.linspace(-1, 1, 5))
    )
    fit = nominal_irt(
        y, category_counts=[2] * 5, answer_keys=[1] * 5, model="2pl", tolerance=1e-13
    )
    binary = logistic_irt(y, model="2pl", tolerance=1e-13)
    assert fit.converged and binary.converged
    np.testing.assert_allclose(fit.means["a"], binary.means["a"], atol=1e-5)
    assert abs(fit.log_likelihood - binary.log_likelihood) < 1e-6


@pytest.mark.parametrize(
    "kwargs",
    [
        {"category_counts": [2, 3]},
        {"category_counts": [3.0, 3.0]},
        {"category_counts": [3, 3], "model": "2pl"},
        {"category_counts": [3, 3], "model": "2pl", "answer_keys": [3, 0]},
        {"category_counts": [3, 3], "answer_keys": [0, 1]},
        {"category_counts": [3, 3], "starts": {"slopes": np.ones((2, 3))}},
        {
            "category_counts": [3, 3],
            "model": "4pl",
            "answer_keys": [0, 1],
            "starts": {"c": np.ones(2) * 0.8, "u": np.ones(2) * 0.2},
        },
    ],
)
def test_invalid_specifications(kwargs):
    with pytest.raises(ValueError):
        nominal_irt([[0, 1], [2, 0]], **kwargs)
