import jax
import numpy as np
import pytest
from scipy.special import expit

from jirt import sequential_irt, sequential_probabilities

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("model", ["sequential", "tutz"])
def test_stage_formula_and_binary_reduction(model):
    theta = np.array([-1e6, -2, 0, 3, 1e6])
    a = np.array([0.8, 1.3])
    steps = np.array([[0.7, np.nan], [1.0, -0.5]])
    p = np.asarray(
        sequential_probabilities(
            theta, a=a, steps=steps, category_counts=[2, 3], model=model
        )
    )
    slope = np.ones(2) if model == "tutz" else a
    np.testing.assert_allclose(p[:, 0, 1], expit(slope[0] * (theta - 0.7)))
    q = expit(slope[1] * (theta[:, None] - steps[1]))
    expected = np.column_stack([1 - q[:, 0], q[:, 0] * (1 - q[:, 1]), q.prod(axis=1)])
    np.testing.assert_allclose(p[:, 1], expected, atol=1e-14)
    np.testing.assert_allclose(p.sum(axis=-1), 1, atol=1e-14)
    np.testing.assert_array_equal(p[:, 0, 2], 0)


@pytest.mark.parametrize("model", ["sequential", "tutz"])
def test_fit_missing_and_predict(model):
    rng = np.random.default_rng(51)
    counts = [2, 3, 4, 3]
    steps = np.array([[0, 0, 0], [-0.8, 0.2, 0], [0.5, -0.4, 0.2], [-1, 0.3, 0]])
    p = np.asarray(
        sequential_probabilities(
            rng.normal(size=500),
            a=np.ones(4),
            steps=steps,
            category_counts=counts,
            model=model,
        )
    )
    y = (rng.random((500, 4, 1)) > p.cumsum(axis=-1)).sum(axis=-1).astype(float)
    y[0] = np.nan
    y[1, 0] = np.nan
    fit = sequential_irt(y, category_counts=counts, model=model)
    assert fit.converged, fit.message
    assert np.isfinite(fit.log_likelihood)
    np.testing.assert_allclose(fit.probabilities.sum(axis=-1), 1, atol=1e-14)
    np.testing.assert_allclose(fit.means["x"][0], 0, atol=1e-14)
    np.testing.assert_allclose(fit.variances["x"][0], 1, atol=1e-14)
    if model == "tutz":
        np.testing.assert_array_equal(fit.means["a"], 1)
