import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import expit

from jirt.noncompensatory import noncompensatory_probabilities, _log_responses

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("model", ["pc1pl", "pc2pl", "pc3pl"])
def test_one_dimension_reduction(model):
    theta = np.linspace(-8, 8, 25)[:, None]
    a = np.array([[0.8], [1.4]])
    d = np.array([[-0.5], [0.7]])
    c = np.array([0.1, 0.2])
    p = noncompensatory_probabilities(
        theta, a=a, d=d, c=c, loading_mask=np.ones_like(a, bool), model=model
    )
    guess = c if model == "pc3pl" else 0
    slope = a[:, 0] if model != "pc1pl" else 1
    expected = guess + (1 - guess) * expit(theta * slope + d[:, 0])
    np.testing.assert_allclose(p, expected, atol=1e-14)


def test_weak_dimension_and_excluded_dimensions():
    theta = np.array([[1000, -10], [1000, 1000]])
    mask = np.array([[True, True], [True, False]])
    a = np.ones((2, 2))
    d = np.array([[0, 0], [0, np.nan]])
    p = np.asarray(noncompensatory_probabilities(theta, a=a, d=d, loading_mask=mask))
    assert p[0, 0] < 0.00005
    np.testing.assert_allclose(p[:, 1], 1)
    np.testing.assert_allclose(p[1], 1)


def test_extreme_log_likelihood_and_gradient():
    mask = jnp.ones((2, 2), dtype=bool)
    theta = jnp.array([[-1000.0, -1000.0], [1000.0, 1000.0]])

    def objective(d):
        yes, no = _log_responses(theta, jnp.ones((2, 2)), d, jnp.zeros(2), mask)
        return jnp.sum(yes + no)

    value, gradient = jax.jit(jax.value_and_grad(objective))(jnp.zeros((2, 2)))
    assert np.isfinite(value)
    assert np.isfinite(gradient).all()


from jirt.noncompensatory import noncompensatory_irt


@pytest.mark.parametrize("model", ["pc1pl", "pc2pl", "pc3pl"])
def test_fit_missing_and_restrictions(model):
    rng = np.random.default_rng(216)
    mask = np.array([[1, 0]] * 3 + [[0, 1]] * 3 + [[1, 1]] * 2, dtype=bool)
    a = mask * rng.uniform(0.8, 1.4, mask.shape)
    d = mask * rng.normal(0, 0.5, mask.shape)
    p = noncompensatory_probabilities(
        rng.normal(size=(1000, 2)),
        a=a,
        d=d,
        c=np.full(8, 0.1),
        loading_mask=mask,
        model=model,
    )
    y = (rng.random(p.shape) < np.asarray(p)).astype(float)
    y[0] = np.nan
    y[1, 0] = np.nan
    fit = noncompensatory_irt(y, loading_mask=mask, model=model, quadrature_points=9)
    assert fit.converged, fit.message
    assert np.isfinite(fit.log_likelihood)
    assert fit.log_likelihood >= fit.trace[0]
    np.testing.assert_array_equal(fit.means["a"][~mask], 0)
    np.testing.assert_array_equal(fit.means["d"][~mask], 0)
    np.testing.assert_allclose(fit.means["x"][0], 0, atol=1e-14)
    np.testing.assert_allclose(fit.covariances["x"][0], np.eye(2), atol=1e-14)
    if model == "pc1pl":
        np.testing.assert_array_equal(fit.means["a"][mask], 1)
    if model != "pc3pl":
        np.testing.assert_array_equal(fit.means["c"], 0)
    assert np.all((fit.probabilities >= 0) & (fit.probabilities <= 1))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"loading_mask": [[True, False], [True, False]]},
        {"loading_mask": [[1], [1]]},
        {"loading_mask": [[True], [False]]},
        {"quadrature_points": 2},
        {"max_nodes": 1},
        {"starts": {"a": [[-1], [1]]}},
        {"starts": {"d": [[np.nan], [0]]}},
        {"model": "unknown"},
        {"tolerance": 0},
    ],
)
def test_invalid_inputs(kwargs):
    options = dict(loading_mask=np.ones((2, 1), bool))
    options.update(kwargs)
    with pytest.raises(ValueError):
        noncompensatory_irt([[0, 1], [1, 0]], **options)
