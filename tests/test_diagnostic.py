import jax
import numpy as np
import pytest
from jirt.diagnostic import diagnostic_irt, diagnostic_probabilities, mastery_profiles

jax.config.update("jax_enable_x64", True)


def test_enumerated_rules():
    profiles = mastery_profiles(2)
    q = np.array([[1, 0], [0, 1], [1, 1]])
    dina = diagnostic_probabilities(
        profiles, q_matrix=q, slip=[0.1] * 3, guess=[0.2] * 3
    )
    dino = diagnostic_probabilities(
        profiles, q_matrix=q, slip=[0.1] * 3, guess=[0.2] * 3, model="dino"
    )
    np.testing.assert_allclose(
        dina, [[0.2, 0.2, 0.2], [0.2, 0.9, 0.2], [0.9, 0.2, 0.2], [0.9, 0.9, 0.9]]
    )
    np.testing.assert_allclose(
        dino, [[0.2, 0.2, 0.2], [0.2, 0.9, 0.9], [0.9, 0.2, 0.9], [0.9, 0.9, 0.9]]
    )
    np.testing.assert_array_equal(dina[:, :2], dino[:, :2])


@pytest.mark.parametrize("model", ["dina", "dino"])
def test_simulation_recovery_missing_classification(model):
    rng = np.random.default_rng(80)
    q = np.tile([[1, 0], [0, 1], [1, 1]], (3, 1))
    profiles = mastery_profiles(2)
    weights = np.array([0.15, 0.25, 0.25, 0.35])
    labels = rng.choice(4, 10000, p=weights)
    slip = np.linspace(0.06, 0.14, 9)
    guess = np.linspace(0.1, 0.18, 9)
    p = np.asarray(
        diagnostic_probabilities(
            profiles, q_matrix=q, slip=slip, guess=guess, model=model
        )
    )
    y = (rng.random((len(labels), 9)) < p[labels]).astype(float)
    y[rng.random(y.shape) < 0.03] = np.nan
    y[0] = np.nan
    fit = diagnostic_irt(y, q_matrix=q, model=model, tolerance=1e-11)
    assert fit.converged
    assert np.min(np.diff(fit.trace)) > -1e-7
    np.testing.assert_allclose(fit.slip, slip, atol=0.025, rtol=0)
    np.testing.assert_allclose(fit.guess, guess, atol=0.025, rtol=0)
    np.testing.assert_allclose(fit.weights, weights, atol=0.02, rtol=0)
    np.testing.assert_allclose(fit.profile_probabilities.sum(axis=1), 1, atol=1e-14)
    np.testing.assert_allclose(fit.profile_probabilities[0], fit.weights, atol=1e-14)
    assert (
        np.mean(np.all(fit.classifications[1:] == profiles[labels[1:]], axis=1)) > 0.85
    )
    assert fit.predict().shape == y.shape


@pytest.mark.parametrize(
    "q", [[[0, 0], [1, 1]], [[1, 0], [1, 0]], [[1, 2], [1, 0]], [[1], [1], [1]]]
)
def test_invalid_q(q):
    with pytest.raises(ValueError):
        diagnostic_irt([[0, 1], [1, 0]], q_matrix=q)


def test_profile_limit():
    with pytest.raises(ValueError, match="max_profiles"):
        mastery_profiles(20, max_profiles=1024)


def test_missing_likelihood_and_pooled_boundary():
    # Equal empirical success rates for every profile force uninformative items.
    y = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (20, 1)).astype(float)
    y = np.vstack([y, [np.nan, np.nan], [np.nan, 1]])
    fit = diagnostic_irt(
        y,
        q_matrix=[[1], [1]],
        starts=dict(slip=[0.5, 0.5], guess=[0.5, 0.5]),
        tolerance=1e-13,
    )
    assert fit.converged
    np.testing.assert_allclose(fit.guess, [0.5, 41 / 81], atol=1e-12)
    np.testing.assert_allclose(1 - fit.slip, fit.guess, atol=1e-12)
    np.testing.assert_allclose(fit.profile_probabilities, 0.5, atol=1e-12)
    assert np.isfinite(fit.log_likelihood)


@pytest.mark.parametrize(
    "options",
    [
        dict(starts=dict(slip=[0.9, 0.9], guess=[0.2, 0.2])),
        dict(starts=dict(weights=[0.0, 1.0])),
        dict(max_iter=0),
        dict(tolerance=0),
        dict(model="unknown"),
        dict(jit=1),
    ],
)
def test_invalid_fit_inputs(options):
    with pytest.raises(ValueError):
        diagnostic_irt([[0, 1], [1, 0]], q_matrix=[[1], [1]], **options)


def test_jit_eager_equivalence():
    y = np.tile([[0, 0], [0, 1], [1, 1]], (10, 1))
    a = diagnostic_irt(y, q_matrix=[[1], [1]], max_iter=5)
    b = diagnostic_irt(y, q_matrix=[[1], [1]], max_iter=5, jit=False)
    np.testing.assert_allclose(a.slip, b.slip, atol=1e-14)
    np.testing.assert_allclose(a.trace, b.trace, atol=1e-12)
