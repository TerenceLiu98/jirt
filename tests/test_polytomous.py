"""Independent probability checks for the ordered logistic response families."""

import jax
import numpy as np
import pytest
from scipy.special import expit, softmax
from jirt.polytomous import polytomous_probabilities

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("model", ["grm", "pcm", "gpcm", "rsm"])
def test_binary_reduction(model):
    theta = np.array([-1e6, -2, 0, 2, 1e6])
    a = np.array([0.8, 1.3])
    steps = np.array([[-0.5], [0.7]])
    p = polytomous_probabilities(
        theta, a=a, steps=steps, category_counts=[2, 2], model=model
    )
    slope = 1 if model in ("pcm", "rsm") else a
    expected = expit(slope * (theta[:, None] - steps[:, 0]))
    np.testing.assert_allclose(p[:, :, 1], expected, atol=1e-14)
    np.testing.assert_allclose(p.sum(axis=-1), 1, atol=1e-14)


@pytest.mark.parametrize("model", ["grm", "gpcm"])
def test_mixed_counts_and_independent_formula(model):
    theta = np.linspace(-4, 4, 11)
    a = np.array([0.8, 1.2])
    steps = np.array([[-0.5, np.nan, np.nan], [-1.0, 0.2, 1.4]])
    p = np.asarray(
        jax.jit(polytomous_probabilities, static_argnames=("model",))(
            theta, a=a, steps=steps, category_counts=np.array([2, 4]), model=model
        )
    )
    np.testing.assert_allclose(p.sum(axis=-1), 1, atol=1e-14)
    np.testing.assert_array_equal(p[:, 0, 2:], 0)
    z = a[1] * (theta[:, None] - steps[1])
    if model == "grm":
        cumulative = np.column_stack(
            [np.ones(len(theta)), expit(z), np.zeros(len(theta))]
        )
        expected = -np.diff(cumulative, axis=1)
    else:
        expected = softmax(
            np.column_stack([np.zeros(len(theta)), np.cumsum(z, axis=1)]), axis=1
        )
    np.testing.assert_allclose(p[:, 1], expected, atol=1e-14)


from itertools import product
from numpy.polynomial.hermite import hermgauss
from jirt.polytomous import polytomous_irt


def calibration_data(model):
    counts = np.array([3, 3, 3, 3] if model == "rsm" else [2, 3, 4, 3])
    a = np.array([0.8, 1.1, 1.3, 0.9]) if model in ("grm", "gpcm") else np.ones(4)
    steps = np.zeros((4, counts.max() - 1))
    if model == "rsm":
        steps = np.arange(4)[:, None] * 0.3 + np.array([0.4, -0.4])[None, :]
    else:
        for j, k in enumerate(counts - 1):
            steps[j, :k] = np.linspace(-1, 1, k) + j * 0.15
            if model in ("pcm", "gpcm"):
                steps[j, :k] = steps[j, :k][::-1]
    nodes, w = hermgauss(41)
    nodes *= np.sqrt(2)
    w /= np.sqrt(np.pi)
    patterns = np.array(list(product(*(range(k) for k in counts))))
    p = np.asarray(
        polytomous_probabilities(
            nodes, a=a, steps=steps, category_counts=counts, model=model
        )
    )
    prob = np.ones((len(patterns), len(nodes)))
    for j in range(4):
        prob *= p[:, j, patterns[:, j]].T
    freq = np.maximum(1, np.round((prob @ w) * 100000).astype(int))
    return np.repeat(patterns, freq, axis=0), counts, a, steps


@pytest.mark.parametrize("model", ["grm", "pcm", "gpcm", "rsm"])
def test_parameter_recovery(model):
    y, counts, a, steps = calibration_data(model)
    fit = polytomous_irt(
        y, category_counts=counts, model=model, tolerance=1e-13, max_iter=2000
    )
    assert fit.converged, fit.message
    np.testing.assert_allclose(fit.means["a"], a, atol=0.01, rtol=0)
    mask = np.arange(steps.shape[1])[None, :] < counts[:, None] - 1
    np.testing.assert_allclose(
        np.asarray(fit.means["steps"])[mask], steps[mask], atol=0.01, rtol=0
    )
    if model == "grm":
        for j, k in enumerate(counts):
            assert np.all(np.diff(fit.means["steps"][j, : k - 1]) > 0)
    if model == "rsm":
        assert abs(float(fit.means["tau"].sum())) < 1e-12
        np.testing.assert_allclose(
            fit.means["steps"], fit.means["b"][:, None] + fit.means["tau"]
        )


@pytest.mark.parametrize("model", ["grm", "pcm", "gpcm", "rsm"])
def test_missing_and_unobserved_categories(model):
    y = np.array([[0, 0], [2, 2], [0, 2], [2, 0], [np.nan, np.nan], [np.nan, 2]])
    fit = polytomous_irt(y, category_counts=[3, 3], model=model, max_iter=3)
    assert np.isfinite(fit.log_likelihood)
    assert fit.predict().shape == (6, 2, 3)
    np.testing.assert_allclose(fit.predict().sum(axis=-1), 1, atol=1e-13)
    assert abs(float(fit.means["x"][4])) < 1e-13
    assert abs(float(fit.variances["x"][4]) - 1) < 1e-13


def test_pcm_and_rsm_restrictions():
    theta = np.linspace(-4, 4, 13)
    steps = np.array([[0.8, -0.2], [1.3, 0.3]])
    for model in ("pcm", "rsm"):
        p = polytomous_probabilities(
            theta, a=[1, 1], steps=steps, category_counts=[3, 3], model=model
        )
        q = polytomous_probabilities(
            theta, a=[1, 1], steps=steps, category_counts=[3, 3], model="gpcm"
        )
        np.testing.assert_array_equal(p, q)


def test_invalid_data_and_ordering():
    y = np.array([[0, 1], [1, 2], [2, 0]])
    for counts in ([2, 3], [3.0, 3.0], [3], [1, 3]):
        with pytest.raises(ValueError):
            polytomous_irt(y, category_counts=counts)
    with pytest.raises(ValueError, match="increasing"):
        polytomous_irt(y, category_counts=[3, 3], starts={"steps": [[1, -1], [0, 1]]})
    with pytest.raises(ValueError, match="equal category"):
        polytomous_irt(y, category_counts=[3, 4], model="rsm")
    with pytest.raises(ValueError, match="shared"):
        polytomous_irt(
            y, category_counts=[3, 3], model="rsm", starts={"steps": [[1, -1], [0, 1]]}
        )


def test_jit_eager_agreement():
    y = np.random.default_rng(4).integers(0, 3, (40, 3))
    opts = dict(category_counts=[3] * 3, max_iter=3)
    a = polytomous_irt(y, **opts)
    b = polytomous_irt(y, **opts, jit=False)
    assert not a.converged
    np.testing.assert_allclose(a.means["steps"], b.means["steps"], atol=1e-12)
