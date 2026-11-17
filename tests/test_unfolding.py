import jax
import jax.numpy as jnp
import numpy as np
from jirt.unfolding import ggum_probabilities, _log_probabilities

jax.config.update("jax_enable_x64", True)


def test_independent_curves_mixed_counts():
    theta = np.linspace(-4, 4, 31)
    a = np.array([0.8, 1.3])
    delta = np.array([-0.7, 0.5])
    tau = np.array([[1.2, np.nan, np.nan], [2.1, 1.3, 0.6]])
    p = np.asarray(
        ggum_probabilities(theta, a=a, delta=delta, tau=tau, category_counts=[2, 4])
    )
    for j, K in enumerate([2, 4]):
        z = a[j] * (theta - delta[j])
        M = 2 * K - 1
        s = np.r_[0, np.cumsum(tau[j, : K - 1])]
        numerator = np.column_stack(
            [
                np.exp(k * z + a[j] * s[k]) + np.exp((M - k) * z + a[j] * s[k])
                for k in range(K)
            ]
        )
        np.testing.assert_allclose(
            p[:, j, :K], numerator / numerator.sum(axis=1)[:, None], atol=1e-14
        )
    np.testing.assert_allclose(p.sum(axis=-1), 1, atol=1e-14)
    np.testing.assert_array_equal(p[:, 0, 2:], 0)


def test_proximity_symmetry_extremes_and_gradient():
    for K in [2, 3, 5]:
        tau = np.linspace(2, 0.5, K - 1)[None, :]
        theta = np.linspace(0, 8, 81)
        kwargs = dict(a=[1.2], delta=[0.0], tau=tau, category_counts=[K])
        p = ggum_probabilities(theta, **kwargs)
        np.testing.assert_allclose(p, ggum_probabilities(-theta, **kwargs), atol=1e-14)
        score = np.asarray(p[:, 0]) @ np.arange(K)
        assert np.all(np.diff(score) < 0)
        extreme = ggum_probabilities([-1e6, 1e6], **kwargs)
        np.testing.assert_allclose(extreme[:, 0, 0], 1)
        grad = jax.grad(
            lambda t: _log_probabilities(
                jnp.array([t]),
                jnp.array([1.2]),
                jnp.array([0.0]),
                jnp.asarray(tau),
                jnp.array([K]),
            )[0, 0, 0]
        )(0.0)
        np.testing.assert_allclose(grad, 0, atol=1e-14)


import itertools
import pytest
from numpy.polynomial.hermite import hermgauss
from jirt.unfolding import ggum_irt


def calibration_case(binary=False):
    counts = np.array([2] * 6 if binary else [2, 3, 4, 3, 3])
    J = len(counts)
    a = np.linspace(0.8, 1.2, J)
    delta = np.array([1.2, -1.3, 0.3, -0.5, 1.8, -2.0])[:J]
    tau = np.zeros((J, counts.max() - 1))
    nodes, w = hermgauss(61)
    nodes *= np.sqrt(2)
    w /= np.sqrt(np.pi)
    patterns = np.array(list(itertools.product(*(range(k) for k in counts))))
    prob = np.ones((len(patterns), len(nodes)))
    for j, K in enumerate(counts):
        tau[j, : K - 1] = np.linspace(1.8, 0.6, K - 1)
        z = a[j] * (nodes - delta[j])
        s = np.r_[0, np.cumsum(tau[j, : K - 1])]
        num = np.column_stack(
            [
                np.exp(k * z + a[j] * s[k]) + np.exp((2 * K - 1 - k) * z + a[j] * s[k])
                for k in range(K)
            ]
        )
        p = num / num.sum(axis=1)[:, None]
        prob *= p[:, patterns[:, j]].T
    freq = np.maximum(1, np.round(prob @ w * 500000).astype(int))
    return counts, dict(a=a, delta=delta, tau=tau), patterns, freq


@pytest.mark.parametrize("binary", [False, True])
def test_recovery_with_orientation_scale_alignment(binary):
    counts, truth, patterns, freq = calibration_case(binary)
    y = np.repeat(patterns, freq, axis=0).astype(float)
    y = np.vstack([y, np.full((1, len(counts)), np.nan)])
    starts = {k: v.copy() for k, v in truth.items()}
    starts["delta"] += 0.15
    starts["a"] *= 0.9
    fit = ggum_irt(y, category_counts=counts, starts=starts, tolerance=1e-13)
    assert fit.converged, fit.message
    # Ground truth is expressed on a reflected, shifted/scaled reporting metric.
    # Transform back to the fixed N(0,1), positive-anchor metric for recovery.
    scale, shift, sign = 1.7, 0.4, -1
    external_delta = shift + sign * scale * truth["delta"]
    aligned_delta = sign * (external_delta - shift) / scale
    external_a = truth["a"] / scale
    external_tau = truth["tau"] * scale
    np.testing.assert_allclose(fit.means["delta"], aligned_delta, atol=0.015, rtol=0)
    np.testing.assert_allclose(fit.means["a"], external_a * scale, atol=0.015, rtol=0)
    np.testing.assert_allclose(
        fit.means["tau"], external_tau / scale, atol=0.015, rtol=0
    )
    np.testing.assert_allclose(fit.means["x"][-1], 0, atol=1e-14)
    np.testing.assert_allclose(fit.variances["x"][-1], 1, atol=1e-14)
    np.testing.assert_allclose(fit.predict([0]).sum(axis=-1), 1, atol=1e-14)


def test_partial_missing_and_input_constraints():
    counts, truth, patterns, freq = calibration_case()
    y = patterns.astype(float)
    y[::7, 0] = np.nan
    fit = ggum_irt(y, category_counts=counts, starts=truth, max_iter=2)
    assert np.isfinite(fit.log_likelihood)
    for options in (
        {"anchor_item": -1},
        {"quadrature_points": 2},
        {"tolerance": 0},
        {"category_counts": [2] * 5},
    ):
        kwargs = dict(category_counts=counts, starts=truth)
        kwargs.update(options)
        with pytest.raises(ValueError):
            ggum_irt(y, **kwargs)
    invalid = {k: v.copy() for k, v in truth.items()}
    invalid["delta"][0] = -1
    with pytest.raises(ValueError, match="anchor"):
        ggum_irt(y, category_counts=counts, starts=invalid)
    invalid = {k: v.copy() for k, v in truth.items()}
    invalid["tau"][2, :3] = [0.5, 1, 2]
    with pytest.raises(ValueError, match="decreasing"):
        ggum_irt(y, category_counts=counts, starts=invalid)


def test_reflection_location_scale_invariance():
    counts, truth, _, _ = calibration_case()
    theta = np.linspace(-4, 4, 51)
    original = ggum_probabilities(theta, category_counts=counts, **truth)
    scale, shift = 1.7, 0.4
    transformed = dict(
        a=truth["a"] / scale,
        delta=shift - scale * truth["delta"],
        tau=scale * truth["tau"],
    )
    reflected = ggum_probabilities(
        shift - scale * theta, category_counts=counts, **transformed
    )
    np.testing.assert_allclose(original, reflected, rtol=0, atol=2e-14)


def test_partial_missing_posterior_matches_direct_integration():
    from jirt.unfolding import _posterior

    counts, truth, patterns, _ = calibration_case()
    patterns = patterns[:7].copy()
    patterns[0, :] = -1
    patterns[1::2, 1] = -1
    nodes, w = hermgauss(61)
    nodes *= np.sqrt(2)
    w /= np.sqrt(np.pi)
    mask = np.arange(counts.max() - 1) < counts[:, None] - 1
    design = np.zeros((*mask.shape, mask.sum()))
    design[mask] = np.eye(mask.sum())
    raw = np.r_[truth["a"], truth["delta"], truth["tau"][mask]]
    ll, post = _posterior(
        jnp.asarray(raw),
        jnp.asarray(patterns),
        jnp.asarray(nodes),
        jnp.log(w),
        jnp.asarray(design),
        jnp.asarray(counts),
    )
    p = np.asarray(ggum_probabilities(nodes, category_counts=counts, **truth))
    expected = np.tile(w, (len(patterns), 1))
    for i, pattern in enumerate(patterns):
        for j, k in enumerate(pattern):
            if k >= 0:
                expected[i] *= p[:, j, k]
    np.testing.assert_allclose(ll, np.log(expected.sum(axis=1)), atol=1e-13)
    np.testing.assert_allclose(
        post, expected / expected.sum(axis=1)[:, None], atol=1e-14
    )
