import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit
from jirt.groups import (
    group_probabilities,
    mixture_probabilities,
    _component_posterior,
    _mix_posterior,
)

jax.config.update("jax_enable_x64", True)


def test_group_binary_and_mixture_zero_weight_limits():
    a = np.array([[1.0, 1.3], [0.7, 1.1]])
    b = np.array([[[-0.5], [0.3]], [[0.2], [-0.7]]])
    theta = np.array([-2.0, 0.0, 2.0])
    groups = np.array([0, 1, 0])
    p = group_probabilities(theta, groups=groups, a=a, steps=b, category_counts=[2, 2])
    np.testing.assert_allclose(
        p[:, :, 1], expit(a[groups] * (theta[:, None] - b[groups, :, 0])), atol=1e-14
    )
    for weights in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]):
        p = mixture_probabilities(
            theta,
            a=a,
            steps=b,
            category_counts=[2, 2],
            weights=weights,
            zero_inflated=True,
        )
        np.testing.assert_allclose(p.sum(axis=-1), 1, atol=1e-14)
        if weights[-1]:
            np.testing.assert_allclose(p[:, :, 0], 1)
        else:
            c = int(weights[1])
            expected = group_probabilities(
                theta, groups=np.full(3, c), a=a, steps=b, category_counts=[2, 2]
            )
            np.testing.assert_allclose(p, expected, atol=1e-14)


def test_structural_zero_and_missing_pattern_posterior():
    patterns = jnp.array([[0, 0], [0, -1], [1, 0], [-1, -1]])
    nodes, w = hermgauss(31)
    nodes *= np.sqrt(2)
    w /= np.sqrt(np.pi)
    a = jnp.ones((1, 2))
    b = jnp.zeros((1, 2, 1))
    ll, post, theta = _component_posterior(
        patterns,
        jnp.asarray(nodes),
        jnp.log(w),
        a,
        b,
        jnp.zeros(1),
        jnp.ones(1),
        jnp.array([2, 2]),
        "2pl",
    )
    mixed, membership = _mix_posterior(ll, patterns, jnp.array([0.7, 0.3]), True)
    expected = np.array(
        [
            np.sum(w * expit(-nodes) ** 2),
            0.5,
            np.sum(w * expit(nodes) * expit(-nodes)),
            1.0,
        ]
    )
    structural = np.array([1, 1, 0, 1])
    np.testing.assert_allclose(
        np.exp(mixed), 0.7 * expected + 0.3 * structural, atol=1e-14
    )
    np.testing.assert_allclose(
        membership[:, 1],
        0.3 * structural / (0.7 * expected + 0.3 * structural),
        atol=1e-14,
    )
    np.testing.assert_allclose(membership.sum(axis=1), 1, atol=1e-14)
    np.testing.assert_allclose(post[-1, 0], w, atol=1e-14)


import pytest
from jirt.groups import group_irt, mixture_irt
from jirt.polytomous import polytomous_irt


@pytest.mark.parametrize("model", ["rasch", "2pl", "grm"])
def test_single_group_class_reduction(model):
    rng = np.random.default_rng(79)
    counts = np.array([2, 3, 3, 2] if model == "grm" else [2] * 4)
    b = np.zeros((1, 4, counts.max() - 1))
    for j, K in enumerate(counts):
        b[0, j, : K - 1] = np.linspace(-0.7, 0.7, K - 1)
    a = np.ones((1, 4))
    p = np.asarray(
        group_probabilities(
            rng.normal(size=500),
            groups=np.zeros(500, int),
            a=a,
            steps=b,
            category_counts=counts,
            model=model,
        )
    )
    y = np.array(
        [
            [rng.choice(K, p=p[i, j, :K]) for j, K in enumerate(counts)]
            for i in range(len(p))
        ],
        float,
    )
    y[0] = np.nan
    y[1, 0] = np.nan
    options = dict(
        category_counts=counts, model=model, starts=dict(a=a, steps=b), tolerance=1e-12
    )
    group = group_irt(y, groups=np.zeros(len(y), int), **options)
    mixture = mixture_irt(y, classes=1, **options)
    base = polytomous_irt(
        y,
        category_counts=counts,
        model="pcm" if model == "rasch" else "grm",
        starts=dict(steps=b[0], **({} if model == "rasch" else dict(a=a[0]))),
        tolerance=1e-12,
    )
    assert group.converged and mixture.converged and base.converged
    np.testing.assert_allclose(
        group.log_likelihood, base.log_likelihood, atol=1e-6, rtol=0
    )
    np.testing.assert_allclose(
        mixture.log_likelihood, base.log_likelihood, atol=1e-6, rtol=0
    )
    np.testing.assert_allclose(group.means["a"][0], base.means["a"], atol=1e-4, rtol=0)
    np.testing.assert_allclose(mixture.means["x"], base.means["x"], atol=1e-4, rtol=0)
    np.testing.assert_allclose(mixture.memberships, 1)
    np.testing.assert_allclose(group.means["x"][0], 0, atol=1e-14)
    np.testing.assert_allclose(group.variances["x"][0], 1, atol=1e-14)


def test_graded_class_permutation_and_structural_normalization():
    a = jnp.array([[0.8, 1.2], [1.1, 0.7]])
    b = jnp.array([[[-0.5, 0.6], [0.2, 0]], [[-1.0, 0.3], [-0.7, 0]]])
    counts = jnp.array([3, 2])
    theta = jnp.array([-2.0, 0.0, 2.0])
    weights = jnp.array([0.2, 0.5, 0.3])
    p = mixture_probabilities(
        theta,
        a=a,
        steps=b,
        category_counts=counts,
        weights=weights,
        model="grm",
        zero_inflated=True,
    )
    perm = mixture_probabilities(
        theta,
        a=a[::-1],
        steps=b[::-1],
        category_counts=counts,
        weights=weights[jnp.array([1, 0, 2])],
        model="grm",
        zero_inflated=True,
    )
    np.testing.assert_allclose(p, perm, atol=1e-14)
    np.testing.assert_allclose(p.sum(axis=-1), 1, atol=1e-14)
    np.testing.assert_array_equal(p[:, 1, 2], 0)
    nodes, w = hermgauss(31)
    nodes = jnp.asarray(nodes * np.sqrt(2))
    lw = jnp.log(w / np.sqrt(np.pi))
    patterns = jnp.array([[0, 0], [1, 1], [2, -1], [-1, -1]])
    mu = jnp.array([0.0, 0.6])
    sigma = jnp.array([1.0, 1.2])
    cll, _, _ = _component_posterior(
        patterns, nodes, lw, a, b, mu, sigma, counts, "grm"
    )
    ll, post = _mix_posterior(cll, patterns, weights, True)
    cll2, _, _ = _component_posterior(
        patterns, nodes, lw, a[::-1], b[::-1], mu[::-1], sigma[::-1], counts, "grm"
    )
    ll2, post2 = _mix_posterior(cll2, patterns, weights[jnp.array([1, 0, 2])], True)
    np.testing.assert_allclose(ll, ll2, atol=1e-14)
    np.testing.assert_allclose(post, post2[:, jnp.array([1, 0, 2])], atol=1e-14)
    np.testing.assert_allclose(post[-1], weights, atol=1e-14)


@pytest.mark.parametrize(
    "option",
    [
        dict(classes=0),
        dict(classes=True),
        dict(quadrature_points=2),
        dict(tolerance=0),
        dict(free_distribution=True),
        dict(shared_a=[1, 1]),
        dict(starts=[]),
        dict(starts=dict(steps=np.zeros((2, 2, 1)), weights=[1.0, 0.0])),
        dict(starts=dict(steps=np.zeros((2, 2, 1)), mu=[1.0, 0.0])),
    ],
)
def test_invalid_mixture_specification(option):
    options = dict(classes=2, starts=dict(steps=np.zeros((2, 2, 1))))
    options.update(option)
    with pytest.raises(ValueError):
        mixture_irt([[0, 1], [1, 0]], **options)


def test_invalid_group_labels_and_shared_starts():
    with pytest.raises(ValueError, match="contiguous"):
        group_irt(
            [[0, 1], [1, 0]], groups=[0, 2], starts=dict(steps=np.zeros((2, 2, 1)))
        )
    with pytest.raises(ValueError, match="equal starts"):
        group_irt(
            [[0, 1], [1, 0]],
            groups=[0, 1],
            shared_steps=[True, True],
            starts=dict(steps=np.array([[[0.0], [0.0]], [[1.0], [1.0]]])),
        )


def test_mixture_free_distribution_and_all_missing_scores():
    rng = np.random.default_rng(413)
    J = 8
    labels = rng.choice(2, size=3000, p=[0.5, 0.5])
    theta = rng.normal(size=len(labels)) + 1.6 * labels
    a = np.ones((2, J))
    b = np.broadcast_to(np.linspace(-1, 1, J)[None, :, None], (2, J, 1)).copy()
    p = np.asarray(
        group_probabilities(
            theta, groups=labels, a=a, steps=b, category_counts=[2] * J, model="rasch"
        )
    )
    y = (rng.random((len(labels), J)) < p[:, :, 1]).astype(float)
    y = np.vstack([y, np.full((1, J), np.nan)])
    fit = mixture_irt(
        y,
        classes=2,
        model="rasch",
        shared_steps=np.ones(J, bool),
        free_distribution=True,
        starts=dict(a=a, steps=b, mu=[0.0, 1.2], sigma=[1.0, 1.0], weights=[0.5, 0.5]),
        quadrature_points=31,
        tolerance=1e-9,
    )
    assert fit.converged, fit.message
    assert fit.means["mu"][0] == 0 and fit.means["sigma"][0] == 1
    assert abs(float(fit.means["mu"][1])) > 0.2
    np.testing.assert_allclose(fit.memberships[-1], fit.weights, atol=1e-14)
    np.testing.assert_allclose(
        fit.means["x"][-1], fit.weights @ np.asarray(fit.means["mu"]), atol=1e-12
    )
    np.testing.assert_allclose(fit.means["steps"][0], fit.means["steps"][1], atol=0)
