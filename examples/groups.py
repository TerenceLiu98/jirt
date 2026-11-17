"""Known groups, unknown classes and structural lowest-category responses."""

import jax
import numpy as np
from jirt import group_irt, group_probabilities, mixture_irt

jax.config.update("jax_enable_x64", True)
rng = np.random.default_rng(700)


def sample(theta, labels, a, b, counts, model):
    p = np.asarray(
        group_probabilities(
            theta, groups=labels, a=a, steps=b, category_counts=counts, model=model
        )
    )
    y = np.array(
        [
            [rng.choice(K, p=p[i, j, :K]) for j, K in enumerate(counts)]
            for i in range(len(theta))
        ],
        float,
    )
    y[rng.random(y.shape) < 0.02] = np.nan
    return y


# Known populations: two anchors link the scales, other items may differ.
counts = np.full(5, 3)
a = np.ones((2, 5))
b = np.tile(np.array([-0.6, 0.7]), (2, 5, 1))
b[1, 2:] += 0.4
labels = np.repeat([0, 1], 1500)
theta = rng.normal(size=len(labels)) * np.where(labels == 1, 1.15, 1) + 0.45 * labels
y = sample(theta, labels, a, b, counts, "grm")
anchors = np.arange(5) < 2
fit = group_irt(
    y,
    groups=labels,
    model="grm",
    category_counts=counts,
    starts=dict(a=a, steps=b),
    shared_a=anchors,
    shared_steps=anchors,
    free_distribution=True,
)
assert fit.converged, fit.message
print(
    "observed groups: means",
    np.round(fit.means["mu"], 3),
    "SDs",
    np.round(fit.means["sigma"], 3),
)

# Unknown classes: contrasting item difficulties distinguish response profiles.
counts = np.full(8, 2)
a = np.ones((2, 8))
b = np.zeros((2, 8, 1))
b[1, :, 0] = np.where(np.arange(8) % 2 == 0, 1.2, -1.2)
labels = rng.choice(2, size=3000, p=[0.4, 0.6])
y = sample(rng.normal(size=len(labels)), labels, a, b, counts, "2pl")
starts = [
    dict(a=a, steps=b + 0.1, weights=[0.5, 0.5]),
    dict(a=a[::-1], steps=(b - 0.2)[::-1], weights=[0.6, 0.4]),
]
fit = mixture_irt(y, classes=2, starts=starts)
assert fit.converged, fit.message
print("latent class weights (arbitrary order):", np.round(fit.weights, 3))
print("start log likelihoods:", np.round(fit.start_log_likelihoods, 3))
np.testing.assert_allclose(fit.memberships.sum(axis=1), 1, atol=1e-14)

# Structural zeros are observed outcomes, not missing responses.
counts = np.full(5, 3)
a = np.ones((1, 5))
b = np.tile([-0.6, 0.7], (1, 5, 1))
y = sample(rng.normal(size=3000), np.zeros(3000, int), a, b, counts, "grm")
structural = rng.random(len(y)) < 0.25
y[structural] = 0
fit = mixture_irt(
    y,
    classes=1,
    model="grm",
    category_counts=counts,
    zero_inflated=True,
    starts=[
        dict(a=a, steps=b, weights=[0.7, 0.3]),
        dict(a=a * 1.1, steps=b + 0.1, weights=[0.5, 0.5]),
    ],
)
assert fit.converged, fit.message
assert np.all(np.asarray(fit.memberships)[np.any(y > 0, axis=1), -1] == 0)
print("structural proportion:", round(float(fit.weights[-1]), 3))
print(
    "mean structural posterior for simulated structural subjects:",
    round(float(fit.memberships[structural, -1].mean()), 3),
)
