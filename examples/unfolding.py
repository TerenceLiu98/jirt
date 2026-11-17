"""Synthetic attitudes: agreement peaks near each statement's ideal location."""

import jax
import numpy as np
from jirt import ggum_irt, ggum_probabilities

jax.config.update("jax_enable_x64", True)
rng = np.random.default_rng(600)
# Statements span opposing positions. Category zero is strongest disagreement.
locations = np.array([1.3, -1.5, 0.4, -0.6, 2.0, -2.0])
counts = np.array([2, 3, 4, 3, 4, 2])
tau = np.zeros((6, 3))
for j, K in enumerate(counts):
    tau[j, : K - 1] = np.linspace(1.8, 0.6, K - 1)
theta = rng.normal(size=3000)
p = np.asarray(
    ggum_probabilities(
        theta, a=np.ones(6), delta=locations, tau=tau, category_counts=counts
    )
)
y = np.array(
    [
        [rng.choice(K, p=p[i, j, :K]) for j, K in enumerate(counts)]
        for i in range(len(theta))
    ],
    dtype=float,
)
y[rng.random(y.shape) < 0.03] = np.nan
# Use a substantive preliminary ordering of statements; try two location scales.
fits = []
for scale in (0.8, 1.2):
    starts = dict(
        a=np.ones(6), delta=scale * np.array([1.0, -1.0, 0.5, -0.5, 1.5, -1.5]), tau=tau
    )
    fit = ggum_irt(y, category_counts=counts, starts=starts, anchor_item=0)
    print(
        "start scale",
        scale,
        "converged",
        fit.converged,
        "log likelihood",
        round(fit.log_likelihood, 4),
    )
    assert fit.converged, fit.message
    fits.append(fit)
best = max(fits, key=lambda fit: fit.log_likelihood)
print("item locations:", np.round(best.means["delta"], 3))
print(
    "EAP/person correlation:",
    round(float(np.corrcoef(theta, best.means["x"])[0, 1]), 3),
)
# Predictions are category probabilities, not a monotone ability-success curve.
print(
    "expected agreement at theta=0:",
    np.round(np.asarray(best.predict([0]))[0] @ np.arange(4), 3),
)
