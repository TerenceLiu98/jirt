"""Fit confirmatory MIRT, bifactor and testlet models to simulated responses."""

import jax
import numpy as np
from jirt import multidimensional_irt, multidimensional_probabilities

jax.config.update("jax_enable_x64", True)
rng = np.random.default_rng(2026)
groups = np.array([0, 0, 0, 1, 1, 1, -1, -1, -1])
a = np.array(
    [
        [1, 0.7, 0],
        [0.8, 0.7, 0],
        [1.3, 0.7, 0],
        [1.1, 0, 0.6],
        [0.9, 0, 0.6],
        [1.4, 0, 0.6],
        [0.9, 0, 0],
        [1.2, 0, 0],
        [1.5, 0, 0],
    ]
)
d = np.linspace(-1, 1, len(a))
theta = rng.normal(size=(3000, 3))
y = rng.binomial(1, np.asarray(multidimensional_probabilities(theta, a=a, d=d))).astype(
    float
)
y[rng.random(y.shape) < 0.02] = np.nan
for structure in ("multidimensional", "bifactor", "testlet"):
    options = (
        dict(dimensions=3, loading_mask=a != 0)
        if structure == "multidimensional"
        else dict(structure=structure, groups=groups)
    )
    fit = multidimensional_irt(y, **options, max_iter=2000, quadrature_points=15)
    print(
        structure,
        "converged:",
        fit.converged,
        "log likelihood:",
        round(fit.log_likelihood, 4),
    )
    print("First subject EAP:", np.asarray(fit.means["x"][0]))
    if structure == "testlet":
        print("Testlet variances:", np.asarray(fit.means["testlet_variance"]))
    assert fit.converged, fit.message
