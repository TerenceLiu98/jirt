"""Products across abilities and successive ordered stage decisions."""

import jax
import numpy as np
from jirt import (
    noncompensatory_irt,
    noncompensatory_probabilities,
    sequential_irt,
    sequential_probabilities,
)

jax.config.update("jax_enable_x64", True)
rng = np.random.default_rng(216)
mask = np.array([[1, 0]] * 3 + [[0, 1]] * 3 + [[1, 1]] * 2, dtype=bool)
for model in ("pc1pl", "pc2pl", "pc3pl"):
    p = np.asarray(
        noncompensatory_probabilities(
            rng.normal(size=(2000, 2)),
            a=mask.astype(float),
            d=np.zeros(mask.shape),
            c=np.full(8, 0.1),
            loading_mask=mask,
            model=model,
        )
    )
    y = (rng.random(p.shape) < p).astype(float)
    y[rng.random(y.shape) < 0.02] = np.nan
    fit = noncompensatory_irt(y, loading_mask=mask, model=model, max_iter=3000)
    print(model, fit.converged, round(fit.log_likelihood, 4))
    assert fit.converged, fit.message
for model in ("sequential", "tutz"):
    counts = np.array([2, 3, 4, 3])
    steps = np.array([[0.2, 0, 0], [0.5, -0.7, 0], [0.4, -0.8, 0.2], [-0.5, 0.7, 0]])
    p = np.asarray(
        sequential_probabilities(
            rng.normal(size=2000),
            a=np.ones(4),
            steps=steps,
            category_counts=counts,
            model=model,
        )
    )
    y = (rng.random((2000, 4, 1)) > p.cumsum(axis=-1)).sum(axis=-1).astype(float)
    y[rng.random(y.shape) < 0.02] = np.nan
    fit = sequential_irt(y, category_counts=counts, model=model)
    print(model, fit.converged, round(fit.log_likelihood, 4))
    assert fit.converged, fit.message
