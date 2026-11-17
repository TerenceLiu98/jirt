"""Four ordered logistic models, including mixed declared category counts."""

import jax
import numpy as np
from jirt import polytomous_irt, polytomous_probabilities

jax.config.update("jax_enable_x64", True)
rng = np.random.default_rng(2026)
for model in ("grm", "pcm", "gpcm", "rsm"):
    counts = np.array([3] * 5 if model == "rsm" else [2, 3, 4, 3, 4])
    steps = np.zeros((5, counts.max() - 1))
    for j, k in enumerate(counts - 1):
        steps[j, :k] = np.linspace(-1, 1, k) + j * 0.1
    theta = rng.normal(size=2000)
    p = np.asarray(
        polytomous_probabilities(
            theta, a=np.ones(5), steps=steps, category_counts=counts, model=model
        )
    )
    y = np.array(
        [
            [rng.choice(counts[j], p=p[i, j, : counts[j]]) for j in range(5)]
            for i in range(len(theta))
        ],
        dtype=float,
    )
    y[rng.random(y.shape) < 0.02] = np.nan
    fit = polytomous_irt(y, category_counts=counts, model=model, max_iter=2000)
    print(model, fit.converged, round(fit.log_likelihood, 4))
    assert fit.converged, fit.message
