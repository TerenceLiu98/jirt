"""Diagnose two discrete skills with repeated single-skill and combined items."""

import jax
import numpy as np
from jirt import diagnostic_irt, diagnostic_probabilities, mastery_profiles

jax.config.update("jax_enable_x64", True)
rng = np.random.default_rng(80)
q = np.tile([[1, 0], [0, 1], [1, 1]], (3, 1))
profiles = mastery_profiles(2)
for model in ("dina", "dino"):
    labels = rng.choice(4, 3000, p=[0.15, 0.25, 0.25, 0.35])
    p = np.asarray(
        diagnostic_probabilities(
            profiles,
            q_matrix=q,
            slip=np.full(9, 0.1),
            guess=np.full(9, 0.15),
            model=model,
        )
    )
    y = (rng.random((len(labels), 9)) < p[labels]).astype(float)
    y[rng.random(y.shape) < 0.03] = np.nan
    fit = diagnostic_irt(y, q_matrix=q, model=model)
    assert fit.converged
    print(model, "weights", np.round(fit.weights, 3))
    print(
        "joint profile accuracy",
        round(
            float(np.mean(np.all(fit.classifications == profiles[labels], axis=1))), 3
        ),
    )
    print(
        "first three marginal mastery probabilities",
        np.round(fit.mastery_probabilities[:3], 3),
    )
