"""Nominal and nested distractor analysis with mixed answer keys."""

import jax
import numpy as np
from jirt import nominal_irt, nominal_probabilities, nested_probabilities

jax.config.update("jax_enable_x64", True)
rng = np.random.default_rng(2026)
counts = np.array([3, 4, 3, 4, 3])
keys = np.array([1, 0, 2, 1, 0])
theta = rng.normal(size=2500)
for model in ("nominal", "2pl", "3pl", "3plu", "4pl"):
    slopes = np.zeros((5, 4))
    intercepts = np.zeros((5, 4))
    for j, k in enumerate(counts):
        options = (
            list(range(k))
            if model == "nominal"
            else [i for i in range(k) if i != keys[j]]
        )
        for t, c in enumerate(options[1:]):
            slopes[j, c] = 0.7 if t == 0 else -0.4
            intercepts[j, c] = -0.3
    if model == "nominal":
        p = nominal_probabilities(
            theta, slopes=slopes, intercepts=intercepts, category_counts=counts
        )
    else:
        p = nested_probabilities(
            theta,
            slopes=slopes,
            intercepts=intercepts,
            category_counts=counts,
            answer_keys=keys,
            a=np.linspace(0.8, 1.6, 5),
            b=np.linspace(-1, 1, 5),
            c=0.1 if model in ("3pl", "4pl") else 0,
            u=0.9 if model in ("3plu", "4pl") else 1,
        )
    p = np.asarray(p)
    y = np.array(
        [
            [rng.choice(k, p=p[i, j, :k]) for j, k in enumerate(counts)]
            for i in range(len(theta))
        ],
        dtype=float,
    )
    y[rng.random(y.shape) < 0.02] = np.nan
    fit = nominal_irt(
        y,
        category_counts=counts,
        answer_keys=None if model == "nominal" else keys,
        model=model,
        max_iter=3000,
    )
    print(
        model,
        "converged:",
        fit.converged,
        "log likelihood:",
        round(fit.log_likelihood, 4),
    )
    print(
        "Item 1 category probabilities at theta=0:",
        np.asarray(fit.predict([0]))[0, 0, : counts[0]],
    )
    assert fit.converged, fit.message
