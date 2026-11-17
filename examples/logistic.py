"""Reproducible logistic calibration; run with uv run python examples/logistic.py."""

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
from jirt import logistic_irt, logistic_probabilities

rng = np.random.default_rng(2026)
abilities = rng.normal(size=1000)
probabilities = np.asarray(
    logistic_probabilities(
        abilities,
        a=np.linspace(0.8, 1.8, 10),
        b=np.linspace(-2, 2, 10),
        c=0.1,
        u=0.95,
    )
)
responses = (rng.random(probabilities.shape) < probabilities).astype(float)
responses[rng.random(responses.shape) < 0.05] = np.nan

for model in ("rasch", "1pl", "2pl", "3pl", "4pl"):
    fit = logistic_irt(responses, model=model)
    print(
        model,
        "converged:",
        fit.converged,
        "iterations:",
        fit.iterations,
        "log likelihood:",
        round(fit.log_likelihood, 3),
    )
    print(
        "Predictions at theta = -1, 0, 1:",
        np.asarray(fit.predict([-1.0, 0.0, 1.0]))[:, :3],
    )
