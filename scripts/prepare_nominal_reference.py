"""Independent SciPy nominal/nested frequency tables and JAX calibrations."""

import itertools
import json
from pathlib import Path
import jax
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit, softmax
from jirt.nominal import nominal_irt

jax.config.update("jax_enable_x64", True)
nodes, w = hermgauss(41)
nodes *= np.sqrt(2)
w /= np.sqrt(np.pi)
root = Path("outputs/nominal")
root.mkdir(parents=True, exist_ok=True)
cases = {}
for model in ("nominal", "2pl", "3pl", "3plu", "4pl"):
    counts = np.array([3, 4, 3, 4, 3])
    keys = np.array([1, 0, 2, 1, 0])
    slopes = np.zeros((5, 4))
    intercepts = np.zeros((5, 4))
    p = np.zeros((41, 5, 4))
    a = np.linspace(0.8, 1.6, 5)
    b = np.linspace(-1.2, 1.2, 5)
    c = np.full(5, 0.1 if model in ("3pl", "4pl") else 0)
    u = np.full(5, 0.9 if model in ("3plu", "4pl") else 1)
    for j, k in enumerate(counts):
        options = (
            list(range(k))
            if model == "nominal"
            else [x for x in range(k) if x != keys[j]]
        )
        for t, category in enumerate(options[1:]):
            slopes[j, category] = (0.8 + 0.1 * j) if t == 0 else -0.5
            intercepts[j, category] = -0.3 + 0.3 * t
        conditional = softmax(
            nodes[:, None] * slopes[j, options] + intercepts[j, options], axis=1
        )
        if model == "nominal":
            p[:, j, options] = conditional
        else:
            correct = c[j] + (u[j] - c[j]) * expit(a[j] * (nodes - b[j]))
            p[:, j, options] = (1 - correct[:, None]) * conditional
            p[:, j, keys[j]] = correct
    patterns = np.array(list(itertools.product(*(range(k) for k in counts))))
    probability = np.ones((len(patterns), 41))
    for j in range(5):
        probability *= p[:, j, patterns[:, j]].T
    frequency = np.maximum(1, np.round(probability @ w * 300000).astype(int))
    y = np.repeat(patterns, frequency, axis=0)
    fit = nominal_irt(
        y,
        category_counts=counts,
        model=model,
        answer_keys=None if model == "nominal" else keys,
        tolerance=1e-13,
        max_iter=3000,
    )
    cases[model] = dict(
        category_counts=counts.tolist(),
        answer_keys=keys.tolist(),
        patterns=patterns.tolist(),
        counts=frequency.tolist(),
        jirt={k: np.asarray(v).tolist() for k, v in fit.means.items() if k != "x"},
        log_likelihood=fit.log_likelihood,
        converged=fit.converged,
    )
    print(model, fit.converged, fit.log_likelihood, flush=True)
    (root / "inputs.json").write_text(
        json.dumps(
            dict(nodes=nodes.tolist(), weights=w.tolist(), cases=cases), indent=2
        )
        + "\n"
    )
