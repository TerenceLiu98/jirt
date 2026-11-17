"""Create deterministic response-pattern frequencies for R/JAX comparison."""

import itertools
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit

root = Path("outputs/logistic-mirt")
root.mkdir(parents=True, exist_ok=True)
nodes, weights = hermgauss(41)
nodes *= np.sqrt(2)
weights /= np.sqrt(np.pi)
patterns = np.array(list(itertools.product([0, 1], repeat=6)))
cases = {}
for model in ("rasch", "1pl", "2pl", "3pl", "4pl"):
    a = (
        np.full(6, 1.0 if model == "rasch" else 1.25)
        if model in ("rasch", "1pl")
        else np.linspace(0.8, 1.6, 6)
    )
    b = np.linspace(-1.5, 1.5, 6)
    c = 0.1 if model in ("3pl", "4pl") else 0.0
    u = 0.9 if model == "4pl" else 1.0
    p = c + (u - c) * expit(a * (nodes[:, None] - b))
    frequency = (
        np.exp(patterns @ np.log(p).T + (1 - patterns) @ np.log1p(-p).T) @ weights
    )
    counts = np.maximum(1, np.round(frequency * 10000).astype(int))
    cases[model] = dict(model=model, patterns=patterns.tolist(), counts=counts.tolist())
(root / "inputs.json").write_text(
    json.dumps(
        dict(nodes=nodes.tolist(), weights=weights.tolist(), cases=cases), indent=2
    )
    + "\n"
)
print("Prepared five full response-pattern frequency datasets.")
