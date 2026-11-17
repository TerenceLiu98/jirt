"""Independent synthetic frequency tables for matched R polytomous calibration."""

import itertools
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit, softmax

nodes, weights = hermgauss(41)
nodes *= np.sqrt(2)
weights /= np.sqrt(np.pi)
cases = {}
for model in ("grm", "pcm", "gpcm", "rsm"):
    counts = np.array([3] * 4 if model == "rsm" else [2, 3, 4, 3])
    a = np.array([0.8, 1.1, 1.3, 0.9]) if model in ("grm", "gpcm") else np.ones(4)
    steps = np.zeros((4, counts.max() - 1))
    patterns = np.array(list(itertools.product(*(range(k) for k in counts))))
    probability = np.ones((len(patterns), len(nodes)))
    for j, k in enumerate(counts):
        b = (
            np.array([0.4, -0.4]) + j * 0.3
            if model == "rsm"
            else np.linspace(-1, 1, k - 1) + j * 0.15
        )
        if model in ("pcm", "gpcm"):
            b = b[::-1]
        steps[j, : k - 1] = b
        z = a[j] * (nodes[:, None] - b)
        p = (
            -np.diff(
                np.column_stack([np.ones(len(nodes)), expit(z), np.zeros(len(nodes))]),
                axis=1,
            )
            if model == "grm"
            else softmax(
                np.column_stack([np.zeros(len(nodes)), np.cumsum(z, axis=1)]), axis=1
            )
        )
        probability *= p[:, patterns[:, j]].T
    frequency = np.maximum(1, np.round(probability @ weights * 100000).astype(int))
    cases[model] = dict(
        patterns=patterns.tolist(),
        counts=frequency.tolist(),
        category_counts=counts.tolist(),
        truth=dict(a=a.tolist(), steps=steps.tolist()),
    )
root = Path("outputs/polytomous")
root.mkdir(parents=True, exist_ok=True)
(root / "inputs.json").write_text(
    json.dumps(
        dict(nodes=nodes.tolist(), weights=weights.tolist(), cases=cases), indent=2
    )
    + "\n"
)
