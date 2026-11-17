"""Independent frequency tables for PC and sequential mirt comparisons."""

import itertools
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit

cases = {}
for model in ("pc1pl", "pc2pl", "pc3pl", "sequential", "tutz"):
    pc = model.startswith("pc")
    x, w = hermgauss(15 if pc else 41)
    x, w = x * np.sqrt(2), w / np.sqrt(np.pi)
    nodes = np.array(list(itertools.product(x, repeat=2))) if pc else x[:, None]
    weights = np.array([u * v for u in w for v in w]) if pc else w
    if pc:
        mask = np.array([[1, 0]] * 3 + [[0, 1]] * 3 + [[1, 1]] * 2, bool)
        a = mask * np.array(
            [
                [0.8, 1],
                [1.1, 1],
                [1.4, 1],
                [1, 0.9],
                [1, 1.2],
                [1, 1.5],
                [0.9, 1.2],
                [1.3, 0.8],
            ]
        )
        if model == "pc1pl":
            a = mask.astype(float)
        d = mask * np.array(
            [
                [-0.7, 0],
                [0.2, 0],
                [0.8, 0],
                [0, -0.8],
                [0, 0.3],
                [0, 0.7],
                [0.3, -0.2],
                [-0.4, 0.6],
            ]
        )
        c = np.full(8, 0.12 if model == "pc3pl" else 0)
        success = c + (1 - c) * np.prod(
            np.where(mask, expit(nodes[:, None, :] * a + d), 1), axis=-1
        )
        probs = np.stack([1 - success, success], axis=-1)
        counts = np.full(8, 2)
        truth = dict(a=a.tolist(), d=d.tolist(), c=c.tolist())
    else:
        counts = np.array([2, 3, 4, 3])
        a = np.array([0.8, 1.1, 1.3, 0.9]) if model == "sequential" else np.ones(4)
        steps = np.array(
            [[0.2, 0, 0], [0.5, -0.7, 0], [0.4, -0.8, 0.2], [-0.5, 0.7, 0]]
        )
        probs = np.zeros((len(nodes), 4, 4))
        for j, k in enumerate(counts):
            q = expit(a[j] * (nodes - steps[j, : k - 1]))
            prefix = np.column_stack([np.ones(len(nodes)), np.cumprod(q, axis=1)])
            probs[:, j, :k] = prefix * np.column_stack([1 - q, np.ones(len(nodes))])
        truth = dict(a=a.tolist(), steps=steps.tolist())
    patterns = np.array(list(itertools.product(*(range(k) for k in counts))))
    marginal = (
        np.prod(probs[:, np.arange(len(counts))[None, :], patterns], axis=-1).T
        @ weights
    )
    frequency = np.maximum(1, np.round(marginal * 300000).astype(int))
    cases[model] = dict(
        nodes=nodes.tolist(),
        weights=weights.tolist(),
        patterns=patterns.tolist(),
        counts=frequency.tolist(),
        category_counts=counts.tolist(),
        truth=truth,
    )
    if pc:
        cases[model]["loading_mask"] = mask.tolist()
root = Path("outputs/response-models")
root.mkdir(parents=True, exist_ok=True)
(root / "inputs.json").write_text(json.dumps(dict(cases=cases), indent=2) + "\n")
