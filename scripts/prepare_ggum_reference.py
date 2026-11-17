"""Independent GGUM frequency tables for matched R calibration."""

import itertools
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.hermite import hermgauss

nodes, weights = hermgauss(61)
nodes *= np.sqrt(2)
weights /= np.sqrt(np.pi)
cases = {}
for name in ("binary", "mixed"):
    counts = np.array([2] * 6 if name == "binary" else [2, 3, 4, 3, 3])
    J = len(counts)
    a = np.linspace(0.8, 1.2, J)
    delta = np.array([1.2, -1.3, 0.3, -0.5, 1.8, -2.0])[:J]
    tau = np.zeros((J, counts.max() - 1))
    patterns = np.array(list(itertools.product(*(range(k) for k in counts))))
    probability = np.ones((len(patterns), len(nodes)))
    for j, K in enumerate(counts):
        tau[j, : K - 1] = np.linspace(1.8, 0.6, K - 1)
        z = a[j] * (nodes - delta[j])
        cumulative = np.r_[0, np.cumsum(tau[j, : K - 1])]
        numerator = np.column_stack(
            [
                np.exp(k * z + a[j] * cumulative[k])
                + np.exp((2 * K - 1 - k) * z + a[j] * cumulative[k])
                for k in range(K)
            ]
        )
        p = numerator / numerator.sum(axis=1)[:, None]
        probability *= p[:, patterns[:, j]].T
    frequency = np.maximum(1, np.round(probability @ weights * 500000).astype(int))
    cases[name] = dict(
        category_counts=counts.tolist(),
        patterns=patterns.tolist(),
        counts=frequency.tolist(),
        truth=dict(a=a.tolist(), delta=delta.tolist(), tau=tau.tolist()),
        starts=dict(
            a=(a * 0.9).tolist(), delta=(delta + 0.15).tolist(), tau=tau.tolist()
        ),
    )
root = Path("outputs/ggum")
root.mkdir(parents=True, exist_ok=True)
(root / "inputs.json").write_text(
    json.dumps(
        dict(nodes=nodes.tolist(), weights=weights.tolist(), cases=cases), indent=2
    )
    + "\n"
)
