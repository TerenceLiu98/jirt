"""Independent binary/graded pattern frequencies for group and mixture checks."""

import itertools
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit

nodes, w = hermgauss(61)
nodes *= np.sqrt(2)
w /= np.sqrt(np.pi)
cases = {}
for kind in ("group", "mixture", "zero"):
    for model in ("2pl", "grm"):
        J = 8 if model == "2pl" else 5
        K = np.full(J, 2 if model == "2pl" else 3)
        C = 1 if kind == "zero" else 2
        a = np.tile(np.linspace(0.8, 1.3, J), (C, 1))
        b = np.zeros((C, J, K.max() - 1))
        for j in range(J):
            b[0, j] = np.linspace(-0.7, 0.7, K[j] - 1) + 0.15 * (j - J / 2)
        if C == 2:
            b[1] = b[0]
            if kind == "group":
                b[1, 2:] += 0.4
                a[1, 2:] *= 1.15
            else:
                b[1] += np.where(np.arange(J) % 2 == 0, 1.2, -1.2)[:, None]
                a[1] *= 1.1
        mu = np.array([0.0, 0.45]) if kind == "group" else np.zeros(C)
        sigma = np.array([1.0, 1.15]) if kind == "group" else np.ones(C)
        weights = np.array([0.7, 0.3]) if kind == "zero" else np.array([0.4, 0.6])
        patterns = np.array(list(itertools.product(*(range(k) for k in K))))
        marginal = []
        for c in range(C):
            theta = mu[c] + sigma[c] * nodes
            prob = np.ones((len(patterns), len(nodes)))
            for j, k in enumerate(K):
                cumulative = expit(a[c, j] * (theta[:, None] - b[c, j]))
                p = -np.diff(
                    np.column_stack(
                        [np.ones(len(nodes)), cumulative, np.zeros(len(nodes))]
                    ),
                    axis=1,
                )
                prob *= p[:, patterns[:, j]].T
            marginal.append(prob @ w)
        marginal = np.asarray(marginal)
        if kind == "zero":
            marginal = np.vstack([marginal, np.all(patterns == 0, axis=1)])
        frequencies = np.maximum(
            1,
            np.round(
                (marginal if kind == "group" else weights @ marginal) * 300000
            ).astype(int),
        )
        share = np.arange(J) < 2 if kind == "group" else np.zeros(J, bool)
        starts = []
        for attempt in range(2 if kind != "group" else 1):
            aa = a * (0.95 if attempt == 0 else 1.1)
            bb = b + (0.1 if attempt == 0 else -0.2)
            start = dict(
                a=aa.tolist(),
                steps=bb.tolist(),
                mu=np.zeros(C).tolist(),
                sigma=np.ones(C).tolist(),
            )
            if kind != "group":
                start["weights"] = [0.5, 0.5] if attempt == 0 else [0.6, 0.4]
            starts.append(start)
        cases[kind + "_" + model] = dict(
            kind=kind,
            model=model,
            classes=C,
            category_counts=K.tolist(),
            patterns=patterns.tolist(),
            counts=frequencies.tolist(),
            shared_a=share.tolist(),
            shared_steps=share.tolist(),
            truth=dict(
                a=a.tolist(),
                steps=b.tolist(),
                mu=mu.tolist(),
                sigma=sigma.tolist(),
                weights=weights.tolist(),
            ),
            starts=starts,
        )
root = Path("outputs/groups")
root.mkdir(parents=True, exist_ok=True)
(root / "inputs.json").write_text(
    json.dumps(dict(nodes=nodes.tolist(), weights=w.tolist(), cases=cases), indent=2)
    + "\n"
)
