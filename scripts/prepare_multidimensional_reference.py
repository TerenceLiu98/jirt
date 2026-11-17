"""Generate deterministic MIRT frequency tables and JAX fits for R validation."""

import itertools
import json
from pathlib import Path

import jax
import numpy as np
from scipy.special import expit
from jirt import multidimensional_irt
from jirt.multidimensional import _quadrature

jax.config.update("jax_enable_x64", True)
root = Path("outputs/multidimensional")
root.mkdir(parents=True, exist_ok=True)
cases = {}
for name in ("confirmatory", "exploratory", "bifactor", "testlet"):
    if name in ("confirmatory", "exploratory"):
        a = np.array([[1.0, 0], [0.8, 0], [1.3, 0], [0, 1.1], [0, 0.9], [0, 1.4]])
        if name == "exploratory":
            a = np.array(
                [[1.0, 0], [0.3, 1.2], [1.3, 0.2], [-0.2, 1.1], [0.8, 0.5], [0.4, 1.4]]
            )
        kwargs = dict(dimensions=2)
        if name == "confirmatory":
            kwargs["loading_mask"] = a != 0
    else:
        a = np.array(
            [
                [1, 0.7, 0],
                [0.8, 0.7, 0],
                [1.3, 0.7, 0],
                [1.1, 0, 0.6],
                [0.9, 0, 0.6],
                [1.4, 0, 0.6],
                [0.9, 0, 0],
                [1.2, 0, 0],
                [1.5, 0, 0],
            ]
        )
        if name == "bifactor":
            a[:3, 1] = [0.6, 0.8, 1.0]
            a[3:6, 2] = [0.8, 0.5, 0.9]
        kwargs = dict(structure=name, groups=[0, 0, 0, 1, 1, 1, -1, -1, -1])
    d = np.linspace(-1, 1, len(a))
    nodes, lw = _quadrature(a.shape[1], 15, 100000)
    patterns = np.array(list(itertools.product([0, 1], repeat=len(a))))
    p = expit(nodes @ a.T + d)
    freq = np.exp(patterns @ np.log(p).T + (1 - patterns) @ np.log1p(-p).T) @ np.exp(lw)
    counts = np.maximum(1, np.round(freq * 200000).astype(int))
    fit = multidimensional_irt(
        np.repeat(patterns, counts, axis=0), **kwargs, tolerance=1e-13, max_iter=3000
    )
    theta = np.random.default_rng(42).normal(size=(31, a.shape[1]))
    cases[name] = dict(
        patterns=patterns.tolist(),
        counts=counts.tolist(),
        nodes=nodes.tolist(),
        weights=np.exp(lw).tolist(),
        mask=fit.loading_mask.tolist(),
        truth=dict(a=a.tolist(), d=d.tolist()),
        jirt=dict(
            a=np.asarray(fit.means["a"]).tolist(),
            d=np.asarray(fit.means["d"]).tolist(),
            log_likelihood=fit.log_likelihood,
            converged=fit.converged,
        ),
        theta=theta.tolist(),
    )
    print(name, fit.converged, fit.log_likelihood, flush=True)
    (root / "inputs.json").write_text(json.dumps(cases, indent=2) + "\n")
