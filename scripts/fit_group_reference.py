"""Fit independent group/mixture reference inputs with jirt."""

import json
from pathlib import Path
import jax
import numpy as np
from jirt.groups import group_irt, mixture_irt

jax.config.update("jax_enable_x64", True)
root = Path("outputs/groups")
inputs = json.loads((root / "inputs.json").read_text())
out = {}
for name, case in inputs["cases"].items():
    patterns = np.asarray(case["patterns"])
    freq = np.asarray(case["counts"])
    options = dict(
        model=case["model"],
        category_counts=case["category_counts"],
        shared_a=case["shared_a"],
        shared_steps=case["shared_steps"],
        quadrature_points=61,
        max_iter=5000,
        tolerance=1e-13,
    )
    if case["kind"] == "group":
        data = [np.repeat(patterns, f, axis=0) for f in freq]
        y = np.concatenate(data)
        labels = np.concatenate([np.full(len(v), i) for i, v in enumerate(data)])
        fit = group_irt(
            y,
            groups=labels,
            free_distribution=True,
            starts=case["starts"][0],
            **options,
        )
    else:
        y = np.repeat(patterns, freq, axis=0)
        fit = mixture_irt(
            y,
            classes=case["classes"],
            zero_inflated=case["kind"] == "zero",
            starts=case["starts"],
            **options,
        )
    result = {
        k: np.asarray(fit.means[k]).tolist() for k in ("a", "steps", "mu", "sigma")
    }
    result.update(
        log_likelihood=fit.log_likelihood,
        converged=fit.converged,
        weights=None if fit.weights is None else fit.weights.tolist(),
        start_log_likelihoods=fit.start_log_likelihoods.tolist(),
    )
    print(name, result, flush=True)
    out[name] = result
    assert fit.converged, fit.message
    (root / "jirt.json").write_text(json.dumps(out, indent=2) + "\n")
