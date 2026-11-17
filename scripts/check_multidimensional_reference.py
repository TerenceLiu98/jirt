"""Audit R MIRT exports and write a compact, reviewable regression fixture."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jirt import multidimensional_probabilities
from jirt.multidimensional import _posterior

jax.config.update("jax_enable_x64", True)
root = Path("outputs/multidimensional")
inputs = json.loads((root / "inputs.json").read_text())
reference = json.loads((root / "reference.json").read_text())
report = {}
fixtures = {}
for name, case in inputs.items():
    r = reference["cases"][name]
    a = np.asarray(r["a"])
    d = np.asarray(r["d"])
    assert case["jirt"]["converged"] and r["converged"]
    predictions = np.asarray(multidimensional_probabilities(case["theta"], a=a, d=d))
    fixed_error = float(np.max(np.abs(predictions - r["probabilities"])))
    ja = case["jirt"]
    jp = np.asarray(multidimensional_probabilities(case["theta"], a=ja["a"], d=ja["d"]))
    mask = np.asarray(case["mask"])
    design = np.zeros((*a.shape, mask.sum()))
    for k, (j, f) in enumerate(zip(*np.where(mask), strict=True)):
        design[j, f, k] = 1
    raw = np.r_[d, a[mask]]
    y = np.asarray(case["patterns"])
    nodes = np.asarray(case["nodes"])
    ll, post = _posterior(
        jnp.asarray(raw),
        jnp.asarray(y == 1),
        jnp.asarray(y == 0),
        jnp.asarray(nodes),
        jnp.log(jnp.asarray(case["weights"])),
        jnp.asarray(design),
    )
    scores = np.asarray(post) @ nodes
    variances = np.sum(
        np.asarray(post)[:, :, None] * (nodes[None, :, :] - scores[:, None, :]) ** 2,
        axis=1,
    )
    errors = dict(
        fixed_probability=fixed_error,
        fixed_likelihood=abs(
            float(np.asarray(ll) @ case["counts"]) - r["log_likelihood"]
        ),
        fixed_scores=float(np.max(np.abs(scores - r["scores"]))),
        fixed_variances=float(np.max(np.abs(variances - r["variances"]))),
        fitted_a=float(np.max(np.abs(a - ja["a"]))),
        fitted_d=float(np.max(np.abs(d - ja["d"]))),
        fitted_probability=float(np.max(np.abs(jp - r["probabilities"]))),
        fitted_likelihood=abs(ja["log_likelihood"] - r["log_likelihood"]),
    )
    if "bfactor_at_jirt" in r:
        errors["bfactor_probability"] = float(np.max(np.abs(jp - r["bfactor_at_jirt"])))
    print(name, errors, flush=True)
    assert fixed_error < 1e-12 and errors["fixed_likelihood"] < 1e-6
    assert errors["fixed_scores"] < 1e-10 and errors["fixed_variances"] < 1e-10
    if "bfactor_probability" in errors:
        assert errors["bfactor_probability"] < 1e-12
    report[name] = errors
    fixtures[name] = {k: case[k] for k in ("patterns", "counts", "mask", "theta")}
    fixtures[name]["reference"] = r
(root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
fixture = dict(
    mirt_version=reference["mirt_version"],
    r_version=reference["r_version"],
    quadrature_points=15,
    cases=fixtures,
)
(root / "fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
