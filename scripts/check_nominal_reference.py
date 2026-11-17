"""Compare matched nominal/nested fits and prepare a regression fixture."""

import json
from pathlib import Path
import jax
import numpy as np
from jirt.nominal import nominal_probabilities, nested_probabilities

jax.config.update("jax_enable_x64", True)
root = Path("outputs/nominal")
inp = json.loads((root / "inputs.json").read_text())
ref = json.loads((root / "reference.json").read_text())
assert set(ref["cases"]) == set(inp["cases"])
report = {}
fixture = dict(mirt_version=ref["mirt_version"], r_version=ref["r_version"], cases={})
for model, case in inp["cases"].items():
    r = ref["cases"][model]
    kwargs = dict(category_counts=case["category_counts"])
    if model != "nominal":
        kwargs["answer_keys"] = case["answer_keys"]
    fn = nominal_probabilities if model == "nominal" else nested_probabilities
    fixed = np.asarray(fn(r["theta"], **kwargs, **r["means"]))
    fitted = np.asarray(fn(r["theta"], **kwargs, **case["jirt"]))
    errors = dict(
        parameters=max(
            float(np.max(abs(np.asarray(r["means"][k]) - case["jirt"][k])))
            for k in r["means"]
        ),
        log_likelihood=abs(r["log_likelihood"] - case["log_likelihood"]),
        fixed_probability=max(
            float(np.max(abs(fixed[:, j, :k] - r["probabilities"][j])))
            for j, k in enumerate(case["category_counts"])
        ),
        fitted_probability=max(
            float(np.max(abs(fitted[:, j, :k] - r["probabilities"][j])))
            for j, k in enumerate(case["category_counts"])
        ),
        mirt_converged=r["converged"],
        jirt_converged=case["converged"],
    )
    print(model, errors, flush=True)
    assert errors["fixed_probability"] < 1e-12
    report[model] = errors
    fixture["cases"][model] = dict(**case, reference=r)
(root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
(root / "fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
