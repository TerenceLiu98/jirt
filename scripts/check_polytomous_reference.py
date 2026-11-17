"""Check independent mirt fits and prepare a regression fixture."""

import json
from pathlib import Path
import jax
import numpy as np
from jirt.polytomous import polytomous_irt, polytomous_probabilities

jax.config.update("jax_enable_x64", True)
root = Path("outputs/polytomous")
inp = json.loads((root / "inputs.json").read_text())
ref = json.loads((root / "reference.json").read_text())
assert set(ref["cases"]) == {"grm", "pcm", "gpcm", "rsm"}
report = {}
fixture = dict(mirt_version=ref["mirt_version"], r_version=ref["r_version"], cases={})
for model, case in inp["cases"].items():
    r = ref["cases"][model]
    counts = np.asarray(case["category_counts"])
    y = np.repeat(case["patterns"], case["counts"], axis=0)
    fit = polytomous_irt(
        y, category_counts=counts, model=model, tolerance=1e-13, max_iter=3000
    )
    assert fit.converged and r["converged"]
    p = np.asarray(fit.predict(r["theta"]))
    fixed = np.asarray(
        polytomous_probabilities(
            r["theta"], a=r["a"], steps=r["steps"], category_counts=counts, model=model
        )
    )
    errors = dict(
        a=float(np.max(abs(np.asarray(fit.means["a"]) - r["a"]))),
        steps=float(np.max(abs(np.asarray(fit.means["steps"]) - r["steps"]))),
        log_likelihood=abs(fit.log_likelihood - r["log_likelihood"]),
        probabilities=max(
            float(np.max(abs(p[:, j, :k] - r["probabilities"][j])))
            for j, k in enumerate(counts)
        ),
        fixed_probabilities=max(
            float(np.max(abs(fixed[:, j, :k] - r["probabilities"][j])))
            for j, k in enumerate(counts)
        ),
    )
    first = np.r_[0, np.cumsum(case["counts"])[:-1]]
    errors["scores"] = float(
        np.max(abs(np.asarray(fit.means["x"])[first] - r["scores"]))
    )
    errors["variances"] = float(
        np.max(abs(np.asarray(fit.variances["x"])[first] - r["variances"]))
    )
    print(model, errors, flush=True)
    assert errors["fixed_probabilities"] < 1e-12
    assert errors["log_likelihood"] < 1e-5
    assert (
        max(errors[k] for k in ("a", "steps", "probabilities", "scores", "variances"))
        < 1e-4
    )
    report[model] = errors
    fixture["cases"][model] = dict(**case, reference=r)
(root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
(root / "fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
