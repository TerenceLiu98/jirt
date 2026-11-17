"""Compare independent fits on identical quadrature and export regression inputs."""

import json
from pathlib import Path
import jax
import numpy as np
from jirt import (
    noncompensatory_irt,
    noncompensatory_probabilities,
    sequential_irt,
    sequential_probabilities,
)

jax.config.update("jax_enable_x64", True)
root = Path("outputs/response-models")
inp = json.loads((root / "inputs.json").read_text())
ref = json.loads((root / "reference.json").read_text())
assert set(ref["cases"]) == set(inp["cases"])
fixture = dict(mirt_version=ref["mirt_version"], r_version=ref["r_version"], cases={})
report = {}
for model, case in inp["cases"].items():
    r = ref["cases"][model]
    pc = model.startswith("pc")
    y = np.repeat(case["patterns"], case["counts"], axis=0)
    kwargs = (
        dict(loading_mask=np.asarray(case["loading_mask"]))
        if pc
        else dict(category_counts=case["category_counts"])
    )
    fit = (noncompensatory_irt if pc else sequential_irt)(
        y,
        model=model,
        **kwargs,
        quadrature_points=15 if pc else 41,
        tolerance=1e-13,
        max_iter=5000,
    )
    assert fit.converged and r["converged"], (model, fit.message)
    pars = {k: r[k] for k in (("a", "d", "c") if pc else ("a", "steps"))}
    theta = np.asarray(r["theta"]) if pc else np.asarray(r["theta"]).ravel()
    fixed = np.asarray(
        (noncompensatory_probabilities if pc else sequential_probabilities)(
            theta, **pars, **kwargs, model=model
        )
    )
    pred = np.asarray(fit.predict(theta))
    expected = np.asarray(r["probabilities"])[:, :, 1].T if pc else None

    def probability_error(p):
        return (
            float(np.max(abs(p - expected)))
            if pc
            else max(
                float(np.max(abs(p[:, j, :k] - r["probabilities"][j])))
                for j, k in enumerate(case["category_counts"])
            )
        )

    first = np.r_[0, np.cumsum(case["counts"])[:-1]]
    score = np.asarray(fit.means["x"])[first].reshape(len(first), -1)
    var = np.asarray(fit.variances["x"])[first].reshape(len(first), -1)
    errors = {k: float(np.max(abs(np.asarray(fit.means[k]) - r[k]))) for k in pars}
    errors.update(
        log_likelihood=abs(fit.log_likelihood - r["log_likelihood"]),
        fixed_probabilities=probability_error(fixed),
        probabilities=probability_error(pred),
        scores=float(np.max(abs(score - r["scores"]))),
        variances=float(np.max(abs(var - r["variances"]))),
    )
    print(model, errors, flush=True)
    assert errors["fixed_probabilities"] < 1e-12
    assert errors["log_likelihood"] < 1e-4
    report[model] = errors
    fixture["cases"][model] = dict(**case, reference=r)
(root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
(root / "fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
