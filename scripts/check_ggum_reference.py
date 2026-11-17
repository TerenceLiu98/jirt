"""Compare independently fitted GGUM models and prepare a regression baseline."""

import json
from pathlib import Path
import jax
import numpy as np
from jirt import ggum_irt, ggum_probabilities

jax.config.update("jax_enable_x64", True)
root = Path("outputs/ggum")
inp = json.loads((root / "inputs.json").read_text())
ref = json.loads((root / "reference.json").read_text())
assert set(ref["cases"]) == set(inp["cases"])
fixture = dict(
    mirt_version=ref["mirt_version"],
    r_version=ref["r_version"],
    nodes=inp["nodes"],
    weights=inp["weights"],
    cases={},
)
report = {}
for name, case in inp["cases"].items():
    r = ref["cases"][name]
    counts = np.asarray(case["category_counts"])
    y = np.repeat(case["patterns"], case["counts"], axis=0)
    fit = ggum_irt(
        y, category_counts=counts, starts=case["starts"], tolerance=1e-13, max_iter=5000
    )
    assert fit.converged and r["converged"], fit.message
    fixed = np.asarray(
        ggum_probabilities(
            r["theta"], a=r["a"], delta=r["delta"], tau=r["tau"], category_counts=counts
        )
    )
    p = np.asarray(fit.predict(r["theta"]))

    def error(pred):
        return max(
            float(np.max(abs(pred[:, j, :k] - r["probabilities"][j])))
            for j, k in enumerate(counts)
        )

    first = np.r_[0, np.cumsum(case["counts"])[:-1]]
    errors = {
        k: float(np.max(abs(np.asarray(fit.means[k]) - r[k])))
        for k in ("a", "delta", "tau")
    }
    errors.update(
        log_likelihood=abs(fit.log_likelihood - r["log_likelihood"]),
        fixed_probabilities=error(fixed),
        probabilities=error(p),
        scores=float(np.max(abs(np.asarray(fit.means["x"])[first] - r["scores"]))),
        variances=float(
            np.max(abs(np.asarray(fit.variances["x"])[first] - r["variances"]))
        ),
    )
    print(name, errors, flush=True)
    assert errors["fixed_probabilities"] < 1e-12
    assert errors["log_likelihood"] < 1e-4
    assert (
        max(
            errors[k]
            for k in ("a", "delta", "tau", "probabilities", "scores", "variances")
        )
        < 1e-3
    )
    report[name] = errors
    fixture["cases"][name] = dict(**case, reference=r)
(root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
(root / "fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
