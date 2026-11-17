"""Compare JAX calibration with an actual R mirt export (run from repo root)."""

import json
from pathlib import Path

import jax
import numpy as np
import jax.numpy as jnp

from jirt import logistic_irt, logistic_probabilities
from jirt.logistic import _posterior

jax.config.update("jax_enable_x64", True)
root = Path("outputs/logistic-mirt")
reference = json.loads((root / "reference.json").read_text())
results = {}
for name, case in reference["cases"].items():
    patterns = np.asarray(case["patterns"], dtype=float)
    counts = np.asarray(case["counts"], dtype=int)
    y = np.repeat(patterns, counts, axis=0)
    fit = logistic_irt(
        y, model=case["model"], quadrature_points=41, tolerance=1e-14, max_iter=3000
    )
    first = np.r_[0, np.cumsum(counts)[:-1]]
    errors = {
        key: float(np.max(np.abs(np.asarray(fit.means[key]) - case["means"][key])))
        for key in ("a", "b", "c", "u")
    }
    errors.update(
        probability=float(
            np.max(
                np.abs(
                    np.asarray(fit.predict(np.asarray(case["theta"])))
                    - case["probabilities"]
                )
            )
        ),
        eap=float(np.max(np.abs(np.asarray(fit.means["x"])[first] - case["scores"]))),
        posterior_variance=float(
            np.max(
                np.abs(np.asarray(fit.variances["x"])[first] - case["score_variances"])
            )
        ),
        log_likelihood=abs(fit.log_likelihood - case["log_likelihood"]),
    )
    means = {key: np.asarray(value) for key, value in case["means"].items()}
    raw = [means["b"]]
    if case["model"] != "rasch":
        raw.append(means["a"][:1] if case["model"] == "1pl" else means["a"])
    if case["model"] in ("3pl", "4pl"):
        raw.append(means["c"])
    if case["model"] == "4pl":
        raw.append(means["u"])
    nodes = jnp.asarray(reference["nodes"])
    ll, posterior = _posterior(
        jnp.asarray(np.concatenate(raw)),
        jnp.asarray(patterns == 1, dtype=float),
        jnp.asarray(patterns == 0, dtype=float),
        nodes,
        jnp.log(jnp.asarray(reference["weights"])),
        model=case["model"],
    )
    score = posterior @ nodes
    variance = jnp.sum(posterior * (nodes[None, :] - score[:, None]) ** 2, axis=1)
    fixed_errors = dict(
        probability=float(
            np.max(
                np.abs(
                    np.asarray(logistic_probabilities(case["theta"], **means))
                    - case["probabilities"]
                )
            )
        ),
        log_likelihood=abs(float(np.asarray(ll) @ counts) - case["log_likelihood"]),
        eap=float(np.max(np.abs(np.asarray(score) - case["scores"]))),
        posterior_variance=float(
            np.max(np.abs(np.asarray(variance) - case["score_variances"]))
        ),
    )
    results[name] = dict(
        errors=errors,
        # A common strict target; differences remain visible rather than
        # relaxing tolerances specifically for 3PL/4PL.
        fit_agreement=(
            max(
                errors[k]
                for k in (
                    "a",
                    "b",
                    "c",
                    "u",
                    "probability",
                    "eap",
                    "posterior_variance",
                )
            )
            < 1e-4
            and errors["log_likelihood"] < 1e-6
        ),
        fixed_parameter_errors=fixed_errors,
        jirt_converged=fit.converged,
        mirt_converged=case["converged"],
        jirt_iterations=fit.iterations,
        mirt_iterations=case["iterations"],
        jirt_log_likelihood=fit.log_likelihood,
        mirt_log_likelihood=case["log_likelihood"],
    )
    print(name, json.dumps(results[name]), flush=True)
(root / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
assert len(results) == 8, "Incomplete R export"
assert all(r["jirt_converged"] and r["mirt_converged"] for r in results.values())
for result in results.values():
    fixed = result["fixed_parameter_errors"]
    assert fixed["log_likelihood"] < 1e-9
    assert max(fixed[k] for k in ("probability", "eap", "posterior_variance")) < 1e-12
mismatches = [name for name, result in results.items() if not result["fit_agreement"]]
print("Independent-fit agreement exceptions:", ", ".join(mismatches) or "none")
