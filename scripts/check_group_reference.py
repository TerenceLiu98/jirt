"""Compare R and JAX fits with class alignment; export reviewed regression data."""

import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment
from jirt.groups import _component_logs, _component_posterior, _mix_posterior

jax.config.update("jax_enable_x64", True)
root = Path("outputs/groups")
inp = json.loads((root / "inputs.json").read_text())
ref = json.loads((root / "reference.json").read_text())
fits = json.loads((root / "jirt.json").read_text())
assert set(inp["cases"]) == set(ref["cases"]) == set(fits)
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
    f = fits[name]
    C = case["classes"]
    counts = jnp.asarray(case["category_counts"])
    assert f["converged"] and r["converged"]
    order = np.arange(C)
    if case["kind"] == "mixture":
        cost = np.sum(
            (np.asarray(r["steps"])[:, None] - np.asarray(f["steps"])[None, :]) ** 2,
            axis=(2, 3),
        )
        _, order = linear_sum_assignment(cost)
    errors = {
        k: float(np.max(abs(np.asarray(f[k])[order] - np.asarray(r[k]))))
        for k in ("a", "steps", "mu", "sigma")
    }
    if case["kind"] != "group":
        fullorder = np.r_[order, C] if case["kind"] == "zero" else order
        errors["weights"] = float(
            np.max(abs(np.asarray(f["weights"])[fullorder] - r["weights"]))
        )
    errors["log_likelihood"] = abs(f["log_likelihood"] - r["log_likelihood"])
    a, b, mu, sd = (
        jnp.asarray(np.atleast_1d(r[k])) for k in ("a", "steps", "mu", "sigma")
    )
    theta = jnp.asarray(r["theta"])
    lp = _component_logs(
        jnp.broadcast_to(theta, (C, len(theta))), a, b, counts, case["model"]
    )
    expected = np.asarray(r["probabilities"]).transpose(0, 2, 1, 3)
    errors["fixed_probabilities"] = float(np.max(abs(np.exp(lp) - expected)))
    patterns = jnp.asarray(case["patterns"])
    ll, post, ability = _component_posterior(
        patterns,
        jnp.asarray(inp["nodes"]),
        jnp.log(jnp.asarray(inp["weights"])),
        a,
        b,
        mu,
        sd,
        counts,
        case["model"],
    )
    x = jnp.sum(post * ability[None, :, :], axis=-1)
    v = jnp.sum(post * (ability[None, :, :] - x[:, :, None]) ** 2, axis=-1)
    errors["fixed_scores"] = float(np.max(abs(x - np.asarray(r["scores"]).T)))
    errors["fixed_variances"] = float(np.max(abs(v - np.asarray(r["variances"]).T)))
    if case["kind"] == "group":
        total = np.sum(np.asarray(ll).T * np.asarray(case["counts"]))
    else:
        mixed, members = _mix_posterior(
            ll, patterns, jnp.asarray(r["weights"]), case["kind"] == "zero"
        )
        total = float(mixed @ jnp.asarray(case["counts"]))
        errors["fixed_memberships"] = float(
            np.max(abs(members - np.asarray(r["memberships"])))
        )
    errors["fixed_log_likelihood"] = abs(total - r["log_likelihood"])
    print(name, errors, flush=True)
    assert (
        max(
            errors[k]
            for k in ("fixed_probabilities", "fixed_scores", "fixed_variances")
        )
        < 1e-10
    )
    assert errors["fixed_log_likelihood"] < 1e-4
    assert errors["log_likelihood"] < 1e-4
    assert max(errors[k] for k in ("a", "steps", "mu", "sigma")) < 1e-3
    report[name] = errors
    fixture["cases"][name] = dict(**case, reference=r)
(root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
(root / "fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
