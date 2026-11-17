"""Prepare JAX multistart fits, or audit exported R fits on one common objective.

Run with 'prepare' before scripts/export_logistic_packages.R, then with 'check'.
Generated artifacts are kept under outputs/logistic-mirt/.
"""

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import expit, logsumexp
from jirt import logistic_irt
from jirt.logistic import _posterior

jax.config.update("jax_enable_x64", True)
root = Path("outputs/logistic-mirt")
inputs = json.loads((root / "inputs.json").read_text())


def dump(name, data):
    (root / name).write_text(json.dumps(data, indent=2) + "\n")


def audit(model, means, nodes=None, weights=None):
    case = inputs["cases"][model]
    y = np.asarray(case["patterns"])
    counts = np.asarray(case["counts"])
    nodes = np.asarray(inputs["nodes"] if nodes is None else nodes)
    weights = np.asarray(inputs["weights"] if weights is None else weights)
    a, b, c, u = (np.asarray(means[k]) for k in ("a", "b", "c", "u"))
    p = c + (u - c) * expit(a * (nodes[:, None] - b))
    joint = y @ np.log(p).T + (1 - y) @ np.log1p(-p).T + np.log(weights)
    ll = float(counts @ logsumexp(joint, axis=1))
    raw = np.concatenate([b, a, c] + ([u] if model == "4pl" else []))

    def loss(raw):
        per_pattern, _ = _posterior(
            raw,
            jnp.asarray(y, dtype=float),
            jnp.asarray(1 - y, dtype=float),
            jnp.asarray(nodes),
            jnp.log(jnp.asarray(weights)),
            model,
        )
        return -jnp.sum(per_pattern * counts) / counts.sum()

    grad = np.asarray(jax.grad(loss)(jnp.asarray(raw)))
    eig = np.linalg.eigvalsh(np.asarray(jax.hessian(loss)(jnp.asarray(raw))))
    return dict(
        log_likelihood=ll,
        mean_loss_gradient_max=float(np.max(np.abs(grad))),
        mean_loss_hessian_min=float(eig[0]),
        mean_loss_hessian_max=float(eig[-1]),
    )


if sys.argv[1] == "prepare":
    runs = {}
    for model in ("3pl", "4pl"):
        case = inputs["cases"][model]
        y = np.repeat(case["patterns"], case["counts"], axis=0)
        truth = dict(
            a=np.linspace(0.8, 1.6, 6), b=np.linspace(-1.5, 1.5, 6), c=np.full(6, 0.1)
        )
        if model == "4pl":
            truth["u"] = np.full(6, 0.9)
        rng = np.random.default_rng(2026)
        jitter = {
            k: v + rng.normal(0, 0.02 if k in ("c", "u") else 0.15, 6)
            for k, v in truth.items()
        }
        runs[model] = {}
        for name, starts in (
            ("neutral", None),
            ("generating", truth),
            ("perturbed", jitter),
        ):
            fit = logistic_irt(
                y, model=model, starts=starts, tolerance=1e-14, max_iter=5000
            )
            means = {k: np.asarray(fit.means[k]).tolist() for k in ("a", "b", "c", "u")}
            runs[model][name] = dict(
                means=means,
                converged=fit.converged,
                iterations=fit.iterations,
                **audit(model, means),
            )
            print(model, name, fit.converged, fit.log_likelihood, flush=True)
            dump("jax-multistart.json", runs)
elif sys.argv[1] == "refine":
    r_runs = json.loads((root / "other-packages.json").read_text())
    result = {}
    for model in ("3pl", "4pl"):
        case = inputs["cases"][model]
        y = np.repeat(case["patterns"], case["counts"], axis=0)
        result[model] = {}
        for name, run in r_runs["fits"][model].items():
            if not name.endswith("neutral"):
                continue
            starts = {
                k: np.asarray(v)
                for k, v in run["means"].items()
                if k != "u" or model == "4pl"
            }
            fit = logistic_irt(
                y, model=model, starts=starts, tolerance=1e-14, max_iter=5000
            )
            means = {k: np.asarray(fit.means[k]).tolist() for k in ("a", "b", "c", "u")}
            result[model][name] = dict(
                means=means,
                converged=fit.converged,
                iterations=fit.iterations,
                **audit(model, means),
            )
            print(model, name, fit.converged, fit.log_likelihood, flush=True)
            dump("jax-cross-starts.json", result)
elif sys.argv[1] == "check":
    jax_runs = json.loads((root / "jax-multistart.json").read_text())
    r_runs = json.loads((root / "other-packages.json").read_text())
    fine_path = root / "other-packages-fine.json"
    if fine_path.exists():
        fine_runs = json.loads(fine_path.read_text())
        for model in ("3pl", "4pl"):
            if isinstance(fine_runs["fits"][model], dict):
                r_runs["fits"][model].update(fine_runs["fits"][model])
    result = {}
    for model in ("3pl", "4pl"):
        best = max(jax_runs[model].values(), key=lambda r: r["log_likelihood"])
        records = {}
        expected = {
            f"sirt_{start}" for start in ("neutral", "generating", "jax_best", "mirt")
        }
        if model == "3pl":
            expected.update(
                f"ltm_{start}"
                for start in ("neutral", "generating", "jax_best", "mirt")
            )
        if fine_path.exists():
            expected.add("sirt_fine_jax_best")
        assert expected <= r_runs["fits"][model].keys(), "Incomplete R exports"
        for name, run in r_runs["fits"][model].items():
            means = run["means"]
            stats = audit(model, means)
            if "nodes" in run:
                native = audit(model, means, run["nodes"], run["weights"])
                stats["quadrature_likelihood_difference"] = abs(
                    stats["log_likelihood"] - native["log_likelihood"]
                )
                stats["reported_likelihood_error"] = abs(
                    native["log_likelihood"] - run["log_likelihood"]
                )
                stats["iteration_limit_reached"] = run["iteration_limit_reached"]
            else:
                stats["reported_likelihood_error"] = abs(
                    stats["log_likelihood"] - run["log_likelihood"]
                )
            stats["gap_to_best_jax"] = best["log_likelihood"] - stats["log_likelihood"]
            stats["max_parameter_difference"] = max(
                float(np.max(np.abs(np.asarray(means[k]) - best["means"][k])))
                for k in means
            )
            theta = np.linspace(-4, 4, 81)[:, None]

            def curve(m):
                return np.asarray(m["c"]) + (np.asarray(m["u"]) - m["c"]) * expit(
                    np.asarray(m["a"]) * (theta - m["b"])
                )

            stats["max_probability_difference"] = float(
                np.max(np.abs(curve(means) - curve(best["means"])))
            )
            stats["status"] = run["status"]
            stats["native_converged"] = (
                not run["iteration_limit_reached"]
                if "iteration_limit_reached" in run
                else run["status"] == 0
            )
            assert stats["reported_likelihood_error"] < 1e-5, (
                "Likelihood convention mismatch"
            )
            assert all(np.all(np.isfinite(means[k])) for k in means)
            assert np.all(np.asarray(means["a"]) > 0)
            assert np.all(np.asarray(means["u"]) - means["c"] > 1e-6)
            records[name] = stats
            print(model, name, json.dumps(stats), flush=True)
        result[model] = records
    dump("other-packages-comparison.json", result)
else:
    raise SystemExit("Usage: compare_logistic_packages.py prepare|refine|check")
