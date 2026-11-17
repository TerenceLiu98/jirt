"""R mirt fixtures: matched quadrature, fixed N(0,1), no item priors."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jirt import logistic_irt, logistic_probabilities
from jirt.logistic import _posterior

jax.config.update("jax_enable_x64", True)
REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "logistic_mirt.json").read_text()
)


@pytest.mark.parametrize("name", REFERENCE["cases"])
def test_mirt_probability_likelihood_and_scoring(name):
    """Isolate formula/scoring parity from optimizer convergence differences."""
    case = REFERENCE["cases"][name]
    means = {key: np.asarray(value) for key, value in case["means"].items()}
    np.testing.assert_allclose(
        logistic_probabilities(case["theta"], **means),
        case["probabilities"],
        atol=1e-12,
        rtol=0,
    )
    raw = [means["b"]]
    if case["model"] != "rasch":
        raw.append(means["a"][:1] if case["model"] == "1pl" else means["a"])
    if case["model"] in ("3pl", "4pl"):
        raw.append(means["c"])
    if case["model"] == "4pl":
        raw.append(means["u"])
    patterns = np.asarray(case["patterns"], dtype=float)
    nodes = jnp.asarray(REFERENCE["nodes"])
    ll, posterior = _posterior(
        jnp.asarray(np.concatenate(raw)),
        jnp.asarray(patterns == 1, dtype=float),
        jnp.asarray(patterns == 0, dtype=float),
        nodes,
        jnp.log(jnp.asarray(REFERENCE["weights"])),
        model=case["model"],
    )
    score = posterior @ nodes
    variance = jnp.sum(posterior * (nodes[None, :] - score[:, None]) ** 2, axis=1)
    np.testing.assert_allclose(
        np.asarray(ll) @ case["counts"], case["log_likelihood"], atol=1e-9, rtol=0
    )
    np.testing.assert_allclose(score, case["scores"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(variance, case["score_variances"], atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    "name",
    [
        name
        for name, case in REFERENCE["cases"].items()
        if case["model"] in ("rasch", "1pl", "2pl")
    ],
)
def test_mirt_independent_calibration(name):
    """Fit from neutral starts, not from the exported R estimates."""
    case = REFERENCE["cases"][name]
    counts = np.asarray(case["counts"], dtype=int)
    y = np.repeat(np.asarray(case["patterns"], dtype=float), counts, axis=0)
    fit = logistic_irt(
        y, model=case["model"], quadrature_points=41, tolerance=1e-14, max_iter=3000
    )
    assert fit.converged and case["converged"]
    for key in ("a", "b", "c", "u"):
        np.testing.assert_allclose(
            fit.means[key], case["means"][key], atol=1e-4, rtol=0
        )
    np.testing.assert_allclose(
        fit.log_likelihood, case["log_likelihood"], atol=1e-6, rtol=0
    )
    first = np.r_[0, np.cumsum(counts)[:-1]]
    np.testing.assert_allclose(
        np.asarray(fit.means["x"])[first], case["scores"], atol=1e-4, rtol=0
    )
