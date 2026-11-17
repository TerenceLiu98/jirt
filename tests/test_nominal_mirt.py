"""Matched R nominal/nested kernels and independent calibration diagnostics."""

import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss
from jirt.nominal import (
    nominal_irt,
    nominal_probabilities,
    nested_probabilities,
    _posterior,
)

jax.config.update("jax_enable_x64", True)
FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "nominal_mirt.json").read_text()
)


@pytest.mark.parametrize("model", FIXTURE["cases"])
def test_fixed_reference_kernel(model):
    case = FIXTURE["cases"][model]
    r = case["reference"]
    m = r["means"]
    counts = np.array(case["category_counts"])
    keys = np.array(case["answer_keys"])
    if model != "nominal":
        assert np.all(np.asarray(m["a"]) > 0)
        assert np.all(np.asarray(m["u"]) - m["c"] > 1e-6)
    fn = nominal_probabilities if model == "nominal" else nested_probabilities
    kwargs = {} if model == "nominal" else dict(answer_keys=keys)
    p = np.asarray(fn(r["theta"], **m, category_counts=counts, **kwargs))
    for j, k in enumerate(counts):
        np.testing.assert_allclose(
            p[:, j, :k], r["probabilities"][j], atol=1e-12, rtol=0
        )
    free = []
    for j, k in enumerate(counts):
        options = (
            list(range(k))
            if model == "nominal"
            else [c for c in range(k) if c != keys[j]]
        )
        free.extend((j, c) for c in options[1:])
    design = np.zeros((len(counts), counts.max(), len(free)))
    for i, (j, c) in enumerate(free):
        design[j, c, i] = 1
    raw = [m[name][j][c] for name in ("slopes", "intercepts") for j, c in free]
    if model != "nominal":
        for name in (
            ["a", "b"]
            + (["c"] if model in ("3pl", "4pl") else [])
            + (["u"] if model in ("3plu", "4pl") else [])
        ):
            raw.extend(m[name])
    nodes, w = hermgauss(41)
    nodes *= np.sqrt(2)
    ll, post = _posterior(
        jnp.array(raw),
        jnp.array(case["patterns"]),
        jnp.array(nodes),
        jnp.log(jnp.array(w) / np.sqrt(np.pi)),
        jnp.array(design),
        jnp.array(counts),
        jnp.array(keys),
        model,
    )
    score = np.asarray(post) @ nodes
    var = np.sum(np.asarray(post) * (nodes[None, :] - score[:, None]) ** 2, axis=1)
    assert abs(float(np.asarray(ll) @ case["counts"]) - r["log_likelihood"]) < 1e-6
    np.testing.assert_allclose(score, r["scores"], atol=1e-11, rtol=0)
    np.testing.assert_allclose(var, r["variances"], atol=1e-11, rtol=0)


@pytest.mark.parametrize("model", FIXTURE["cases"])
def test_independent_calibration(model):
    case = FIXTURE["cases"][model]
    r = case["reference"]
    y = np.repeat(case["patterns"], case["counts"], axis=0)
    fit = nominal_irt(
        y,
        category_counts=case["category_counts"],
        answer_keys=None if model == "nominal" else case["answer_keys"],
        model=model,
        tolerance=1e-13,
        max_iter=3000,
    )
    assert fit.converged and r["converged"]
    # A one-sided objective check does not claim fitted-parameter parity for
    # asymptote models. Their measured differences are reported in VALIDATION.md.
    assert fit.log_likelihood >= r["log_likelihood"] - 1e-5
    if model in ("nominal", "2pl"):
        for key, value in r["means"].items():
            np.testing.assert_allclose(fit.means[key], value, atol=1e-4, rtol=0)
