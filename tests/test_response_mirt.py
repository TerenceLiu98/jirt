"""R mirt regression checks on matched priors, quadrature and restrictions."""

import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jirt import noncompensatory_irt, sequential_irt
from jirt.noncompensatory import (
    _posterior as pc_posterior,
    noncompensatory_probabilities,
)
from jirt.polytomous import _posterior as seq_posterior
from jirt import sequential_probabilities

jax.config.update("jax_enable_x64", True)
CASES = json.loads((Path(__file__).parent / "fixtures/response_mirt.json").read_text())[
    "cases"
]


@pytest.mark.parametrize("model", CASES)
def test_fixed_reference(model):
    case = CASES[model]
    r = case["reference"]
    pc = model.startswith("pc")
    nodes = np.asarray(case["nodes"])
    lw = np.log(case["weights"])
    if pc:
        mask = np.asarray(case["loading_mask"])
        n = mask.sum()
        design = np.zeros((*mask.shape, n))
        design[mask] = np.eye(n)
        raw = np.asarray(r["d"])[mask]
        if model != "pc1pl":
            raw = np.r_[raw, np.asarray(r["a"])[mask]]
        if model == "pc3pl":
            raw = np.r_[raw, r["c"]]
        ll, post = pc_posterior(
            jnp.asarray(raw),
            jnp.asarray(case["patterns"]),
            jnp.asarray(nodes),
            jnp.asarray(lw),
            jnp.asarray(design),
            model,
        )
        p = noncompensatory_probabilities(
            r["theta"], a=r["a"], d=r["d"], c=r["c"], loading_mask=mask, model=model
        )
        np.testing.assert_allclose(
            p, np.asarray(r["probabilities"])[:, :, 1].T, atol=2e-14
        )
    else:
        counts = np.asarray(case["category_counts"])
        mask = np.arange(counts.max() - 1) < counts[:, None] - 1
        design = np.zeros((*mask.shape, mask.sum()))
        design[mask] = np.eye(mask.sum())
        raw = np.asarray(r["steps"])[mask]
        if model == "sequential":
            raw = np.r_[raw, r["a"]]
        ll, post = seq_posterior(
            jnp.asarray(raw),
            jnp.asarray(case["patterns"]),
            jnp.asarray(nodes[:, 0]),
            jnp.asarray(lw),
            jnp.asarray(design),
            jnp.asarray(counts),
            model,
        )
        p = sequential_probabilities(
            np.asarray(r["theta"]).ravel(),
            a=r["a"],
            steps=r["steps"],
            category_counts=counts,
            model=model,
        )
        for j, k in enumerate(counts):
            np.testing.assert_allclose(p[:, j, :k], r["probabilities"][j], atol=2e-14)
    score = post @ nodes
    var = post @ (nodes**2) - score**2
    np.testing.assert_allclose(
        ll @ np.asarray(case["counts"]), r["log_likelihood"], atol=1e-6, rtol=0
    )
    np.testing.assert_allclose(score, r["scores"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(var, r["variances"], atol=1e-10, rtol=0)


@pytest.mark.parametrize("model", CASES)
def test_independent_fit(model):
    case = CASES[model]
    r = case["reference"]
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
    assert fit.converged, fit.message
    assert r["converged"]
    assert fit.log_likelihood >= r["log_likelihood"] - 1e-5
    # Strict parameter parity for the fixed-slope PC and sequential families.
    # PC2PL/PC3PL retain reported parameter differences in VALIDATION.md.
    if model in ("pc1pl", "sequential", "tutz"):
        for key in ("a", "d", "c") if pc else ("a", "steps"):
            np.testing.assert_allclose(fit.means[key], r[key], atol=2e-5, rtol=0)
