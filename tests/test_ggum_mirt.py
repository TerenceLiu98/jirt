"""Independent R GGUM fits on matched N(0,1) and quadrature."""

import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jirt import ggum_irt, ggum_probabilities
from jirt.unfolding import _posterior

jax.config.update("jax_enable_x64", True)
FIXTURE = json.loads((Path(__file__).parent / "fixtures/ggum_mirt.json").read_text())


@pytest.mark.parametrize("name", FIXTURE["cases"])
def test_fixed_mirt_parameters(name):
    case = FIXTURE["cases"][name]
    r = case["reference"]
    counts = np.asarray(case["category_counts"])
    p = ggum_probabilities(
        r["theta"], a=r["a"], delta=r["delta"], tau=r["tau"], category_counts=counts
    )
    for j, K in enumerate(counts):
        np.testing.assert_allclose(
            p[:, j, :K], r["probabilities"][j], atol=1e-14, rtol=0
        )
    mask = np.arange(counts.max() - 1) < counts[:, None] - 1
    design = np.zeros((*mask.shape, mask.sum()))
    design[mask] = np.eye(mask.sum())
    raw = np.r_[r["a"], r["delta"], np.asarray(r["tau"])[mask]]
    nodes = np.asarray(FIXTURE["nodes"])
    ll, post = _posterior(
        jnp.asarray(raw),
        jnp.asarray(case["patterns"]),
        jnp.asarray(nodes),
        jnp.log(jnp.asarray(FIXTURE["weights"])),
        jnp.asarray(design),
        jnp.asarray(counts),
    )
    score = post @ nodes
    var = jnp.sum(post * (nodes[None, :] - score[:, None]) ** 2, axis=1)
    np.testing.assert_allclose(
        ll @ np.asarray(case["counts"]), r["log_likelihood"], atol=1e-6, rtol=0
    )
    np.testing.assert_allclose(score, r["scores"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(var, r["variances"], atol=1e-10, rtol=0)


@pytest.mark.parametrize("name", FIXTURE["cases"])
def test_independent_mirt_fit(name):
    case = FIXTURE["cases"][name]
    r = case["reference"]
    fit = ggum_irt(
        np.repeat(case["patterns"], case["counts"], axis=0),
        category_counts=case["category_counts"],
        starts=case["starts"],
        tolerance=1e-13,
        max_iter=5000,
    )
    assert fit.converged and r["converged"], fit.message
    np.testing.assert_allclose(
        fit.log_likelihood, r["log_likelihood"], atol=1e-5, rtol=0
    )
    for k in ("a", "delta", "tau"):
        np.testing.assert_allclose(fit.means[k], r[k], atol=1e-4, rtol=0)
