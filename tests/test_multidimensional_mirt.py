"""Matched mirt calibration and bfactor conditional-prediction regressions."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jirt import multidimensional_irt, multidimensional_probabilities
from jirt.multidimensional import _posterior, _quadrature

jax.config.update("jax_enable_x64", True)
FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "multidimensional_mirt.json").read_text()
)


@pytest.mark.parametrize("name", FIXTURE["cases"])
def test_mirt_calibration_and_scores(name):
    case = FIXTURE["cases"][name]
    r = case["reference"]
    mask = np.asarray(case["mask"])
    a = np.asarray(r["a"])
    d = np.asarray(r["d"])
    patterns = np.asarray(case["patterns"])
    counts = np.asarray(case["counts"])
    if name in ("bifactor", "testlet"):
        kwargs = dict(structure=name, groups=[0, 0, 0, 1, 1, 1, -1, -1, -1])
    else:
        kwargs = dict(dimensions=2)
        if name == "confirmatory":
            kwargs["loading_mask"] = mask
    fit = multidimensional_irt(
        np.repeat(patterns, counts, axis=0),
        **kwargs,
        quadrature_points=15,
        tolerance=1e-13,
        max_iter=3000,
    )
    assert fit.converged and r["converged"]
    np.testing.assert_allclose(fit.means["a"], a, atol=1e-4, rtol=0)
    np.testing.assert_allclose(fit.means["d"], d, atol=1e-4, rtol=0)
    assert abs(fit.log_likelihood - r["log_likelihood"]) < 1e-5
    np.testing.assert_allclose(
        fit.predict(case["theta"]), r["probabilities"], atol=3e-5, rtol=0
    )
    first = np.r_[0, np.cumsum(counts)[:-1]]
    np.testing.assert_allclose(
        np.asarray(fit.means["x"])[first], r["scores"], atol=1e-4, rtol=0
    )
    np.testing.assert_allclose(
        np.asarray(fit.variances["x"])[first], r["variances"], atol=1e-4, rtol=0
    )
    if "bfactor_at_jirt" in r:
        np.testing.assert_allclose(
            fit.predict(case["theta"]), r["bfactor_at_jirt"], atol=3e-5, rtol=0
        )


@pytest.mark.parametrize("name", FIXTURE["cases"])
def test_fixed_mirt_parameter_kernel(name):
    case = FIXTURE["cases"][name]
    r = case["reference"]
    mask = np.asarray(case["mask"])
    a = np.asarray(r["a"])
    d = np.asarray(r["d"])
    patterns = np.asarray(case["patterns"])
    np.testing.assert_allclose(
        multidimensional_probabilities(case["theta"], a=a, d=d),
        r["probabilities"],
        atol=1e-13,
        rtol=0,
    )
    design = np.zeros((*a.shape, mask.sum()))
    for k, (j, f) in enumerate(zip(*np.where(mask), strict=True)):
        design[j, f, k] = 1
    nodes, lw = _quadrature(a.shape[1], FIXTURE["quadrature_points"], 100000)
    ll, posterior = _posterior(
        jnp.asarray(np.r_[d, a[mask]]),
        jnp.asarray(patterns == 1),
        jnp.asarray(patterns == 0),
        jnp.asarray(nodes),
        jnp.asarray(lw),
        jnp.asarray(design),
    )
    scores = np.asarray(posterior) @ nodes
    var = np.sum(
        np.asarray(posterior)[:, :, None]
        * (nodes[None, :, :] - scores[:, None, :]) ** 2,
        axis=1,
    )
    assert abs(float(np.asarray(ll) @ case["counts"]) - r["log_likelihood"]) < 1e-7
    np.testing.assert_allclose(scores, r["scores"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(var, r["variances"], atol=1e-12, rtol=0)
