"""Matched R fits and label-aligned recovery for groups, mixtures and zeros."""

import copy
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from jirt import group_irt, mixture_irt
from jirt.groups import _component_logs, _component_posterior, _mix_posterior

jax.config.update("jax_enable_x64", True)
DATA = json.loads((Path(__file__).parent / "fixtures/groups_mirt.json").read_text())


@pytest.mark.parametrize("name", DATA["cases"])
def test_fixed_reference(name):
    case = DATA["cases"][name]
    r = case["reference"]
    C = case["classes"]
    a, b, mu, sd = (
        jnp.asarray(np.atleast_1d(r[k])) for k in ("a", "steps", "mu", "sigma")
    )
    theta = jnp.asarray(r["theta"])
    counts = jnp.asarray(case["category_counts"])
    patterns = jnp.asarray(case["patterns"])
    p = jnp.exp(
        _component_logs(
            jnp.broadcast_to(theta, (C, len(theta))), a, b, counts, case["model"]
        )
    )
    np.testing.assert_allclose(
        p, np.asarray(r["probabilities"]).transpose(0, 2, 1, 3), atol=1e-14, rtol=0
    )
    ll, post, ability = _component_posterior(
        patterns,
        jnp.asarray(DATA["nodes"]),
        jnp.log(jnp.asarray(DATA["weights"])),
        a,
        b,
        mu,
        sd,
        counts,
        case["model"],
    )
    x = jnp.sum(post * ability[None, :, :], axis=-1)
    var = jnp.sum(post * (ability[None, :, :] - x[:, :, None]) ** 2, axis=-1)
    np.testing.assert_allclose(x, np.asarray(r["scores"]).T, atol=1e-10, rtol=0)
    np.testing.assert_allclose(var, np.asarray(r["variances"]).T, atol=1e-10, rtol=0)
    if case["kind"] == "group":
        total = np.sum(np.asarray(ll).T * np.asarray(case["counts"]))
    else:
        mixed, membership = _mix_posterior(
            ll, patterns, jnp.asarray(r["weights"]), case["kind"] == "zero"
        )
        total = float(mixed @ jnp.asarray(case["counts"]))
        np.testing.assert_allclose(membership, r["memberships"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(total, r["log_likelihood"], atol=1e-6, rtol=0)


@pytest.mark.parametrize("name", DATA["cases"])
def test_independent_fit_and_recovery(name):
    case = DATA["cases"][name]
    r = case["reference"]
    truth = case["truth"]
    C = case["classes"]
    patterns = np.asarray(case["patterns"])
    freq = np.asarray(case["counts"])
    options = dict(
        model=case["model"],
        category_counts=case["category_counts"],
        shared_a=case["shared_a"],
        shared_steps=case["shared_steps"],
        quadrature_points=61,
        tolerance=1e-13,
        max_iter=5000,
    )
    if case["kind"] == "group":
        parts = [np.repeat(patterns, f, axis=0) for f in freq]
        y = np.concatenate(parts)
        groups = np.concatenate([np.full(len(p), i) for i, p in enumerate(parts)])
        fit = group_irt(
            y,
            groups=groups,
            starts=case["starts"][0],
            free_distribution=True,
            **options,
        )
    else:
        starts = copy.deepcopy(case["starts"])
        # A genuinely different initialization with reversed regular class labels.
        if case["kind"] == "mixture":
            starts[1] = {k: np.asarray(v)[::-1].copy() for k, v in starts[1].items()}
        fit = mixture_irt(
            np.repeat(patterns, freq, axis=0),
            classes=C,
            starts=starts,
            zero_inflated=case["kind"] == "zero",
            **options,
        )
        assert len(fit.start_log_likelihoods) == 2
        np.testing.assert_allclose(
            fit.start_log_likelihoods, fit.log_likelihood, atol=1e-4, rtol=0
        )
        np.testing.assert_allclose(fit.memberships.sum(axis=1), 1, atol=1e-14)
    assert fit.converged and r["converged"], fit.message
    np.testing.assert_allclose(
        fit.log_likelihood, r["log_likelihood"], atol=1e-4, rtol=0
    )
    order = np.arange(C)
    if case["kind"] == "mixture":
        cost = np.sum(
            (
                np.asarray(truth["steps"])[:, None]
                - np.asarray(fit.means["steps"])[None, :]
            )
            ** 2,
            axis=(2, 3),
        )
        _, order = linear_sum_assignment(cost)
    for key in ("a", "steps", "mu", "sigma"):
        np.testing.assert_allclose(
            np.asarray(fit.means[key])[order], np.atleast_1d(r[key]), atol=2e-4, rtol=0
        )
        np.testing.assert_allclose(
            np.asarray(fit.means[key])[order], truth[key], atol=0.003, rtol=0
        )
    if case["kind"] != "group":
        fullorder = np.r_[order, C] if case["kind"] == "zero" else order
        np.testing.assert_allclose(
            fit.weights[fullorder], r["weights"], atol=2e-5, rtol=0
        )
        np.testing.assert_allclose(
            fit.weights[fullorder], truth["weights"], atol=0.001, rtol=0
        )
        first = np.r_[0, np.cumsum(freq)[:-1]]
        np.testing.assert_allclose(
            np.asarray(fit.memberships)[first][:, fullorder],
            r["memberships"],
            atol=2e-4,
            rtol=0,
        )
    else:
        np.testing.assert_allclose(fit.means["a"][0, :2], fit.means["a"][1, :2], atol=0)
        np.testing.assert_allclose(
            fit.means["steps"][0, :2], fit.means["steps"][1, :2], atol=0
        )
