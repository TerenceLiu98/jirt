"""Cross-solver checks on identified calibration problems."""

import jax
import numpy as np
import pytest
from jirt import logistic_irt, polytomous_irt, ggum_irt

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("model", ["rasch", "1pl", "2pl"])
def test_logistic_device_fit(model):
    from test_logistic import data

    y = data()
    device = logistic_irt(
        y, model=model, optimizer="jax", tolerance=1e-6, max_iter=2000
    )
    reference = logistic_irt(y, model=model, tolerance=1e-11, max_iter=2000)
    assert device.converged, device.message
    assert reference.converged, reference.message
    np.testing.assert_allclose(
        device.log_likelihood, reference.log_likelihood, atol=1e-4, rtol=0
    )
    np.testing.assert_allclose(device.probabilities, reference.probabilities, atol=3e-4)
    np.testing.assert_allclose(
        device.variances["x"], reference.variances["x"], atol=3e-4
    )


@pytest.mark.parametrize("model", ["grm", "pcm", "gpcm", "rsm"])
def test_polytomous_device_fit(model):
    from test_polytomous import calibration_data

    y, counts, _, _ = calibration_data(model)
    device = polytomous_irt(
        y,
        category_counts=counts,
        model=model,
        optimizer="jax",
        tolerance=1e-6,
        max_iter=3000,
    )
    reference = polytomous_irt(
        y, category_counts=counts, model=model, tolerance=1e-12, max_iter=3000
    )
    assert device.converged, device.message
    assert reference.converged, reference.message
    np.testing.assert_allclose(
        device.log_likelihood / len(y), reference.log_likelihood / len(y), atol=1e-8
    )
    np.testing.assert_allclose(
        device.predict([-2, 0, 2]), reference.predict([-2, 0, 2]), atol=2e-4
    )
    np.testing.assert_allclose(device.means["x"], reference.means["x"], atol=2e-4)


def test_unfolding_device_fit():
    from test_unfolding import calibration_case

    counts, starts, patterns, freq = calibration_case()
    y = np.repeat(patterns, np.maximum(1, np.round(freq / 10).astype(int)), axis=0)
    options = dict(category_counts=counts, starts=starts, max_iter=3000)
    device = ggum_irt(y, optimizer="jax", tolerance=1e-6, **options)
    reference = ggum_irt(y, tolerance=1e-12, **options)
    assert device.converged, device.message
    assert reference.converged, reference.message
    np.testing.assert_allclose(
        device.log_likelihood / len(y), reference.log_likelihood / len(y), atol=1e-7
    )
    np.testing.assert_allclose(
        device.predict([-2, 0, 2]), reference.predict([-2, 0, 2]), atol=1e-3
    )


@pytest.mark.parametrize(
    "family", ["multidimensional", "noncompensatory", "nominal", "group", "mixture"]
)
def test_other_device_families(family):
    from jirt import (
        multidimensional_irt,
        noncompensatory_irt,
        nominal_irt,
        group_irt,
        mixture_irt,
    )
    from test_logistic import data

    y = data()
    options = dict(max_iter=3000, tolerance=1e-6)
    if family == "multidimensional":
        function = multidimensional_irt
        options.update(dimensions=1)
    elif family == "noncompensatory":
        function = noncompensatory_irt
        options.update(loading_mask=np.ones((6, 1), bool), model="pc2pl")
    elif family == "nominal":
        function = nominal_irt
        options.update(category_counts=[2] * 6, model="2pl", answer_keys=[1] * 6)
    else:
        function = group_irt if family == "group" else mixture_irt
        options.update(
            model="2pl", starts=dict(a=np.ones((1, 6)), steps=np.zeros((1, 6, 1)))
        )
        options.update(
            dict(groups=np.zeros(len(y), int)) if family == "group" else dict(classes=1)
        )
    device = function(y, optimizer="jax", **options)
    reference = function(y, **{**options, "tolerance": 1e-12})
    assert device.converged, device.message
    assert reference.converged, reference.message
    np.testing.assert_allclose(
        device.log_likelihood, reference.log_likelihood, atol=1e-4, rtol=0
    )
    np.testing.assert_allclose(device.means["x"], reference.means["x"], atol=5e-4)
    np.testing.assert_allclose(
        device.variances["x"], reference.variances["x"], atol=5e-4
    )


@pytest.mark.parametrize("model", ["3pl", "4pl"])
def test_asymptote_constraints_and_limit(model):
    from test_logistic import data

    result = logistic_irt(
        data(), model=model, optimizer="jax", max_iter=3, tolerance=1e-12
    )
    assert not result.converged and result.iterations == 3
    assert result.message == "Iteration limit reached"
    assert np.all(result.means["a"] >= 1e-6)
    assert np.all(result.means["c"] >= 1e-6 - 1e-10)
    assert np.all(result.means["u"] - result.means["c"] >= 1e-6 - 1e-10)
    if model == "4pl":
        assert np.all(result.means["u"] <= 1 - 1e-6 + 1e-10)


def test_group_kernel_reuses_layout_with_new_response_frequencies():
    from itertools import product
    from jirt.groups import group_irt, _group_value_grad

    patterns = np.array(list(product([0.0, 1.0], repeat=4)))
    y = np.vstack([patterns, np.zeros((10, 4))])
    z = np.vstack([patterns, np.ones((10, 4))])
    options = dict(
        groups=np.zeros(len(y), int),
        model="rasch",
        starts=dict(steps=np.zeros((1, 4, 1))),
        max_iter=100,
    )
    _group_value_grad.clear_cache()
    first = group_irt(y, **options)
    size = _group_value_grad._cache_size()
    second = group_irt(z, **options)
    assert _group_value_grad._cache_size() == size
    eager = group_irt(z, jit=False, **options)
    np.testing.assert_allclose(second.means["steps"], eager.means["steps"], atol=1e-8)
    assert not np.allclose(first.means["steps"], second.means["steps"])
