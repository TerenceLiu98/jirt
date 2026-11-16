import json
from pathlib import Path
import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest
from jirt import (
    bin_irt,
    ord_irt,
    dyn_irt,
    hier_irt,
    network_irt,
    pois_irt,
    Control,
    bootstrap,
)

MODELS = {
    "binary_em": bin_irt,
    "ordinal": ord_irt,
    "dynamic": dyn_irt,
    "hierarchical": hier_irt,
    "network": network_irt,
    "poisson": pois_irt,
}


def kwargs(name):
    return json.loads(
        (Path(__file__).parent / "fixtures" / f"{name}.json").read_text()
    )["kwargs"]


@pytest.mark.parametrize("name", MODELS)
def test_nonfinite_starts_rejected(name):
    kw = kwargs(name)
    first = next(iter(kw["starts"]))
    a = np.asarray(kw["starts"][first], dtype=float)
    a.flat[0] = np.nan
    kw["starts"][first] = a
    with pytest.raises(ValueError):
        MODELS[name](**kw)


@pytest.mark.parametrize("name", MODELS)
def test_x64_required(name):
    previous = jax.config.x64_enabled
    try:
        jax.config.update("jax_enable_x64", False)
        with pytest.raises(ValueError, match="x64"):
            MODELS[name](**kwargs(name))
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_bootstrap_failure_is_visible():
    kw = kwargs("binary_em")
    y = kw.pop("votes")
    kw["control"] = Control(max_iter=3)
    result = bin_irt(y, **kw)
    rep = np.stack([y, y])
    rep[1, 0, 0] = 99
    out = bootstrap(
        result,
        y,
        model="binary",
        fit_kwargs=kw,
        key=jax.random.key(1),
        n_trials=2,
        pseudo_data=rep,
    )
    assert 1 in out.failures
    assert np.isnan(out.estimates[1]).all()
    assert np.isnan(out.standard_errors).all()


def test_network_anchor_prediction_invariance():
    kw = kwargs("network")
    kw["control"] = Control(max_iter=5, tolerance=0)
    plain = network_irt(**kw)
    anchored = network_irt(**kw, anchor_item=0)
    np.testing.assert_allclose(plain.probabilities, anchored.probabilities, atol=1e-12)
    assert anchored.means["w"][0] >= np.mean(anchored.means["w"])


def test_positive_definite_prior_required():
    kw = kwargs("dynamic")
    kw["priors"]["beta_covariance"] = [[1, 2], [2, 1]]
    with pytest.raises(ValueError):
        dyn_irt(**kw)


@pytest.mark.parametrize(
    "model,fn,input_key",
    [
        ("binary", bin_irt, "votes"),
        ("ordinal", ord_irt, "responses"),
        ("dynamic", dyn_irt, "votes"),
        ("hierarchical", hier_irt, "votes"),
    ],
)
def test_bootstrap_random_reproducible(model, fn, input_key):
    name = "binary_em" if model == "binary" else model
    kw = kwargs(name)
    y = kw.pop(input_key)
    kw["control"] = Control(max_iter=3, tolerance=0)
    fit = fn(y, **kw)
    a = bootstrap(
        fit, y, model=model, fit_kwargs=kw, key=jax.random.key(123), n_trials=2
    )
    b = bootstrap(
        fit, y, model=model, fit_kwargs=kw, key=jax.random.key(123), n_trials=2
    )
    assert not a.failures
    np.testing.assert_array_equal(a.estimates, b.estimates)


def test_bootstrap_rejects_invalid_probabilities():
    kw = kwargs("binary_em")
    y = kw.pop("votes")
    kw["control"] = Control(max_iter=3)
    result = bin_irt(y, **kw)
    result.probabilities = np.full(np.shape(y), np.nan)
    with pytest.raises(ValueError, match="probabilities"):
        bootstrap(
            result, y, model="binary", fit_kwargs=kw, key=jax.random.key(1), n_trials=2
        )
