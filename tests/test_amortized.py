import jax
import numpy as np
import pytest
from scipy.special import expit
from jirt.amortized import amortized_irt

jax.config.update("jax_enable_x64", True)


def data():
    rng = np.random.default_rng(92)
    x = rng.normal(size=(20, 3))
    b = x @ np.array([0.7, -0.4, 0.3]) + 0.2
    theta = rng.normal(size=500)
    y = (rng.random((500, 20)) < expit(theta[:, None] - b)).astype(float)
    return x, y, b


def test_alignment_reproducibility_and_frozen_transform():
    x, y, b = data()
    ids = [f"i{j}" for j in range(20)]
    options = dict(item_ids=ids, key=jax.random.key(4), hidden_size=0, l2=0)
    fit = amortized_irt(y, item_features=x, feature_item_ids=ids, **options)
    reverse = amortized_irt(
        y, item_features=x[::-1], feature_item_ids=ids[::-1], **options
    )
    assert fit.converged, fit.message
    np.testing.assert_allclose(fit.difficulty, reverse.difficulty, atol=1e-12)
    np.testing.assert_allclose(fit.difficulty, b, atol=0.25)
    old = fit.feature_mean.copy()
    pred = fit.infer_difficulty(x[:2] + 100)
    np.testing.assert_array_equal(fit.feature_mean, old)
    assert pred.shape == (2,)
    np.testing.assert_allclose(fit.feature_mean, x.mean(axis=0))
    assert fit.predict(x[:3]).shape == (500, 3)


@pytest.mark.parametrize("hidden", [0, 5])
def test_missing_and_new_items(hidden):
    x, y, _ = data()
    y[0] = np.nan
    y[1, 0] = np.nan
    ids = [str(j) for j in range(20)]
    fit = amortized_irt(
        y,
        item_features=x,
        item_ids=ids,
        feature_item_ids=ids,
        key=jax.random.key(5),
        hidden_size=hidden,
        max_iter=2000,
        l2=0.01,
    )
    assert fit.converged, fit.message
    np.testing.assert_allclose(fit.ability[0], 0, atol=1e-14)
    np.testing.assert_allclose(fit.ability_variance[0], 1, atol=1e-14)
    assert np.isfinite(fit.predict([[1.0, 2.0, 3.0]], theta=[-1, 0, 1])).all()


def test_alignment_rejected():
    x, y, _ = data()
    ids = [str(j) for j in range(20)]
    with pytest.raises(ValueError, match="IDs"):
        amortized_irt(
            y,
            item_features=x,
            item_ids=ids,
            feature_item_ids=ids[:-1] + ["new"],
            key=jax.random.key(1),
        )


def test_whole_item_holdout():
    from importlib.util import spec_from_file_location, module_from_spec
    from pathlib import Path

    spec = spec_from_file_location(
        "amortized_example", Path(__file__).parents[1] / "examples/amortized.py"
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.run()
    assert report["training_items"] == 60 and report["held_out_items"] == 20
    assert report["amortized_log_loss"] < report["non_amortized_1pl_log_loss"] - 0.05
    assert report["held_out_difficulty_rmse"] < 0.15


def test_affine_matches_independent_scipy_objective():
    from numpy.polynomial.hermite import hermgauss
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    x, y, _ = data()
    ids = [str(j) for j in range(len(x))]
    fit = amortized_irt(
        y,
        item_features=x,
        item_ids=ids,
        feature_item_ids=ids,
        key=jax.random.key(3),
        hidden_size=0,
        l2=0.001,
        tolerance=1e-13,
    )
    z = (x - x.mean(axis=0)) / x.std(axis=0)
    nodes, w = hermgauss(41)
    nodes *= np.sqrt(2)
    lw = np.log(w / np.sqrt(np.pi))

    def loss(raw):
        b = z @ raw[:-1] + raw[-1]
        eta = nodes[:, None] - b
        joint = (
            y @ (-np.logaddexp(0, -eta)).T + (1 - y) @ (-np.logaddexp(0, eta)).T + lw
        )
        return -np.mean(logsumexp(joint, axis=1)) + 0.0005 * np.sum(raw[:-1] ** 2)

    opt = minimize(loss, np.zeros(4), method="BFGS", options=dict(gtol=1e-6))
    assert opt.success, opt.message
    np.testing.assert_allclose(fit.objective, opt.fun, atol=1e-9, rtol=0)
    np.testing.assert_allclose(
        fit.difficulty, z @ opt.x[:-1] + opt.x[-1], atol=2e-5, rtol=0
    )


@pytest.mark.parametrize("hidden", [0, 5])
def test_device_optimizer_matches_scipy(hidden):
    x, y, _ = data()
    ids = [str(j) for j in range(len(x))]
    options = dict(
        item_features=x,
        item_ids=ids,
        feature_item_ids=ids,
        key=jax.random.key(5),
        hidden_size=hidden,
        l2=0.01,
        max_iter=2000,
        tolerance=1e-6,
    )
    device = amortized_irt(y, optimizer="jax", **options)
    reference = amortized_irt(y, **{**options, "tolerance": 1e-11})
    assert device.converged, device.message
    assert reference.converged, reference.message
    np.testing.assert_allclose(device.objective, reference.objective, atol=1e-7)
    np.testing.assert_allclose(device.predict(x), reference.predict(x), atol=2e-4)
    np.testing.assert_allclose(device.ability, reference.ability, atol=2e-4)
    np.testing.assert_allclose(
        device.ability_variance, reference.ability_variance, atol=2e-4
    )
    assert device.trace.shape == (2,)
    limited = amortized_irt(y, optimizer="jax", **{**options, "max_iter": 1})
    assert not limited.converged and limited.iterations == 1
    assert "limit" in limited.message.lower()


def test_encoder_kernel_reuses_layout_with_new_features():
    from jirt.amortized import _value_grad

    x, y, _ = data()
    ids = [str(j) for j in range(len(x))]
    options = dict(
        item_ids=ids,
        feature_item_ids=ids,
        key=jax.random.key(5),
        hidden_size=0,
        max_iter=30,
    )
    _value_grad.clear_cache()
    first = amortized_irt(y, item_features=x, **options)
    size = _value_grad._cache_size()
    second = amortized_irt(y, item_features=x[:, ::-1], **options)
    assert _value_grad._cache_size() == size
    # Reversing features must reverse fitted coefficients, not capture old data.
    np.testing.assert_allclose(
        first.params["output"], second.params["output"][::-1], atol=1e-5
    )
