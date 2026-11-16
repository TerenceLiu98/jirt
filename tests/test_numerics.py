import json
from pathlib import Path
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import truncnorm
from jirt.ordinal import _moments
from jirt import dyn_irt, pois_irt, Control


@pytest.mark.parametrize(
    "category,bounds", [(1, (-np.inf, 0)), (2, (0, 1)), (3, (1, np.inf))]
)
def test_truncated_moments(category, bounds):
    eta = np.array([-30.0, -10.0, 0.0, 0.5, 1.0, 10.0, 30.0])
    mean, var = _moments(jnp.asarray(eta), jnp.ones(7), jnp.full(7, category))
    expected = truncnorm.stats(bounds[0] - eta, bounds[1] - eta, loc=eta, moments="mv")
    np.testing.assert_allclose(mean, expected[0], atol=1e-10, rtol=1e-8)
    np.testing.assert_allclose(var, expected[1], atol=1e-9, rtol=1e-6)


def fixture(name):
    return json.loads(
        (Path(__file__).parent / "fixtures" / f"{name}.json").read_text()
    )["kwargs"]


def test_dynamic_zero_diffusion_and_unordered_items():
    kw = fixture("dynamic")
    kw["priors"]["omega2"] = np.zeros(8)
    kw["control"] = Control(max_iter=3, tolerance=0)
    expected = dyn_irt(**kw)
    order = np.array([5, 0, 4, 1, 3, 2])
    kw["votes"] = np.asarray(kw["votes"])[:, order]
    kw["item_period"] = np.asarray(kw["item_period"])[order]
    for k in ("alpha", "beta"):
        kw["starts"][k] = np.asarray(kw["starts"][k])[order]
    actual = dyn_irt(**kw)
    np.testing.assert_allclose(actual.means["x"], expected.means["x"], atol=1e-10)
    assert np.isfinite(actual.variances["x"]).all()


def test_poisson_zero_counts():
    kw = fixture("poisson")
    kw["counts"] = np.zeros_like(kw["counts"])
    fit = pois_irt(**kw, control=Control(max_iter=5))
    assert all(np.isfinite(v).all() for v in fit.means.values())


def test_dynamic_invalid_period():
    kw = fixture("dynamic")
    kw["item_period"][0] = [100]
    with pytest.raises(ValueError):
        dyn_irt(**kw)


def test_dynamic_single_period_without_information():
    result = dyn_irt(
        np.zeros((1, 1)),
        starts=dict(alpha=np.zeros(1), beta=np.zeros(1), x=np.zeros((1, 1))),
        priors=dict(
            x_mu0=np.array([2.0]),
            x_variance0=np.array([3.0]),
            omega2=np.array([0.5]),
            beta_mu=np.zeros(2),
            beta_covariance=np.eye(2),
        ),
        start_period=[0],
        end_period=[0],
        item_period=[0],
        control=Control(max_iter=1),
    )
    np.testing.assert_allclose(result.means["x"], [[2.0]], atol=1e-12)
    np.testing.assert_allclose(result.variances["x"], [[3.5]], atol=1e-12)
    assert not result.converged
