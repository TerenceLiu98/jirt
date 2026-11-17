"""Independent SciPy likelihood and optimizer checks for logistic calibration."""

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize, LinearConstraint
from scipy.special import expit, logsumexp
from jirt import logistic_irt, logistic_probabilities


@pytest.mark.parametrize("c,u", [(0, 1), (0.2, 1), (0.2, 0.9)])
def test_response_formula_and_extremes(c, u):
    theta = np.array([-1e6, -2, 0, 2, 1e6])
    a = np.array([0.5, 1, 2])
    b = np.array([-1, 0, 1])
    expected = c + (u - c) * expit(a * (theta[:, None] - b))
    for fn in (logistic_probabilities, jax.jit(logistic_probabilities)):
        np.testing.assert_allclose(fn(theta, a=a, b=b, c=c, u=u), expected, atol=1e-15)
    np.testing.assert_allclose(expected[0], c)
    np.testing.assert_allclose(expected[-1], u)


def test_model_reductions():
    theta = np.linspace(-5, 5, 21)
    p4 = logistic_probabilities(theta, a=[1, 1], b=[-0.5, 0.5], c=[0, 0], u=[1, 1])
    p2 = logistic_probabilities(theta, a=[1, 1], b=[-0.5, 0.5])
    p1 = logistic_probabilities(theta, a=1, b=[-0.5, 0.5])
    np.testing.assert_array_equal(p4, p2)
    np.testing.assert_array_equal(p2, p1)


def data():
    rng = np.random.default_rng(314)
    x = rng.normal(size=400)
    p = 0.08 + 0.84 * expit((x[:, None] - np.linspace(-1.5, 1.5, 6)) * 1.2)
    y = (rng.random(p.shape) < p).astype(float)
    y[rng.random(y.shape) < 0.08] = np.nan
    y[0] = np.nan
    return y


@pytest.mark.parametrize("model", ["rasch", "1pl", "2pl", "3pl", "4pl"])
def test_independent_marginal_likelihood_fit(model):
    # Near-population pattern frequencies avoid weakly identified, extreme-slope
    # finite-sample optima when comparing item parameters across optimizers.
    import itertools

    patterns = np.array(list(itertools.product([0.0, 1.0], repeat=6)))
    q, qw = hermgauss(31)
    slope = 1.0 if model in ("rasch", "1pl") else np.linspace(0.8, 1.6, 6)
    c = 0.1 if model in ("3pl", "4pl") else 0.0
    u = 0.9 if model == "4pl" else 1.0
    p = c + (u - c) * expit(
        slope * (q[:, None] * np.sqrt(2) - np.linspace(-1.5, 1.5, 6))
    )
    freq = np.exp(patterns @ np.log(p).T + (1 - patterns) @ np.log1p(-p).T) @ (
        qw / np.sqrt(np.pi)
    )
    y = np.repeat(patterns, np.maximum(1, np.round(freq * 10000).astype(int)), axis=0)
    y = np.vstack([np.full(6, np.nan), y])
    n, m = y.shape
    nodes, w = hermgauss(31)
    nodes *= np.sqrt(2)
    w /= np.sqrt(np.pi)
    start = [*np.zeros(m)]
    bounds = [(None, None)] * m
    if model != "rasch":
        k = 1 if model == "1pl" else m
        start.extend([1.0] * k)
        bounds.extend([(1e-6, None)] * k)
    if model in ("3pl", "4pl"):
        start.extend([0.1] * m)
        bounds.extend([(1e-6, 1 - 1e-6)] * m)
    if model == "4pl":
        start.extend([0.9] * m)
        bounds.extend([(1e-6, 1 - 1e-6)] * m)

    def unpack(v):
        b = v[:m]
        offset = m
        if model == "rasch":
            a = np.ones(m)
        elif model == "1pl":
            a = np.full(m, v[offset])
            offset += 1
        else:
            a = v[offset : offset + m]
            offset += m
        c = v[offset : offset + m] if model in ("3pl", "4pl") else np.zeros(m)
        u = v[offset + m : offset + 2 * m] if model == "4pl" else np.ones(m)
        return a, b, c, u

    def loss(v):
        a, b, c, u = unpack(v)
        z = a * (nodes[:, None] - b)
        yes = c * expit(-z) + u * expit(z)
        no = (1 - c) * expit(-z) + (1 - u) * expit(z)
        joint = (
            (y == 1) @ np.log(np.maximum(yes, 1e-300)).T
            + (y == 0) @ np.log(np.maximum(no, 1e-300)).T
            + np.log(w)
        )
        return -logsumexp(joint, axis=1).mean()

    constraints = []
    if model == "4pl":
        mat = np.zeros((m, 4 * m))
        mat[:, 2 * m : 3 * m] = -np.eye(m)
        mat[:, 3 * m :] = np.eye(m)
        constraints = [LinearConstraint(mat, 1e-6, np.inf)]
    ref = minimize(
        loss,
        start,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    actual = logistic_irt(y, model=model, quadrature_points=31, tolerance=1e-10)
    assert ref.success, ref.message
    assert actual.converged, actual.message
    np.testing.assert_allclose(actual.log_likelihood, -ref.fun * n, atol=2e-5)
    for key, expected in zip(("a", "b", "c", "u"), unpack(ref.x)):
        np.testing.assert_allclose(actual.means[key], expected, atol=3e-3, rtol=3e-3)
    np.testing.assert_allclose(actual.means["x"][0], 0, atol=1e-12)
    np.testing.assert_allclose(actual.variances["x"][0], 1, atol=1e-12)
    np.testing.assert_allclose(actual.predict(), actual.probabilities)
    assert np.all(np.isfinite(actual.trace))


@pytest.mark.parametrize("model", ["rasch", "1pl", "2pl", "3pl", "4pl"])
def test_jit_agreement_and_iteration_limit(model):
    kw = dict(model=model, max_iter=2, quadrature_points=11)
    compiled = logistic_irt(data(), **kw)
    eager = logistic_irt(data(), jit=False, **kw)
    assert not compiled.converged
    assert compiled.iterations == 2
    for key in compiled.means:
        np.testing.assert_allclose(compiled.means[key], eager.means[key], atol=1e-10)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": "unknown"},
        {"max_iter": 0},
        {"max_iter": True},
        {"quadrature_points": 2},
        {"tolerance": 0},
        {"tolerance": np.nan},
        {"jit": 1},
        {"starts": {"a": -1}},
        {"starts": {"b": [1, 2]}},
        {"model": "rasch", "starts": {"a": 2}},
        {"model": "4pl", "starts": {"c": 0.8, "u": 0.2}},
    ],
)
def test_validation(kwargs):
    with pytest.raises(ValueError):
        logistic_irt(data(), **kwargs)


def test_missing_and_invalid_codes():
    y = data()
    y[:, 0] = np.nan
    with pytest.raises(ValueError, match="Each item"):
        logistic_irt(y)
    y = data()
    y[1, 0] = -1
    with pytest.raises(ValueError, match="0/1/NaN"):
        logistic_irt(y)
