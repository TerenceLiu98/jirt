"""Independent SciPy direct marginal optimization, without JAX or EM updates."""

import itertools
import jax
import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from jirt.diagnostic import diagnostic_irt

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("model", ["dina", "dino"])
def test_independent_scipy_fit(model):
    q = np.array([[1, 0], [0, 1], [1, 1]] * 2)
    profiles = np.array(list(itertools.product([0, 1], repeat=2)))
    ideal = np.array(
        [
            [
                all(alpha[qj == 1]) if model == "dina" else any(alpha[qj == 1])
                for qj in q
            ]
            for alpha in profiles
        ]
    )
    truth = np.where(ideal, 0.88, 0.16)
    patterns = np.array(list(itertools.product([0, 1], repeat=6)))
    probability = np.prod(
        np.where(patterns[:, None, :] == 1, truth[None, :, :], 1 - truth[None, :, :]),
        axis=-1,
    )
    freq = np.maximum(
        1, np.round(probability @ np.array([0.15, 0.25, 0.3, 0.3]) * 100000).astype(int)
    )
    # Partial missing and fully missing patterns have independent known frequencies.
    patterns = np.vstack([patterns, [-1, 0, 1, -1, 0, 1], [-1] * 6])
    freq = np.r_[freq, 15, 7]
    yes = patterns == 1
    no = patterns == 0

    def evaluate(raw, return_post=False):
        slip = raw[:6]
        guess = raw[6:12]
        weights = softmax(np.r_[0, raw[12:]])
        p = np.where(ideal, 1 - slip, guess)
        joint = yes @ np.log(p).T + no @ np.log1p(-p).T + np.log(weights)
        ll = logsumexp(joint, axis=1)
        if return_post:
            return ll, np.exp(joint - ll[:, None]), weights
        return -ll @ freq / freq.sum()

    initial = np.r_[np.full(6, 0.2), np.full(6, 0.2), np.zeros(3)]
    opt = minimize(
        evaluate,
        initial,
        method="SLSQP",
        bounds=[(1e-6, 1 - 1e-6)] * 12 + [(None, None)] * 3,
        constraints=[dict(type="ineq", fun=lambda r: 1 - r[:6] - r[6:12])],
        options=dict(ftol=1e-13, maxiter=2000),
    )
    assert opt.success, opt.message
    y = np.repeat(patterns, freq, axis=0).astype(float)
    y[y < 0] = np.nan
    fit = diagnostic_irt(y, q_matrix=q, model=model, tolerance=1e-13, max_iter=5000)
    assert fit.converged
    ll, posterior, weights = evaluate(opt.x, True)
    np.testing.assert_allclose(fit.log_likelihood, ll @ freq, atol=1e-5, rtol=0)
    np.testing.assert_allclose(fit.slip, opt.x[:6], atol=2e-5, rtol=0)
    np.testing.assert_allclose(fit.guess, opt.x[6:12], atol=2e-5, rtol=0)
    np.testing.assert_allclose(fit.weights, weights, atol=2e-5, rtol=0)
    first = np.r_[0, np.cumsum(freq)[:-1]]
    np.testing.assert_allclose(
        np.asarray(fit.profile_probabilities)[first], posterior, atol=3e-5, rtol=0
    )
