"""Discrete cognitive diagnosis with conjunctive and disjunctive response rules."""

from dataclasses import dataclass
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np


def mastery_profiles(attributes, *, max_profiles=65536):
    """Enumerate binary profiles in lexicographic order, last attribute fastest."""
    for value, name in ((attributes, "attributes"), (max_profiles, "max_profiles")):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    if 2**attributes > max_profiles:
        raise ValueError("Profile enumeration exceeds max_profiles")
    return np.asarray(list(product((0, 1), repeat=attributes)), dtype=int)


def _ideal(profiles, q_matrix, model):
    required = jnp.asarray(q_matrix, dtype=bool)
    mastered = jnp.asarray(profiles, dtype=bool)[:, None, :]
    if model == "dina":
        return jnp.all(~required[None, :, :] | mastered, axis=-1)
    return jnp.any(required[None, :, :] & mastered, axis=-1)


def diagnostic_probabilities(profiles, *, q_matrix, slip, guess, model="dina"):
    """Return profile-by-item success probabilities for valid binary inputs.

    DINA requires every listed skill; DINO requires at least one. Nonmasters
    succeed with guess, masters with 1-slip. This pure JAX kernel assumes a
    valid Q matrix, profiles and 0 <= guess <= 1-slip <= 1.
    """
    if model not in ("dina", "dino"):
        raise ValueError("model must be dina or dino")
    return jnp.where(
        _ideal(profiles, q_matrix, model), 1 - jnp.asarray(slip), jnp.asarray(guess)
    )


def _posterior(patterns, ideal, slip, guess, weights):
    log_yes = jnp.where(ideal, jnp.log1p(-slip), jnp.log(guess))
    log_no = jnp.where(ideal, jnp.log(slip), jnp.log1p(-guess))
    joint = (patterns == 1) @ log_yes.T + (patterns == 0) @ log_no.T + jnp.log(weights)
    return jax.scipy.special.logsumexp(joint, axis=1), jax.nn.softmax(joint, axis=1)


@dataclass
class DiagnosticResult:
    model: str
    q_matrix: np.ndarray
    profiles: np.ndarray
    slip: jax.Array
    guess: jax.Array
    weights: jax.Array
    profile_probabilities: jax.Array
    mastery_probabilities: jax.Array
    converged: bool
    iterations: int
    log_likelihood: float
    trace: np.ndarray

    @property
    def classifications(self):
        """Joint MAP profiles (not marginal mastery thresholding)."""
        return self.profiles[np.asarray(jnp.argmax(self.profile_probabilities, axis=1))]

    def predict(self, profiles=None):
        """Conditional curves if profiles supplied; posterior predictive otherwise."""
        if profiles is not None:
            return diagnostic_probabilities(
                profiles,
                q_matrix=self.q_matrix,
                slip=self.slip,
                guess=self.guess,
                model=self.model,
            )
        return self.profile_probabilities @ diagnostic_probabilities(
            self.profiles,
            q_matrix=self.q_matrix,
            slip=self.slip,
            guess=self.guess,
            model=self.model,
        )


def diagnostic_irt(
    responses,
    *,
    q_matrix,
    model="dina",
    starts=None,
    max_profiles=65536,
    max_iter=2000,
    tolerance=1e-9,
    jit=True,
):
    """Fit DINA/DINO by exact finite-profile marginal maximum likelihood EM.

    Binary responses use NaN missing. Q is binary (J,K), with at least one
    attribute per item and one item per attribute. Enumerate all 2**K profiles;
    labels follow the supplied Q columns, with no continuous ability scale.
    starts optionally supplies slip/guess (J,) and strictly positive weights
    (2**K,). EM enforces 1e-6 <= guess <= 1-slip <= 1-1e-6. Equality permits
    uninformative items. Q validation does not guarantee identification.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable JAX x64 before fitting")
    if model not in ("dina", "dino"):
        raise ValueError("model must be dina or dino")
    if (
        isinstance(max_iter, bool)
        or not isinstance(max_iter, (int, np.integer))
        or max_iter < 1
    ):
        raise ValueError("max_iter must be a positive integer")
    if not isinstance(jit, bool):
        raise ValueError("jit must be boolean")
    if not np.isscalar(tolerance) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    y = np.asarray(responses, dtype=float)
    if y.ndim != 2 or min(y.shape) < 2:
        raise ValueError("responses must be a matrix with at least two subjects/items")
    if not (np.isnan(y) | (y == 0) | (y == 1)).all():
        raise ValueError("responses must be binary or NaN")
    if np.any(np.all(np.isnan(y), axis=0)):
        raise ValueError("Every item needs observed responses")
    q = np.asarray(q_matrix)
    if (
        q.ndim != 2
        or q.shape[0] != y.shape[1]
        or q.shape[1] < 1
        or not ((q == 0) | (q == 1)).all()
        or not q.any(axis=0).all()
        or not q.any(axis=1).all()
    ):
        raise ValueError(
            "Q must be binary (items,attributes) with active rows and columns"
        )
    profiles = mastery_profiles(q.shape[1], max_profiles=max_profiles)
    ideal = _ideal(profiles, q, model)
    start = dict(starts or {})
    if set(start) - {"slip", "guess", "weights"}:
        raise ValueError("Unexpected starts")
    slip = np.asarray(start.get("slip", np.full(y.shape[1], 0.15)), float)
    guess = np.asarray(start.get("guess", np.full(y.shape[1], 0.2)), float)
    for value in (slip, guess):
        if (
            value.shape != (y.shape[1],)
            or not np.isfinite(value).all()
            or np.any((value < 1e-6) | (value > 1 - 1e-6))
        ):
            raise ValueError("Invalid slip/guess starts")
    if np.any(slip + guess > 1):
        raise ValueError("Require guess <= 1-slip")
    weights = np.asarray(
        start.get("weights", np.full(len(profiles), 1 / len(profiles))), float
    )
    if (
        weights.shape != (len(profiles),)
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
        or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-10)
    ):
        raise ValueError("Weights must be positive and sum to one")
    patterns, inverse, frequency = np.unique(
        np.where(np.isnan(y), -1, y).astype(int),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    patterns = jnp.asarray(patterns)
    frequency = jnp.asarray(frequency)
    yes = (patterns == 1).astype(float)
    observed = (patterns >= 0).astype(float)
    slip, guess, weights = map(jnp.asarray, (slip, guess, weights))
    score = jax.jit(_posterior) if jit else _posterior

    def step(slip, guess, weights):
        _, post = _posterior(patterns, ideal, slip, guess, weights)
        weighted = post * frequency[:, None]
        masters = weighted @ ideal.astype(float)
        nonmasters = weighted @ (~ideal).astype(float)
        mden = jnp.sum(masters * observed, axis=0)
        nden = jnp.sum(nonmasters * observed, axis=0)
        mnum = jnp.sum(masters * yes, axis=0)
        nnum = jnp.sum(nonmasters * yes, axis=0)
        high = jnp.where(mden > 0, mnum / jnp.maximum(mden, 1e-300), 1 - slip)
        low = jnp.where(nden > 0, nnum / jnp.maximum(nden, 1e-300), guess)
        # Two-bin isotonic Bernoulli MLE when the unconstrained rates cross.
        pooled = (mnum + nnum) / jnp.maximum(mden + nden, 1e-300)
        crossing = low > high
        low = jnp.clip(jnp.where(crossing, pooled, low), 1e-6, 1 - 1e-6)
        high = jnp.clip(jnp.where(crossing, pooled, high), 1e-6, 1 - 1e-6)
        weights = jnp.maximum(weighted.sum(axis=0), 1e-300)
        weights /= weights.sum()
        return 1 - high, low, weights

    update = jax.jit(step) if jit else step
    ll, post = score(patterns, ideal, slip, guess, weights)
    trace = [float(ll @ frequency)]
    converged = False
    for iteration in range(1, max_iter + 1):
        slip, guess, weights = update(slip, guess, weights)
        ll, post = score(patterns, ideal, slip, guess, weights)
        value = float(ll @ frequency)
        if not np.isfinite(value):
            raise FloatingPointError("Nonfinite diagnostic likelihood")
        trace.append(value)
        if abs(trace[-1] - trace[-2]) / len(y) < tolerance:
            converged = True
            break
    return DiagnosticResult(
        model,
        q.astype(int),
        profiles,
        slip,
        guess,
        weights,
        post[inverse],
        post[inverse] @ jnp.asarray(profiles),
        converged,
        iteration,
        trace[-1],
        np.asarray(trace),
    )
