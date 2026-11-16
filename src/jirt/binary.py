"""One-dimensional probit IRT with EM or variational updates."""

from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr, ndtr
import numpy as np
from .types import BinaryStarts, BinaryPriors


@dataclass(frozen=True)
class Control:
    max_iter: int = 500
    tolerance: float = 1e-6
    jit: bool = True
    convergence: str = "correlation"

    def __post_init__(self):
        if (
            isinstance(self.max_iter, bool)
            or not isinstance(self.max_iter, (int, np.integer))
            or self.max_iter < 1
        ):
            raise ValueError("max_iter must be a positive integer")
        if (
            not np.isscalar(self.tolerance)
            or not np.isfinite(self.tolerance)
            or self.tolerance < 0
        ):
            raise ValueError("tolerance must be finite and nonnegative")
        if self.convergence not in ("absolute", "correlation"):
            raise ValueError("Unknown convergence criterion")
        if not isinstance(self.jit, bool):
            raise ValueError("jit must be boolean")


@dataclass
class FitResult:
    means: dict[str, jax.Array]
    variances: dict[str, jax.Array]
    converged: bool
    iterations: int
    trace: np.ndarray
    probabilities: jax.Array | None


def bin_irt(
    votes,
    *,
    starts: BinaryStarts,
    priors: BinaryPriors | None = None,
    control: Control | None = None,
    method="em",
    anchor_subject=None,
    anchor_outcomes=False,
):
    """Fit binary IRT. Votes are -1/1, with 0/9 denoting missing values.

    starts contains alpha (items), beta (items), x (subjects).
    priors contains x_mu, x_variance, beta_mu (2,), beta_covariance (2,2).
    Enable JAX x64 before calling. Missing observations retain their latent
    conditional expectations in the update, as in the reference algorithm.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable JAX x64 before fitting (JAX_ENABLE_X64=1).")
    c = control or Control()
    y = np.asarray(votes, dtype=float)
    if y.ndim != 2 or min(y.shape) < 2 or not np.isin(y, [-1, 0, 1, 9]).all():
        raise ValueError(
            "votes must be a matrix with at least two rows and columns and codes -1,0,1,9"
        )
    if method not in ("em", "variational") or c.convergence not in (
        "correlation",
        "absolute",
    ):
        raise ValueError("Invalid method or convergence criterion")
    if c.max_iter < 1 or not np.isfinite(c.tolerance) or c.tolerance < 0:
        raise ValueError("Invalid iteration controls")
    n, m = y.shape
    arrays = [
        np.asarray(starts[k], dtype=float).reshape(-1) for k in ("alpha", "beta", "x")
    ]
    if [a.size for a in arrays] != [m, m, n] or not all(
        np.isfinite(a).all() for a in arrays
    ):
        raise ValueError("Invalid starts")
    p = dict(
        x_mu=0.0, x_variance=1.0, beta_mu=np.zeros(2), beta_covariance=25 * np.eye(2)
    )
    p.update(priors or {})
    cov = np.asarray(p["beta_covariance"], dtype=float)
    mu = np.asarray(p["beta_mu"], dtype=float)
    if (
        cov.shape != (2, 2)
        or mu.shape != (2,)
        or not np.isfinite(cov).all()
        or not np.isfinite(mu).all()
        or not np.allclose(cov, cov.T)
        or np.linalg.eigvalsh(cov).min() <= 0
    ):
        raise ValueError("Invalid item prior")
    if (
        not np.isfinite(p["x_variance"])
        or p["x_variance"] <= 0
        or not np.isfinite(p["x_mu"])
    ):
        raise ValueError("Invalid subject prior")
    if anchor_subject is not None and (
        isinstance(anchor_subject, bool)
        or not isinstance(anchor_subject, (int, np.integer))
        or not 0 <= anchor_subject < n
    ):
        raise ValueError("Invalid anchor_subject")
    if anchor_subject is None and anchor_outcomes:
        observed = np.abs(y) == 1
        counts = observed.sum(axis=0)
        difficulty = np.divide(
            (y == -1).sum(axis=0), counts, out=np.zeros(m), where=counts > 0
        )
        denominator = observed @ difficulty
        scores = np.divide(
            (y == 1) @ difficulty,
            denominator,
            out=np.full(n, np.nan),
            where=denominator > 0,
        )
        if not np.isfinite(scores).any():
            raise ValueError(
                "Outcome anchoring requires informative observed responses"
            )
        anchor_subject = int(np.nanargmax(scores))
    precision = jnp.linalg.inv(jnp.asarray(cov))
    pm = precision @ jnp.asarray(mu)
    xp = 1 / p["x_variance"]
    yy = jnp.asarray(y)

    def expectation(state):
        a, b, x, vx, vb = state
        eta = a[None, :] + x[:, None] * b[None, :]
        sign = jnp.where(yy == -1, -1.0, 1.0)
        z = sign * eta
        ratio = jnp.exp(-0.5 * z * z - 0.5 * jnp.log(2 * jnp.pi) - log_ndtr(z))
        latent = eta + jnp.where(jnp.abs(yy) == 1, sign * ratio, 0.0)
        return latent

    def maximize(state, latent):
        a, b, x, vx, vb = state
        design = jnp.stack([jnp.ones(n), x], axis=1)
        moment = (design.T @ design).at[1, 1].add(n * vx)
        vb = jnp.linalg.inv(precision + moment)
        weights = (
            jnp.linalg.inv(precision + design.T @ design) if method == "em" else vb
        )
        ab = (latent.T @ design + pm) @ weights
        a, b = ab[:, 0], ab[:, 1]
        cross = b @ a + (m * vb[0, 1] if method == "variational" else 0.0)
        vx = 1 / (xp + b @ b + m * vb[1, 1])
        scale = 1 / (xp + b @ b) if method == "em" else vx
        x = scale * (xp * p["x_mu"] + latent @ b - cross)
        return a, b, x, vx, vb

    def step(state):
        return maximize(state, expectation(state))

    update = jax.jit(step) if c.jit else step
    state = (*map(jnp.asarray, arrays), jnp.asarray(p["x_variance"]), jnp.asarray(cov))
    history = []
    converged = False
    for iteration in range(1, c.max_iter + 1):
        old = state
        state = update(state)
        if not all(np.isfinite(np.asarray(a)).all() for a in state):
            raise FloatingPointError(f"Nonfinite update at iteration {iteration}")
        errors = []
        for before, after in zip(old[:3], state[:3]):
            u, v = np.asarray(before), np.asarray(after)
            if c.convergence == "absolute" or np.std(u) == 0 or np.std(v) == 0:
                error = float(np.max(np.abs(v - u)))
            else:
                error = float(max(0.0, 1 - np.corrcoef(u, v)[0, 1]))
            errors.append(error)
        history.append(errors)
        if iteration >= 2 and max(errors) < c.tolerance:
            converged = True
            # Reference finalization repeats maximization with the SAME latent expectations.
            state = maximize(state, expectation(old))
            break
    if not all(np.isfinite(np.asarray(v)).all() for v in state):
        raise FloatingPointError("Nonfinite finalization")
    a, b, x, vx, vb = state
    if anchor_subject is not None and x[anchor_subject] < jnp.mean(x):
        x, b = -x, -b
        vb = vb * jnp.array([[1.0, -1.0], [-1.0, 1.0]])
    return FitResult(
        dict(x=x, beta=jnp.stack([a, b], axis=1)),
        dict(x=vx, beta=vb),
        converged,
        iteration,
        np.asarray(history),
        ndtr(a[None, :] + x[:, None] * b[None, :]),
    )
