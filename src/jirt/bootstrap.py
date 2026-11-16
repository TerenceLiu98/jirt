"""Parametric bootstrap with explicit replicate diagnostics."""

from dataclasses import dataclass
import jax
import numpy as np
from .binary import bin_irt, FitResult
from .ordinal import ord_irt
from .dynamic import dyn_irt
from .hierarchical import hier_irt


@dataclass
class BootstrapResult:
    standard_errors: np.ndarray
    estimates: np.ndarray
    converged: np.ndarray
    failures: dict[int, str]


def bootstrap(
    fit: FitResult, data, *, model, fit_kwargs, key, n_trials=50, pseudo_data=None
):
    """Refit binary/ordinal/dynamic/hierarchical parametric replicates.

    ``fit_kwargs`` supplies the original starts, priors, controls and index
    arrays. Optional pseudo_data provides identical replicates for cross-language
    validation. Failed fits remain NaN and are listed; finite unconverged fits
    are retained with their status. Standard errors use sample SD (ddof=1).
    """
    functions = {
        "binary": bin_irt,
        "ordinal": ord_irt,
        "dynamic": dyn_irt,
        "hierarchical": hier_irt,
    }
    if model not in functions:
        raise ValueError("Bootstrap is supported for the four IRT models")
    if not isinstance(n_trials, int) or n_trials < 2:
        raise ValueError("n_trials must be an integer >= 2")
    y = np.asarray(data)
    prob = np.asarray(fit.probabilities)
    if model == "hierarchical":
        y = y.reshape(-1)
    if prob.shape != (y.shape + (3,) if model == "ordinal" else y.shape):
        raise ValueError("Data and fitted probabilities have incompatible shapes")
    observed = (y != 0) if model == "ordinal" else (np.abs(y) == 1)
    if model == "dynamic":
        first = np.asarray(fit_kwargs["start_period"]).reshape(-1)
        last = np.asarray(fit_kwargs["end_period"]).reshape(-1)
        periods = np.asarray(fit_kwargs["item_period"]).reshape(-1)
        observed &= (periods[None, :] >= first[:, None]) & (
            periods[None, :] <= last[:, None]
        )
    valid_prob = prob[observed]
    if not np.isfinite(valid_prob).all() or ((valid_prob < 0) | (valid_prob > 1)).any():
        raise ValueError("Observed responses require finite probabilities in [0,1]")
    if model == "ordinal" and not np.allclose(valid_prob.sum(axis=-1), 1):
        raise ValueError("Ordinal category probabilities must sum to one")
    target = "x_implied" if model == "hierarchical" else "x"
    base = np.asarray(fit.means[target])
    estimates = np.full((n_trials,) + base.shape, np.nan)
    statuses = np.zeros(n_trials, dtype=bool)
    failures = {}
    if pseudo_data is not None:
        pseudo_data = np.asarray(pseudo_data)
        if pseudo_data.shape != (n_trials,) + y.shape:
            raise ValueError("Invalid pseudo_data shape")
    keys = jax.random.split(key, n_trials)
    for trial in range(n_trials):
        if pseudo_data is not None:
            sample = pseudo_data[trial]
        else:
            u = np.asarray(jax.random.uniform(keys[trial], y.shape))
            if model == "ordinal":
                sample = (
                    1
                    + (u > prob[..., 0]).astype(int)
                    + (u > prob[..., :2].sum(axis=-1)).astype(int)
                )
                sample = np.where(y == 0, 0, sample)
            else:
                sample = np.where(u < prob, 1, -1)
                sample = np.where(observed, sample, y)
        try:
            result = functions[model](sample, **fit_kwargs)
            value = np.asarray(result.means[target])
            if not np.isfinite(value).all():
                raise FloatingPointError("Nonfinite bootstrap estimates")
            estimates[trial] = value
            statuses[trial] = result.converged
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            failures[trial] = str(exc)
    valid = np.isfinite(estimates).all(axis=tuple(range(1, estimates.ndim)))
    errors = (
        np.std(estimates[valid], axis=0, ddof=1)
        if valid.sum() >= 2
        else np.full(base.shape, np.nan)
    )
    if model == "dynamic":
        start = np.asarray(fit_kwargs["start_period"]).reshape(-1)
        end = np.asarray(fit_kwargs["end_period"]).reshape(-1)
        active = (np.arange(base.shape[1])[None, :] >= start[:, None]) & (
            np.arange(base.shape[1])[None, :] <= end[:, None]
        )
        errors = np.where(active, errors, np.nan)
    return BootstrapResult(errors, estimates, statuses, failures)
