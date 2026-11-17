"""Typed mappings accepted by the model functions.

Arrays accept NumPy arrays, JAX arrays or nested numeric sequences. Shapes are
validated at runtime by the model entry points.
"""

from typing import Any, TypedDict

ArrayLike = Any


class LogisticStarts(TypedDict, total=False):
    """Optional free item parameters; only keys estimated by the model are valid."""

    a: ArrayLike
    b: ArrayLike
    c: ArrayLike
    u: ArrayLike


class BinaryStarts(TypedDict):
    alpha: ArrayLike
    beta: ArrayLike
    x: ArrayLike


class BinaryPriors(TypedDict):
    x_mu: float
    x_variance: float
    beta_mu: ArrayLike
    beta_covariance: ArrayLike


class GaussianPrior(TypedDict):
    mu: ArrayLike
    sigma: ArrayLike


class OrdinalStarts(TypedDict):
    tau: ArrayLike
    beta: ArrayLike
    x: ArrayLike
    DD: ArrayLike


class OrdinalPriors(TypedDict):
    x: GaussianPrior
    beta: GaussianPrior


class DynamicPriors(TypedDict):
    x_mu0: ArrayLike
    x_variance0: ArrayLike
    omega2: ArrayLike
    beta_mu: ArrayLike
    beta_covariance: ArrayLike


class HierarchicalStarts(TypedDict):
    alpha: ArrayLike
    beta: ArrayLike
    gamma: ArrayLike
    eta: ArrayLike
    sigma: ArrayLike


class HierarchicalPriors(TypedDict):
    beta_mu: ArrayLike
    beta_covariance: ArrayLike
    gamma_mu: ArrayLike
    gamma_covariance: ArrayLike
    sigma_v: float
    sigma_s: float


class NetworkStarts(TypedDict):
    alpha: ArrayLike
    beta: ArrayLike
    w: ArrayLike
    theta: ArrayLike
    gamma: ArrayLike


class NetworkPriors(TypedDict):
    alpha: GaussianPrior
    beta: GaussianPrior
    w: GaussianPrior
    theta: GaussianPrior
    gamma: GaussianPrior


class PoissonStarts(TypedDict):
    alpha: ArrayLike
    psi: ArrayLike
    beta: ArrayLike
    x: ArrayLike


class ScalarPrior(TypedDict):
    mu: float
    sigma2: float


class PoissonPriors(TypedDict):
    alpha: ScalarPrior
    psi: ScalarPrior
    beta: ScalarPrior
    x: ScalarPrior
