# JIRT: Item Response Theory Models in JAX

[![Tests](https://github.com/TerenceLiu98/jirt/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/TerenceLiu98/jirt/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/python-%3E%3D3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](LICENSE)
[![JAX](https://img.shields.io/badge/powered%20by-JAX-orange)](https://docs.jax.dev/)
[![uv](https://img.shields.io/badge/managed%20with-uv-DE5FE9)](https://docs.astral.sh/uv/)

Item response theory in Python and JAX, with binary, ordinal, dynamic,
hierarchical, network and Poisson models, plus logistic Rasch/1PL–4PL calibration.

## Environment

Python 3.12, managed by uv:

```sh
uv sync --locked
uv run pytest
uv build
```

Dependencies are pinned in `uv.lock`; uv creates an isolated `.venv`.
Numerical fitting requires 64-bit JAX. Enable it before creating arrays:

```python
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from jirt import bin_irt, get_starts, make_priors, Control

votes = np.array([[1, -1, 0], [-1, 1, 1], [1, 1, -1], [-1, -1, 1]])
result = bin_irt(
    votes,
    starts=get_starts(4, 3, key=jax.random.key(42)),
    priors=make_priors(),
    control=Control(max_iter=500),
)
print(result.means["x"], result.converged)
```

## Models

| Function | Formulas | Input | Main parameters |
| --- | --- | --- | --- |
| `multidimensional_irt` | $P(Y_{ij}=1)=\mathrm{sigmoid}(a_j^T\theta_i+d_j)$ | Subject × item 0/1 responses; NaN missing; loading mask or testlet groups | `a`, `d`, EAP `x`; multidimensional 2PL, bifactor or testlet |
| `logistic_irt` | $P(Y_{ij}=1)=c_j+(u_j-c_j)\mathrm{sigmoid}(a_j(\theta_i-b_j))$ | Subject × item responses, 0/1; NaN missing | `a`, `b`, `c`, `u`, EAP `x`; Rasch, shared-slope 1PL, 2PL, 3PL or full 4PL |
| `bin_irt` | $P(Y_{ij}=1)=\Phi(\alpha_j+\beta_j x_i)$ | Subject × item votes, -1/1; 0/9 missing | `x`; output `beta` contains intercept $\alpha$ and slope $\beta$ |
| `ord_irt` | $\mu_{ij}=\tau_j+\beta_j x_i$; $P(Y_{ij}\le k)=\Phi((c_k-\mu_{ij})\Delta_j)$, $c_1=0,c_2=1$ | Subject × item categories 1/2/3; 0 missing | `x`, `beta`, `tau`, `Delta`, `Delta_sq` |
| `dyn_irt` | $P(Y_{ij}=1)=\Phi(\alpha_j+\beta_j x_{i,t_j})$; $x_{i,t}=x_{i,t-1}+\epsilon_{i,t}$, $\epsilon_{i,t}\sim N(0,\omega_i^2)$ | Subject × item votes, -1/1; 0 missing; item periods and inclusive subject tenures | `x` by subject and period, `alpha`, `beta`; fixed diffusion variance `omega2` |
| `hier_irt` | $P(Y_{ij}=1)=\Phi(\alpha_j+\beta_j x_i)$; $x_i=z_i^\top\gamma_{g_i}+\eta_i$ | Indexed votes, -1/1; 0 missing; subject groups and covariates | `gamma`, `eta`, `sigma` (residual variance), `x_implied`, `alpha`, `beta` |
| `network_irt` | $P(Y_{ij}=1)=\Phi(\alpha_j+\beta_i-\gamma(\theta_i-w_j)^2)$ | Subject × item edges, 0/1; NaN missing | `alpha`, `beta`, `w`, `theta`; fixed input `gamma` |
| `pois_irt` | $Y_{jk}\sim\mathrm{Poisson}(\lambda_{jk})$; $\log\lambda_{jk}=\alpha_k+\psi_j+\beta_j x_{s(k)}$ | Term × document nonnegative counts; optional document-to-subject mapping | `alpha`, `psi`, `beta`, `x` |

Here $\Phi$ is the standard normal cumulative distribution function. These are
conditional response formulas given the model parameters. In the non-Poisson
rows, $i$ indexes subjects and $j$ items; $t_j$ is the period of item $j$,
$g_i$ is subject $i$'s group, and $z_i$ is their covariate vector. For the ordinal
model, $k=1,2$ indexes the finite category boundaries and $\Delta_j>0$ is the
inverse residual standard deviation. In the Poisson row, $j$ indexes terms,
$k$ documents, and $s(k)$ maps a document to its subject (for example, its author).

All indices start at zero. For the six original estimators, supply starts and
study-specific priors explicitly;
function docstrings describe their shapes. Typed input mappings are available in
`jirt.types`. Run `uv run python examples/all_models.py` for self-contained
examples of those six models. `Control` selects the iteration limit,
tolerance, JIT and correlation or absolute convergence criterion. Check
`result.converged`: reaching the iteration limit returns the last finite estimate.
`FitResult` includes means, variances, iteration count, convergence trace and
probabilities (None for Poisson). Hierarchical covariance arrays put group/item
first. Dynamic estimates outside active tenure are zero and probabilities are NaN.

`convert_votes` handles explicit recoding. `get_starts`, `network_starts` and
`make_priors` supply initialization helpers. `bootstrap` supports binary, ordinal,
dynamic and hierarchical fits with explicit random keys, replicate estimates,
standard errors and failure diagnostics.

`logistic_irt` uses standard-normal marginal maximum likelihood and returns a
`LogisticResult` with item parameters, EAP person scores, conditional person
variances and optimizer diagnostics. Its controls and response coding differ
from the probit estimators. Run `uv run python examples/logistic.py` for the
logistic variants.

`multidimensional_irt` supports confirmatory loading masks, bifactor loadings
and shared testlet effects under a standard multivariate normal ability
distribution. Run `uv run python examples/multidimensional.py` for examples.

See [API.md](API.md) for parameter shapes and identification constraints.

## TODO

- [x] Logistic IRT: Rasch/1PL, 2PL, 3PL and 4PL.
- [x] Multidimensional IRT, including bifactor and testlet models.
- [ ] General polytomous IRT: graded response, partial credit and rating scale models.
- [ ] Nominal and nested response models.
- [ ] Noncompensatory and sequential response models.
- [ ] Unfolding models, including GGUM.
- [ ] Multiple-group and mixture IRT, including zero-inflated response models.
- [ ] Cognitive diagnosis models, including DINA and DINO.
- [ ] Amortized IRT using item content.
