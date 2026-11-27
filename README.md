# JIRT: Item Response Theory Models in JAX

[![Tests](https://github.com/TerenceLiu98/jirt/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/TerenceLiu98/jirt/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/python-%3E%3D3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](https://github.com/TerenceLiu98/jirt/blob/main/LICENSE)
[![JAX](https://img.shields.io/badge/powered%20by-JAX-orange)](https://docs.jax.dev/)
[![uv](https://img.shields.io/badge/managed%20with-uv-DE5FE9)](https://docs.astral.sh/uv/)

Item response theory in Python and JAX, with binary, ordinal, dynamic,
hierarchical, network and Poisson models, plus logistic Rasch/1PL–4PL calibration.

## Installation

Use Python 3.12 or later on a platform with compatible
[JAX wheels](https://docs.jax.dev/en/latest/installation.html#supported-platforms).
On macOS, the required JAX version provides wheels for Apple Silicon.

```sh
python -m pip install jirt
```

## Quick start

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

For execution modes, optional device-side optimizers, and stopping rules, see
[Performance](https://github.com/TerenceLiu98/jirt/blob/main/PERFORMANCE.md).

## Models

| Function | Formulas | Input | Main parameters |
| --- | --- | --- | --- |
| `amortized_irt` | $P(Y_{ij}=1)=\mathrm{sigmoid}(\theta_i-f_\phi(\mathbf x_j))$ | Binary responses; NaN missing; aligned item features and explicit random key | Shared encoder, content-only difficulty, EAP ability; affine or tanh network |
| `diagnostic_irt` | $P(Y_{ij}=1)=g_j+(1-s_j-g_j)\eta_j(\alpha_i)$; DINA requires all skills, DINO any required skill | Binary responses; NaN missing; item × attribute Q matrix | `slip`, `guess`, profile weights, posterior profile and mastery probabilities |
| `group_irt` | $P(Y_{ij}\ge k\mid g_i)=\mathrm{sigmoid}(a_{g_i j}(\theta_i-b_{g_i jk}))$; $\theta_i\mid g_i\sim N(\mu_{g_i},\sigma_{g_i}^2)$ | Binary or ordered categories; NaN missing; observed group labels and sharing masks | Group `a`, `steps`, `mu`, `sigma`, EAP `x`; Rasch, 2PL or GRM |
| `mixture_irt` | $P(\mathbf y_i)=\sum_c\pi_c P_c(\mathbf y_i)$; optional structural component $P_0(\mathbf y_i)=\mathbf1[\text{all observed responses}=0]$ | Binary or ordered categories; NaN missing; class count and starts | Class parameters, weights, posterior memberships; optional zero inflation |
| `ggum_irt` | $P(Y_{ij}=k)\propto e^{a_j\sum_{h=1}^k\tau_{jh}}\cosh((K_j-\frac12-k)a_j(\theta_i-\delta_j))$ | Binary or ordered 0-based categories; NaN missing; declared counts and starts | `a`, ideal locations `delta`, thresholds `tau`, EAP `x` |
| `noncompensatory_irt` | $P(Y_{ij}=1)=c_j+(1-c_j)\prod_{h\in A_j}\mathrm{sigmoid}(a_{jh}\theta_{ih}+d_{jh})$ | Binary responses; NaN missing; explicit loading mask | `a`, `d`, `c`, EAP `x`; PC1PL, PC2PL or PC3PL |
| `sequential_irt` | $P(Y_{ij}=k)=(1-q_{jk})\prod_{h<k}q_{jh}$, $q_{jh}=\mathrm{sigmoid}(a_j(\theta_i-b_{jh}))$; final category passes every stage | 0-based ordered categories; NaN missing; declared category counts | `a`, `steps`, EAP `x`; sequential or unit-slope Tutz |
| `nominal_irt` | Nominal: $P(Y_{ij}=k)=\mathrm{softmax}_k(s_{jk}\theta_i+t_{jk})$; nested: key probability $q_j$, distractor mass $1-q_j$ | Unordered 0-based categories; NaN missing; counts and optional answer keys | `slopes`, `intercepts`, EAP `x`; nested `a`, `b`, `c`, `u` |
| `polytomous_irt` | GRM: $P(Y_{ij}\ge k)=\mathrm{sigmoid}(a_j(\theta_i-b_{jk}))$; GPCM: $P(Y_{ij}=k)\propto\exp(a_j\sum_{h=1}^k(\theta_i-b_{jh}))$ | 0-based categories; NaN missing; declared category counts | `a`, `steps`, EAP `x`; GRM, PCM, GPCM or RSM |
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

`noncompensatory_irt` models success as a product across required dimensions.
`sequential_irt` models ordered responses as successive stage decisions. Run
`uv run python examples/response_models.py` for both mechanisms.

`ggum_irt` models agreement peaking near an item’s ideal location. Run
`uv run python examples/unfolding.py` for an attitude-response example with
multiple starting values.

`group_irt` fits known populations; `mixture_irt` infers latent classes and
optionally a structural all-lowest-response class. Run
`uv run python examples/groups.py` for shared anchors, multiple starts and
zero inflation.

`diagnostic_irt` estimates discrete skill mastery under DINA or DINO. Run
`uv run python examples/diagnostic.py` for a reproducible diagnostic example.

`amortized_irt` learns difficulty from aligned local item features and predicts
unseen items without responses. Run `uv run python examples/amortized.py` for
an entire-item holdout comparison against a non-amortized 1PL baseline.

See the [API reference](https://github.com/TerenceLiu98/jirt/blob/main/API.md)
for parameter shapes and identification constraints.

## Development

From the repository checkout, use uv to create an isolated `.venv` with the
dependencies pinned in `uv.lock`:

```sh
uv sync --locked
uv run pytest
uv build
```

## TODO

- [x] Logistic IRT: Rasch/1PL, 2PL, 3PL and 4PL.
- [x] Multidimensional IRT, including bifactor and testlet models.
- [x] General polytomous IRT: graded response, partial credit and rating scale models.
- [x] Nominal and nested response models.
- [x] Noncompensatory and sequential response models.
- [x] Unfolding models: unidimensional GGUM for binary and polytomous responses.
- [x] Multiple-group and mixture IRT, including zero-inflated binary/graded responses.
- [x] Cognitive diagnosis models: DINA and DINO.
- [x] Amortized 1PL using item content.
