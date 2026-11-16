# jirt

Item response theory in Python and JAX, with binary, ordinal, dynamic,
hierarchical, network and Poisson models.

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

| Function | Input | Main parameters |
| --- | --- | --- |
| `bin_irt` | Subject × item votes, -1/1; 0/9 missing | x, beta (intercept and slope) |
| `ord_irt` | Subject × item categories 1/2/3; 0 missing | x, beta, tau, Delta |
| `dyn_irt` | Votes with item periods and inclusive subject tenures | x by subject and period, alpha, beta |
| `hier_irt` | Indexed votes, subject groups and covariates | gamma, eta, implied x, alpha, beta |
| `network_irt` | Subject × item edges, 0/1; NaN missing | alpha, beta, w, theta |
| `pois_irt` | Term × document nonnegative counts | alpha, psi, beta, x |

All indices start at zero. Supply starts and study-specific priors explicitly;
function docstrings describe their shapes. Typed input mappings are available in
`jirt.types`. Run `uv run python examples/all_models.py` for self-contained
examples of all six models. `Control` selects the iteration limit,
tolerance, JIT and correlation or absolute convergence criterion. Check
`result.converged`: reaching the iteration limit returns the last finite estimate.
`FitResult` includes means, variances, iteration count, convergence trace and
probabilities (None for Poisson). Hierarchical covariance arrays put group/item
first. Dynamic estimates outside active tenure are zero and probabilities are NaN.

`convert_votes` handles explicit recoding. `get_starts`, `network_starts` and
`make_priors` supply initialization helpers. `bootstrap` supports binary, ordinal,
dynamic and hierarchical fits with explicit random keys, replicate estimates,
standard errors and failure diagnostics.

## Validation status

Small independent numerical fixtures exercise fixed updates with JIT both enabled
and disabled. Full-data checks live in `scripts/check_*_parity.py`; their local
inputs and reports are stored under `outputs/` and are not needed for unit tests.
All six full-data checks pass at `atol=1e-8, rtol=1e-6`, including parameter,
variance, iteration-count and convergence-status comparisons. The test suite
contains 60 passing tests. Network fits reach the 500-iteration limit without
convergence; their final estimates pass the comparison.

See [API.md](API.md) for all parameter shapes and [VALIDATION.md](VALIDATION.md)
for dataset coverage and reproducible validation commands.
