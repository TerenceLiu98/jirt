"""Smoke-test binary fits on the bundled judicial roll-call data.

Run scripts/export_reference.R first, then uv run python scripts/check_real_binary.py.
This checks execution on real observations, not parity with a dynamic model.
"""

import json
from pathlib import Path
import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
from jirt import bin_irt, Control

raw = json.loads(Path("outputs/reference-data/mq_data.rda.json").read_text())
a = raw["mq_data"]["data.mq"]["rc"]
y = np.array(a["values"]).reshape(a["shape"], order="F")
starts = dict(
    alpha=np.zeros(y.shape[1]),
    beta=np.zeros(y.shape[1]),
    x=np.random.default_rng(42).normal(size=y.shape[0]),
)
report = {}
for method in ("em", "variational"):
    result = bin_irt(y, starts=starts, method=method, control=Control(max_iter=500))
    observed = np.abs(y) == 1
    prob = np.asarray(result.probabilities)
    assert np.isfinite(prob).all()
    assert np.all((prob >= 0) & (prob <= 1))
    report[method] = dict(
        shape=list(y.shape),
        observed=int(observed.sum()),
        iterations=result.iterations,
        converged=result.converged,
        accuracy=float(np.mean((prob[observed] > 0.5) == (y[observed] == 1))),
    )
Path("outputs/binary-real-report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
