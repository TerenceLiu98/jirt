import json
from pathlib import Path
import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
from jirt import network_irt, Control


def decode(v):
    if isinstance(v, dict) and "values" in v:
        a = np.asarray(v["values"])
        return a.reshape(v["shape"], order="F") if v["shape"] else a
    return {k: decode(x) for k, x in v.items()}


r = decode(json.loads(Path("outputs/reference-data/ustweet.rda.json").read_text()))[
    "ustweet"
]
fit = network_irt(
    r["data"],
    starts=r["starts"],
    priors=r["priors"],
    control=Control(max_iter=500, convergence="absolute"),
)
report = dict(
    shape=list(r["data"].shape),
    iterations=fit.iterations,
    converged=fit.converged,
    finite=all(np.isfinite(v).all() for v in fit.means.values()),
    positive_variances=all((v > 0).all().item() for v in fit.variances.values()),
)
Path("outputs/network-real-report.json").write_text(json.dumps(report, indent=2) + "\n")
print(report)

ref = json.loads(Path("outputs/baselines/network.json").read_text())
for group, actual in [("means", fit.means), ("vars", fit.variances)]:
    for key, value in actual.items():
        np.testing.assert_allclose(
            value,
            np.asarray(ref[group][key]).reshape(value.shape),
            atol=1e-8,
            rtol=1e-6,
        )
assert fit.iterations == ref["runtime"]["iters"]
assert fit.converged == bool(ref["runtime"]["conv"])
assert all(np.isfinite(v).all() for v in fit.means.values())
print("network parity passed")
