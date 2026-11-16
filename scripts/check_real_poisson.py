import json
from pathlib import Path
import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
from jirt import pois_irt


def decode(v):
    if isinstance(v, dict) and "values" in v:
        a = np.asarray(v["values"])
        return a.reshape(v["shape"], order="F") if v["shape"] else a
    return {k: decode(x) for k, x in v.items()}


raw = decode(json.loads(Path("outputs/reference-data/manifesto.rda.json").read_text()))[
    "manifesto"
]
r = pois_irt(raw["data.manif"], starts=raw["starts.manif"], priors=raw["priors.manif"])
report = dict(
    shape=list(raw["data.manif"].shape),
    iterations=r.iterations,
    converged=r.converged,
    finite=all(np.isfinite(v).all() for v in r.means.values()),
)
Path("outputs/poisson-real-report.json").write_text(json.dumps(report, indent=2) + "\n")
print(report)
