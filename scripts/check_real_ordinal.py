import json
from pathlib import Path
import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
from jirt import ord_irt


def decode(v):
    if isinstance(v, dict) and "values" in v:
        a = np.asarray(v["values"])
        return a.reshape(v["shape"], order="F") if v["shape"] else a
    return {k: decode(x) for k, x in v.items()}


source = json.loads(Path("outputs/reference-data/AsahiTodai.rda.json").read_text())[
    "AsahiTodai"
]
raw = {k: decode(source[k]) for k in ("dat.all", "start.values", "priors")}
r = ord_irt(raw["dat.all"], starts=raw["start.values"], priors=raw["priors"])
report = dict(
    shape=list(raw["dat.all"].shape),
    iterations=r.iterations,
    converged=r.converged,
    finite=all(np.isfinite(v).all() for v in r.means.values()),
)
Path("outputs/ordinal-real-report.json").write_text(json.dumps(report, indent=2) + "\n")
print(report)
