import json
from pathlib import Path
import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
from jirt import hier_irt


def decode(v):
    if isinstance(v, dict) and "values" in v:
        a = np.asarray(v["values"])
        return a.reshape(v["shape"], order="F") if v["shape"] else a
    return {k: decode(x) for k, x in v.items()}


source = json.loads(Path("outputs/reference-data/dwnom.rda.json").read_text())["dwnom"]
d = decode(source["data.in"])
s = decode(source["cur"])
p = decode(source["priors"])
rename = {
    "beta.mu": "beta_mu",
    "beta.sigma": "beta_covariance",
    "gamma.mu": "gamma_mu",
    "gamma.sigma": "gamma_covariance",
    "sigma.v": "sigma_v",
    "sigma.s": "sigma_s",
}
p = {rename[k]: v for k, v in p.items()}
r = hier_irt(
    d["y"],
    subject=d["i"],
    item=d["j"],
    groups=d["g"],
    covariates=d["z"],
    starts=s,
    priors=p,
)
report = dict(
    observations=d["y"].size,
    iterations=r.iterations,
    converged=r.converged,
    finite=all(np.isfinite(v).all() for v in r.means.values()),
)
Path("outputs/hierarchical-real-report.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(report)
