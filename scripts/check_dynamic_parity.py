import json
from pathlib import Path
import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
from jirt import dyn_irt


def decode(v):
    if isinstance(v, dict) and "values" in v:
        a = np.asarray(v["values"])
        return a.reshape(v["shape"], order="F") if v["shape"] else a
    return {k: decode(x) for k, x in v.items()}


source = json.loads(Path("outputs/reference-data/mq_data.rda.json").read_text())[
    "mq_data"
]
d = decode(source["data.mq"])
s = decode(source["cur.mq"])
p = decode(source["priors.mq"])
p = {
    dict(
        **{
            "x.mu0": "x_mu0",
            "x.sigma0": "x_variance0",
            "beta.mu": "beta_mu",
            "beta.sigma": "beta_covariance",
        }
    ).get(k, k): v
    for k, v in p.items()
}
r = dyn_irt(
    d["rc"],
    starts=s,
    priors=p,
    start_period=d["startlegis"],
    end_period=d["endlegis"],
    item_period=d["bill.session"],
)
report = dict(
    shape=list(d["rc"].shape),
    iterations=r.iterations,
    converged=r.converged,
    finite=all(np.isfinite(v).all() for v in r.means.values()),
)
Path("outputs/dynamic-real-report.json").write_text(json.dumps(report, indent=2) + "\n")
print(report)

ref = json.loads(Path("outputs/baselines/dynamic.json").read_text())
for group, actual in [("means", r.means), ("vars", r.variances)]:
    for key, value in actual.items():
        np.testing.assert_allclose(
            value,
            np.asarray(ref[group][key]).reshape(value.shape),
            atol=1e-8,
            rtol=1e-6,
        )
assert r.iterations == ref["runtime"]["iters"]
assert r.converged == bool(ref["runtime"]["conv"])
assert all(np.isfinite(v).all() for v in r.means.values())
print("Dynamic parity passed")
