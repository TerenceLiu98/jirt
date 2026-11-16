import json
from pathlib import Path
import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
from jirt import bin_irt

raw = json.loads(Path("outputs/baselines/binary.json").read_text())
for method in ("em", "variational"):
    result = bin_irt(raw["votes"], starts=raw["starts"], method=method)
    ref = raw[method]
    for group, actual in [("means", result.means), ("vars", result.variances)]:
        for key, value in actual.items():
            expected = np.asarray(ref[group][key]).reshape(value.shape)
            np.testing.assert_allclose(value, expected, atol=1e-8, rtol=1e-6)
    assert result.iterations == ref["runtime"]["iters"]
    assert result.converged == bool(ref["runtime"]["conv"])
    assert all(np.isfinite(v).all() for v in result.means.values())
    print(method, result.iterations, "parity passed")
