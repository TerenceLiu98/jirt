"""Fixed five-update outputs generated independently by the R implementation."""

import json
from pathlib import Path
import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest
from jirt import bin_irt, ord_irt, network_irt, pois_irt, dyn_irt, hier_irt, Control

FUNCTIONS = {
    "binary_em": bin_irt,
    "binary_variational": bin_irt,
    "ordinal": ord_irt,
    "network": network_irt,
    "poisson": pois_irt,
    "dynamic": dyn_irt,
    "hierarchical": hier_irt,
    "hierarchical_real": hier_irt,
}


@pytest.mark.parametrize("name", FUNCTIONS)
@pytest.mark.parametrize("compiled", [False, True])
def test_fixed_updates(name, compiled):
    case = json.loads((Path(__file__).parent / "fixtures" / f"{name}.json").read_text())
    result = FUNCTIONS[name](
        **case["kwargs"],
        control=Control(max_iter=5, tolerance=0, jit=compiled, convergence="absolute"),
    )
    for group, actual in [("means", result.means), ("vars", result.variances)]:
        for key, value in actual.items():
            expected = np.asarray(case["expected"][group][key]).reshape(value.shape)
            np.testing.assert_allclose(
                value, expected, atol=1e-8, rtol=1e-6, err_msg=f"{name}: {group}.{key}"
            )
    assert result.iterations == 5
    assert not result.converged
