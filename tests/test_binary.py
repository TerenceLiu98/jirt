import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
from jirt import bin_irt, Control


def test_jit_and_anchor():
    y = np.array([[1, -1, 0], [-1, 1, 1], [1, 1, -1], [-1, -1, 9]])
    starts = dict(
        alpha=np.zeros(3), beta=np.zeros(3), x=np.array([-1.0, 0.2, 0.5, 1.0])
    )
    for method in ("em", "variational"):
        eager = bin_irt(
            y, starts=starts, method=method, control=Control(max_iter=20, jit=False)
        )
        compiled = bin_irt(
            y, starts=starts, method=method, control=Control(max_iter=20)
        )
        np.testing.assert_allclose(
            eager.means["x"], compiled.means["x"], rtol=1e-8, atol=1e-8
        )
        anchored = bin_irt(
            y,
            starts=starts,
            method=method,
            control=Control(max_iter=20),
            anchor_subject=0,
        )
        assert anchored.means["x"][0] >= np.mean(anchored.means["x"])
        np.testing.assert_allclose(compiled.probabilities, anchored.probabilities)
