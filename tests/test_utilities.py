import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest
from jirt import convert_votes, get_starts, make_priors, bin_irt, bootstrap, Control


def test_recode():
    np.testing.assert_array_equal(
        convert_votes([2, 3, 8, 9, np.nan], yes=(2,), no=(3,), missing=(8,)),
        [1, -1, 0, 9, 0],
    )
    with pytest.raises(ValueError):
        convert_votes([4])
    with pytest.raises(ValueError):
        convert_votes([1], no=(1,))


def test_bootstrap_explicit_replicates():
    y = np.array([[1, -1, 0], [-1, 1, 1], [1, 1, -1], [-1, -1, 9]])
    kwargs = dict(
        starts=get_starts(4, 3, key=jax.random.key(1)),
        priors=make_priors(),
        control=Control(max_iter=10),
    )
    fit = bin_irt(y, **kwargs)
    reps = np.stack([y, y.copy()])
    reps[1, 0, 0] = -1
    boot = bootstrap(
        fit,
        y,
        model="binary",
        fit_kwargs=kwargs,
        key=jax.random.key(3),
        n_trials=2,
        pseudo_data=reps,
    )
    expected = np.stack([bin_irt(v, **kwargs).means["x"] for v in reps])
    np.testing.assert_allclose(boot.standard_errors, expected.std(axis=0, ddof=1))
    assert not boot.failures
