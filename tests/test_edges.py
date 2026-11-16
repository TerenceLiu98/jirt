import json
from pathlib import Path
import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest
from jirt import Control, bin_irt, ord_irt, dyn_irt, hier_irt, bootstrap


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iter": 0},
        {"max_iter": 1.5},
        {"max_iter": True},
        {"tolerance": np.nan},
        {"tolerance": -1},
        {"convergence": "bad"},
        {"jit": 1},
    ],
)
def test_control_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        Control(**kwargs)


def case(name):
    return json.loads(
        (Path(__file__).parent / "fixtures" / f"{name}.json").read_text()
    )["kwargs"]


@pytest.mark.parametrize(
    "name,fn,input_key",
    [
        ("ordinal", ord_irt, "responses"),
        ("dynamic", dyn_irt, "votes"),
        ("hierarchical", hier_irt, "votes"),
    ],
)
def test_bootstrap_other_models(name, fn, input_key):
    kw = case(name)
    y = np.asarray(kw.pop(input_key))
    kw["control"] = Control(max_iter=3, tolerance=0)
    result = fn(y, **kw)
    if name == "hierarchical":
        y = y.reshape(-1)
    reps = np.stack([y, y.copy()])
    reps[1].flat[0] = 2 if name == "ordinal" else -reps[1].flat[0]
    r = bootstrap(
        result,
        y,
        model=name,
        fit_kwargs=kw,
        key=jax.random.key(2),
        n_trials=2,
        pseudo_data=reps,
    )
    assert not r.failures
    target = "x_implied" if name == "hierarchical" else "x"
    expected = np.stack([fn(v, **kw).means[target] for v in reps]).std(axis=0, ddof=1)
    if name == "dynamic":
        first = np.asarray(kw["start_period"]).reshape(-1)
        last = np.asarray(kw["end_period"]).reshape(-1)
        active = (np.arange(expected.shape[1])[None, :] >= first[:, None]) & (
            np.arange(expected.shape[1])[None, :] <= last[:, None]
        )
        expected = np.where(active, expected, np.nan)
    np.testing.assert_allclose(r.standard_errors, expected, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("value", [0, 1, -1])
def test_binary_constant_missing(value):
    y = np.full((4, 3), value)
    r = bin_irt(
        y,
        starts=dict(alpha=np.zeros(3), beta=np.zeros(3), x=np.zeros(4)),
        control=Control(max_iter=5),
    )
    assert np.isfinite(r.probabilities).all()


def test_outcome_anchor():
    kw = case("binary_em")
    plain = bin_irt(**kw, control=Control(max_iter=5))
    anchored = bin_irt(**kw, control=Control(max_iter=5), anchor_outcomes=True)
    np.testing.assert_allclose(plain.probabilities, anchored.probabilities)
