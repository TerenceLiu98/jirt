"""Regression against independently fitted R mirt ordered-response models."""

import json
from pathlib import Path
import jax
import numpy as np
import pytest
from jirt import polytomous_irt, polytomous_probabilities

jax.config.update("jax_enable_x64", True)
FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "polytomous_mirt.json").read_text()
)


@pytest.mark.parametrize("model", FIXTURE["cases"])
def test_mirt_calibration(model):
    case = FIXTURE["cases"][model]
    r = case["reference"]
    counts = np.asarray(case["category_counts"])
    y = np.repeat(case["patterns"], case["counts"], axis=0)
    fit = polytomous_irt(
        y, category_counts=counts, model=model, tolerance=1e-13, max_iter=3000
    )
    assert fit.converged and r["converged"]
    for key in ("a", "steps"):
        np.testing.assert_allclose(fit.means[key], r[key], atol=1e-4, rtol=0)
    assert abs(fit.log_likelihood - r["log_likelihood"]) < 1e-5
    first = np.r_[0, np.cumsum(case["counts"])[:-1]]
    np.testing.assert_allclose(
        np.asarray(fit.means["x"])[first], r["scores"], atol=1e-4, rtol=0
    )
    np.testing.assert_allclose(
        np.asarray(fit.variances["x"])[first], r["variances"], atol=1e-4, rtol=0
    )
    fitted = np.asarray(fit.predict(r["theta"]))
    fixed = np.asarray(
        polytomous_probabilities(
            r["theta"], a=r["a"], steps=r["steps"], category_counts=counts, model=model
        )
    )
    for j, k in enumerate(counts):
        np.testing.assert_allclose(
            fitted[:, j, :k], r["probabilities"][j], atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(
            fixed[:, j, :k], r["probabilities"][j], atol=1e-12, rtol=0
        )
