"""Entire-item holdout with a content encoder and non-amortized Rasch baseline."""

import jax
import numpy as np
from scipy.special import expit
from jirt.amortized import amortized_irt
from jirt import logistic_irt


def run():
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(901)
    # Locally generated content descriptors; no remote embedding service.
    features = rng.normal(size=(80, 4))
    difficulty = features @ np.array([1.0, -0.7, 0.5, 0.2]) + 0.3
    theta = rng.normal(size=1200)
    responses = (rng.random((1200, 80)) < expit(theta[:, None] - difficulty)).astype(
        float
    )
    permutation = rng.permutation(80)
    train = permutation[:60]
    test = permutation[60:]
    ids = [f"item-{j}" for j in train]
    fit = amortized_irt(
        responses[:, train],
        item_features=features[train],
        item_ids=ids,
        feature_item_ids=ids,
        key=jax.random.key(901),
        hidden_size=0,
        l2=1e-4,
    )
    baseline = logistic_irt(responses[:, train], model="rasch")
    assert fit.converged, fit.message
    assert baseline.converged, baseline.message
    # No responses or preprocessing statistics from the 20 held-out items enter
    # either fit. Rasch has no content mapping: use mean TRAINING difficulty as
    # an explicit cold-start fallback, not a calibrated estimate for new items.
    p = np.asarray(fit.predict(features[test]))
    baseline_p = expit(
        np.asarray(baseline.means["x"])[:, None] - float(np.mean(baseline.means["b"]))
    )

    def loss(probability):
        probability = np.clip(probability, 1e-12, 1 - 1e-12)
        y = responses[:, test]
        return float(
            -np.mean(y * np.log(probability) + (1 - y) * np.log1p(-probability))
        )

    report = dict(
        training_items=len(train),
        held_out_items=len(test),
        amortized_log_loss=loss(p),
        non_amortized_1pl_log_loss=loss(baseline_p),
        held_out_difficulty_rmse=float(
            np.sqrt(
                np.mean(
                    (
                        np.asarray(fit.infer_difficulty(features[test]))
                        - difficulty[test]
                    )
                    ** 2
                )
            )
        ),
    )
    return report


if __name__ == "__main__":
    for key, value in run().items():
        print(key, value)
