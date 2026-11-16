"""Measure full-data discrepancies using the existing parity checks and inputs."""

import argparse
import json
from pathlib import Path
import runpy

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        choices=["binary", "ordinal", "dynamic", "hierarchical", "network", "poisson"],
    )
    args = parser.parse_args()
    comparisons = []
    original = np.testing.assert_allclose

    def measure(actual, desired, *a, **kw):
        original(actual, desired, *a, **kw)
        actual, desired = np.asarray(actual), np.asarray(desired)
        diff = np.abs(actual - desired)
        nonzero = desired != 0
        comparisons.append(
            dict(
                count=diff.size,
                max_absolute=float(diff.max()),
                sum_squared=float(np.square(diff).sum()),
                max_relative_nonzero=float(
                    (diff[nonzero] / np.abs(desired[nonzero])).max()
                )
                if nonzero.any()
                else None,
                zero_reference_count=int((~nonzero).sum()),
                max_absolute_at_zero=float(diff[~nonzero].max())
                if (~nonzero).any()
                else None,
            )
        )

    np.testing.assert_allclose = measure
    try:
        state = runpy.run_path(
            f"scripts/check_{args.model}_parity.py", run_name="__main__"
        )
    finally:
        np.testing.assert_allclose = original
    fit = (
        state["result"]
        if args.model == "binary"
        else state["fit"]
        if args.model == "network"
        else state["r"]
    )
    fields = [("means", k) for k in fit.means] + [
        ("variances", k) for k in fit.variances
    ]
    methods = ["em", "variational"] if args.model == "binary" else [args.model]
    assert len(comparisons) == len(fields) * len(methods)
    result = {}
    for i, method in enumerate(methods):
        groups = {}
        for (group, key), stats in zip(
            fields, comparisons[i * len(fields) : (i + 1) * len(fields)], strict=True
        ):
            groups.setdefault(group, {})[key] = stats
        summary = {}
        for group, values in groups.items():
            summary[group] = dict(
                max_absolute=max(v["max_absolute"] for v in values.values()),
                rmse=float(
                    np.sqrt(
                        sum(v["sum_squared"] for v in values.values())
                        / sum(v["count"] for v in values.values())
                    )
                ),
                max_relative_nonzero=max(
                    v["max_relative_nonzero"] or 0 for v in values.values()
                ),
            )
        result[method] = dict(summary=summary, fields=groups)
    Path(f"outputs/{args.model}-errors.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps({k: v["summary"] for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
