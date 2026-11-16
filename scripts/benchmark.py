"""Measure JIT compilation and execution on the deterministic model fixtures.

Uses a fresh subprocess per case to avoid sharing compiled functions between
measurements. Timing is diagnostic, not a portable speed guarantee.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    "binary_em",
    "binary_variational",
    "ordinal",
    "dynamic",
    "hierarchical",
    "network",
    "poisson",
)


def worker(model):
    import jax

    jax.config.update("jax_enable_x64", True)
    import jirt

    original_jit = jax.jit
    stats = {"compile_seconds": 0.0, "update_seconds": 0.0, "updates": 0}

    def measured_jit(fn, *args, **kwargs):
        wrapped = original_jit(fn, *args, **kwargs)
        compiled = None

        def invoke(*values):
            nonlocal compiled
            if compiled is None:
                start = time.perf_counter()
                compiled = wrapped.lower(*values).compile()
                stats["compile_seconds"] += time.perf_counter() - start
            start = time.perf_counter()
            result = jax.block_until_ready(compiled(*values))
            stats["update_seconds"] += time.perf_counter() - start
            stats["updates"] += 1
            return result

        return invoke

    functions = dict(
        binary_em=jirt.bin_irt,
        binary_variational=jirt.bin_irt,
        ordinal=jirt.ord_irt,
        dynamic=jirt.dyn_irt,
        hierarchical=jirt.hier_irt,
        network=jirt.network_irt,
        poisson=jirt.pois_irt,
    )
    case = json.loads((ROOT / "tests" / "fixtures" / f"{model}.json").read_text())
    jax.jit = measured_jit
    try:
        start = time.perf_counter()
        fit = functions[model](
            **case["kwargs"], control=jirt.Control(max_iter=30, tolerance=0)
        )
        jax.block_until_ready(fit.means)
        stats["total_seconds"] = time.perf_counter() - start
    finally:
        jax.jit = original_jit
    stats["mean_update_seconds"] = stats["update_seconds"] / stats["updates"]
    print(json.dumps(stats))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    if args.model:
        worker(args.model)
    else:
        report = {}
        for model in MODELS:
            result = subprocess.run(
                [sys.executable, __file__, "--model", model],
                check=True,
                capture_output=True,
                text=True,
                cwd=ROOT,
            )
            report[model] = json.loads(result.stdout)
            print(model, report[model], flush=True)
        (ROOT / "outputs").mkdir(exist_ok=True)
        (ROOT / "outputs" / "benchmark.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
