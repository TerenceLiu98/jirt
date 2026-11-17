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

    import platform
    import jirt._iteration as iteration

    original = iteration._cached_iterations
    stats = {
        "compile_seconds": 0.0,
        "kernel_seconds": [],
        "total_seconds": [],
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "platform": platform.platform(),
        "dtype": "float64",
        "max_iter": 30,
        "tolerance": 0,
        "convergence": "correlation",
    }
    executable = None

    def measured(step, state, data, tolerance, **options):
        nonlocal executable
        if executable is None:
            start = time.perf_counter()
            executable = original.lower(
                step, state, data, tolerance, **options
            ).compile()
            stats["compile_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        result = jax.block_until_ready(executable(state, data, tolerance))
        stats["kernel_seconds"].append(time.perf_counter() - start)
        return result

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
    iteration._cached_iterations = measured
    try:
        for _ in range(4):
            start = time.perf_counter()
            fit = functions[model](
                **case["kwargs"], control=jirt.Control(max_iter=30, tolerance=0)
            )
            jax.block_until_ready(fit.means)
            stats["total_seconds"].append(time.perf_counter() - start)
        stats["iterations"] = fit.iterations
        stats["converged"] = fit.converged
    finally:
        iteration._cached_iterations = original
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
