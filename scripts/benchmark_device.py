"""Synchronized cold/warm end-to-end benchmarks, with explicit solver outcomes.

Use --source PATH/src to benchmark a previous checkout with --engine baseline.
Run each case in a fresh process; generated reports belong under outputs/.
"""

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time


def worker(case, size, engine):
    import jax
    import numpy as np
    import jirt

    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(2026)
    n, items = (200, 6) if size == "small" else (2000, 20)
    theta = rng.normal(size=n)
    difficulty = np.linspace(-1.5, 1.5, items)
    probabilities = 1 / (1 + np.exp(-(theta[:, None] - difficulty)))
    y = (rng.random((n, items)) < probabilities).astype(float)
    if case == "binary":
        fn = jirt.bin_irt
        kwargs = dict(
            votes=2 * y - 1,
            starts=jirt.get_starts(n, items, key=jax.random.key(1)),
            priors=jirt.make_priors(),
            control=jirt.Control(max_iter=200, tolerance=1e-6),
        )
    elif case == "diagnostic":
        fn = jirt.diagnostic_irt
        q = np.array([[1, 0], [0, 1], [1, 1]])
        q = q[np.arange(items) % 3]
        kwargs = dict(responses=y, q_matrix=q, max_iter=200, tolerance=1e-6)
    elif case == "logistic":
        fn = jirt.logistic_irt
        kwargs = dict(
            responses=y,
            model="2pl",
            max_iter=3000,
            tolerance=1e-6 if engine == "jax" else 1e-11,
        )
    else:
        fn = jirt.amortized_irt
        ids = [str(i) for i in range(items)]
        kwargs = dict(
            responses=y,
            item_features=difficulty[:, None],
            item_ids=ids,
            feature_item_ids=ids,
            hidden_size=0,
            key=jax.random.key(1),
            l2=0.01,
            max_iter=3000,
            tolerance=1e-6 if engine == "jax" else 1e-11,
        )
    if case in ("logistic", "amortized") and engine != "baseline":
        kwargs["optimizer"] = engine
    compile_seconds = []
    kernel_seconds = []
    if engine == "jax":
        import importlib

        module_name, attribute = {
            "binary": ("_iteration", "_cached_iterations"),
            "diagnostic": ("diagnostic", "_fit_compiled"),
            "logistic": ("_optimize", "_solve"),
            "amortized": ("amortized", "_fit_bfgs"),
        }[case]
        module = importlib.import_module("jirt." + module_name)
        original = getattr(module, attribute)
        executable = None

        def measured(*values, **options):
            nonlocal executable
            if executable is None:
                start = time.perf_counter()
                executable = original.lower(*values, **options).compile()
                compile_seconds.append(time.perf_counter() - start)
            if case in ("binary", "logistic"):
                dynamic = values[1:]
            elif case == "amortized":
                dynamic = (values[0], values[1], values[4])
            else:
                dynamic = values
            start = time.perf_counter()
            result = jax.block_until_ready(executable(*dynamic))
            kernel_seconds.append(time.perf_counter() - start)
            return result

        setattr(module, attribute, measured)
    times = []
    for _ in range(4):
        start = time.perf_counter()
        fit = fn(**kwargs)
        for value in vars(fit).values():
            if isinstance(value, (jax.Array, dict)):
                jax.block_until_ready(value)
        times.append(time.perf_counter() - start)
    record = dict(
        case=case,
        size=size,
        engine=engine,
        shape=[n, items],
        backend=jax.default_backend(),
        devices=[str(d) for d in jax.devices()],
        platform=platform.platform(),
        dtype="float64",
        jax=jax.__version__,
        cold_seconds=times[0],
        warm_seconds=times[1:],
        iterations=fit.iterations,
        converged=fit.converged,
        tolerance=kwargs.get("tolerance", 1e-6),
        max_iter=kwargs.get("max_iter", 200),
    )
    record["compile_seconds"] = compile_seconds
    record["kernel_seconds"] = kernel_seconds
    if hasattr(fit, "log_likelihood"):
        record["log_likelihood"] = fit.log_likelihood
    if hasattr(fit, "objective"):
        record["objective"] = fit.objective
    if hasattr(fit, "message"):
        record["message"] = fit.message
    print(json.dumps(record))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=["binary", "diagnostic", "logistic", "amortized"]
    )
    parser.add_argument("--size", choices=["small", "large"], default="small")
    parser.add_argument("--engine", choices=["jax", "scipy", "baseline"], default="jax")
    parser.add_argument("--source", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/device-benchmark.json")
    )
    options = parser.parse_args()
    if options.source:
        sys.path.insert(0, str(options.source.resolve()))
    if options.case:
        worker(options.case, options.size, options.engine)
        return
    report = []
    for case in ("binary", "diagnostic", "logistic", "amortized"):
        for size in ("small", "large"):
            command = [
                sys.executable,
                __file__,
                "--case",
                case,
                "--size",
                size,
                "--engine",
                options.engine,
            ]
            if options.source:
                command.extend(["--source", str(options.source.resolve())])
            result = subprocess.run(command, text=True, capture_output=True, check=True)
            record = json.loads(result.stdout)
            report.append(record)
            print(json.dumps(record), flush=True)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
