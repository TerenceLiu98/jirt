"""Time three updates on full parity datasets, excluding input file loading."""

import argparse
import json
from pathlib import Path
import runpy
import time

import jax

jax.config.update("jax_enable_x64", True)
import jirt


class Measured(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()
    model = args.model
    base = "binary" if model.startswith("binary_") else model
    name = dict(
        binary="bin_irt",
        ordinal="ord_irt",
        dynamic="dyn_irt",
        hierarchical="hier_irt",
        network="network_irt",
        poisson="pois_irt",
    )[base]
    original = getattr(jirt, name)
    results = []

    def measure(*a, **kw):
        old = kw.get("control", jirt.Control())
        kw["control"] = jirt.Control(
            max_iter=3, tolerance=0, convergence=old.convergence
        )
        if base == "binary":
            kw["method"] = model.removeprefix("binary_")
        for repeat in range(2):
            start = time.perf_counter()
            fit = original(*a, **kw)
            jax.block_until_ready((fit.means, fit.variances, fit.probabilities))
            elapsed = time.perf_counter() - start
            assert fit.iterations == 3
            results.append(
                dict(
                    call=repeat + 1, elapsed_seconds=elapsed, iterations=fit.iterations
                )
            )
        raise Measured

    setattr(jirt, name, measure)
    try:
        runpy.run_path(f"scripts/check_{base}_parity.py", run_name="__main__")
    except Measured:
        pass
    finally:
        setattr(jirt, name, original)
    assert len(results) == 2
    Path(f"outputs/speed-jax-{model}.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print(model, results, flush=True)


if __name__ == "__main__":
    main()
