"""Count Python/host conversion entry points in warmed binary fits.

These CPU profiles measure Python overhead and potential synchronization points,
not GPU transfer volume. --source can select a previous checkout's src directory.
"""

import argparse
import cProfile
import json
from pathlib import Path
import pstats
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    options = parser.parse_args()
    if options.source:
        sys.path.insert(0, str(options.source.resolve()))
    import jax
    import numpy as np
    from jirt import Control, bin_irt, get_starts, make_priors

    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(2026)
    votes = rng.choice([-1.0, 1.0], size=(100, 6))
    kwargs = dict(
        starts=get_starts(100, 6, key=jax.random.key(1)), priors=make_priors()
    )
    report = []
    for iterations in (5, 50):
        control = Control(max_iter=iterations, tolerance=0)
        jax.block_until_ready(bin_irt(votes, control=control, **kwargs).means)
        profiler = cProfile.Profile()
        profiler.enable()
        fit = bin_irt(votes, control=control, **kwargs)
        jax.block_until_ready(fit.means)
        profiler.disable()
        stats = pstats.Stats(profiler)
        conversions = {}
        for (filename, line, function), (_, calls, _, _, _) in stats.stats.items():
            if (
                "jax" in filename
                and function in ("__array__", "__bool__", "__float__", "__int__")
            ) or "numpy.asarray" in function:
                key = Path(filename).name + ":" + function
                conversions[key] = conversions.get(key, 0) + calls
        report.append(
            dict(
                iterations=fit.iterations,
                python_calls=stats.total_calls,
                conversion_calls=conversions,
            )
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
