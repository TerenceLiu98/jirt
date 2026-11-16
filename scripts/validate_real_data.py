"""Run all full-data parity checks and persist an auditable report.

Prepare exported data and R baselines first. This command fails if any input
is missing or any model fails; it never silently skips a model.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("binary", "ordinal", "dynamic", "hierarchical", "network", "poisson")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wait-for-hierarchical",
        action="store_true",
        help="Wait for an already running R baseline export",
    )
    args = parser.parse_args()
    if args.wait_for_hierarchical:
        target = ROOT / "outputs" / "baselines" / "hierarchical.json"
        while True:
            try:
                json.loads(target.read_text())
                break
            except (FileNotFoundError, json.JSONDecodeError):
                time.sleep(10)
    source_digest = hashlib.sha256()
    for path in sorted((ROOT / "src" / "jirt").glob("*.py")):
        source_digest.update(path.name.encode())
        source_digest.update(path.read_bytes())
    report = {}
    for model in MODELS:
        baseline = ROOT / "outputs" / "baselines" / f"{model}.json"
        if not baseline.is_file():
            raise FileNotFoundError(f"Missing baseline: {baseline}")
    for model in MODELS:
        start = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / f"check_{model}_parity.py")],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        report[model] = dict(
            passed=proc.returncode == 0,
            source_sha256=source_digest.hexdigest(),
            seconds=time.perf_counter() - start,
            stdout=proc.stdout,
            stderr=proc.stderr,
            baseline_sha256=hashlib.sha256(
                (ROOT / "outputs" / "baselines" / f"{model}.json").read_bytes()
            ).hexdigest(),
        )
        (ROOT / "outputs" / "real-validation.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(model, "PASS" if proc.returncode == 0 else "FAIL", flush=True)
    if not all(v["passed"] for v in report.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
