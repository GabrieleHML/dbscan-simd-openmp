#!/usr/bin/env python3
"""End-to-end tests: every binary and variant must reproduce scikit-learn's labels.

Usage: tests/run_tests.py BINARY...

The 64-bit builds work in double precision like scikit-learn and must match it
exactly. The 32-bit builds work in single precision, so a point whose distance
is within rounding of Eps may fall on the other side: a few mismatches are allowed.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from compare_sklearn import sklearn_labels  # noqa: E402
from dsio import read_labels, write_dataset  # noqa: E402
from make_dataset import blobs  # noqa: E402

MAX_MISMATCH_32 = 0.002  # share of points allowed to differ in single precision


def datasets(rng: np.random.Generator):
    """(name, points, eps, min_pts) of every case, from the typical to the corner ones."""
    yield "blobs d=2", blobs(3000, 2, 4, 1.0, 0.05, rng), 0.5, 5
    yield "blobs d=13 (padded to 16)", blobs(2000, 13, 5, 1.0, 0.1, rng), 3.5, 5
    yield "blobs d=64", blobs(1500, 64, 3, 1.0, 0.05, rng), 11.0, 4
    uniform = rng.random((1000, 50))
    yield "uniform, one cluster (exam case)", uniform, 5.0, 5
    yield "uniform, all noise", uniform, 1.2, 5
    same = np.vstack([np.ones((300, 3)), rng.random((100, 3)) * 5])
    yield "300 identical points", same, 0.3, 5
    yield "min_pts=1, every point core", blobs(500, 5, 3, 1.0, 0.2, rng), 1.0, 1
    yield "1 point", np.zeros((1, 3)), 1.0, 1
    yield "3 points", np.array([[0.0, 0], [0, 1], [5, 5]]), 1.5, 2


def variants(binary: Path):
    """Command-line options to test for a binary."""
    threads = [["-t", "1"], ["-t", "4"]] if binary.name.endswith("-omp") else [[]]
    for t in threads:
        for simd in [[], ["--no-simd"]]:
            yield ["--index", "linear", *simd, *t]
            yield ["--index", "balltree", *simd, *t]
            yield ["--index", "balltree", "--leaf-size", "1", *simd, *t]


def run(binary: Path, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([str(binary), *args], capture_output=True, text=True, timeout=120)


def test_cli(binary: Path) -> list[str]:
    failures = []
    if run(binary, ["--help"]).returncode != 0:
        failures.append("--help does not exit with 0")
    if run(binary, ["-i", "x.ds", "-e", "-1", "-m", "5"]).returncode != 2:
        failures.append("a negative Eps is not rejected with exit code 2")
    if run(binary, ["-i", "/nonexistent.ds", "-e", "1", "-m", "5"]).returncode != 1:
        failures.append("a missing file does not exit with 1")
    return failures


def main() -> int:
    binaries = [Path(b) for b in sys.argv[1:]]
    if not binaries:
        print(__doc__)
        return 2

    failed = 0
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as tmp:
        cases = []
        for i, (name, points, eps, min_pts) in enumerate(datasets(rng)):
            path = Path(tmp) / f"case{i}.ds"
            write_dataset(path, points)
            cases.append((name, path, eps, min_pts, sklearn_labels(points, eps, min_pts)))

        for binary in binaries:
            print(f"\n{binary}")
            for failure in test_cli(binary):
                print(f"  FAIL  command line: {failure}")
                failed += 1
            single = "32" in binary.name
            for name, path, eps, min_pts, expected in cases:
                worst = 0
                for options in variants(binary):
                    out = Path(tmp) / "labels"
                    result = run(binary, ["-i", str(path), "-e", str(eps), "-m", str(min_pts), "-o", str(out), "-q", "--json", *options])
                    if result.returncode != 0:
                        print(f"  FAIL  {name} {' '.join(options)}: exit code {result.returncode}\n{result.stderr}")
                        failed += 1
                        continue
                    stats = json.loads(result.stdout)
                    labels = read_labels(out)
                    mismatches = int(np.count_nonzero(labels != expected))
                    allowed = int(MAX_MISMATCH_32 * len(expected)) if single else 0
                    if mismatches > allowed or stats["clusters"] != labels.max() + 1:
                        print(f"  FAIL  {name} {' '.join(options)}: {mismatches} labels differ from scikit-learn")
                        failed += 1
                    worst = max(worst, mismatches)
                print(f"  ok    {name}" + (f" ({worst} mismatches at most)" if worst else ""))

    print(f"\n{'FAILED: ' + str(failed) + ' checks' if failed else 'All tests passed'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
