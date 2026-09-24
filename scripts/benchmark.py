#!/usr/bin/env python3
"""Runs the experiments of the report and writes one CSV per experiment in results/.

Every measurement is the median of several runs. The C program times itself
(clustering only, file I/O excluded); scikit-learn is timed around fit().
Generated datasets are cached in data/.

Usage: scripts/benchmark.py [--only NAME ...] [--reps N]
"""

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import sklearn
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from dsio import read_dataset, write_dataset
from make_dataset import blobs

ROOT = Path(__file__).resolve().parent.parent
BIN, DATA, RESULTS = ROOT / "bin", ROOT / "data", ROOT / "results"
MIN_PTS = 5
FIELDS = [
    "experiment", "dataset", "n", "d", "eps", "min_pts", "program", "index", "simd", "threads", "leaf_size",
    "opt", "time", "time_min", "time_max", "index_time", "neighbour_time", "expand_time",
    "distance_evals", "tree_depth", "clusters", "noise_points", "reps",
]


def dataset(kind: str, n: int, d: int) -> tuple[Path, float]:
    """Path of a cached dataset and the Eps to use with it."""
    DATA.mkdir(exist_ok=True)
    path = DATA / f"{kind}_{n}x{d}.ds"
    if kind == "exam":
        # The dataset used for the exam: uniform points and Eps=5, larger than most distances
        if not path.exists():
            write_dataset(path, np.random.default_rng(0).random((n, d)))
        return path, 5.0
    if not path.exists():
        write_dataset(path, blobs(n, d, 8, 1.0, 0.05, np.random.default_rng(n + d)))
    # Eps from the k-distance heuristic of the DBSCAN paper: 90% of the points become core points
    eps_path = path.with_suffix(".eps")
    if not eps_path.exists():
        x = read_dataset(path)
        dist, _ = NearestNeighbors(n_neighbors=MIN_PTS).fit(x).kneighbors(x)
        eps_path.write_text(f"{np.quantile(dist[:, -1], 0.9):.6f}\n")
    return path, float(eps_path.read_text())


def run_c(binary: Path, path: Path, eps: float, options: list[str], reps: int) -> dict:
    runs = []
    for _ in range(reps):
        out = subprocess.run(
            [str(binary), "-i", str(path), "-e", str(eps), "-m", str(MIN_PTS), "-q", "--json", *options],
            capture_output=True, text=True, check=True,
        )
        runs.append(json.loads(out.stdout))
    times = [r["total_time"] for r in runs]
    median = sorted(runs, key=lambda r: r["total_time"])[len(runs) // 2]
    return {
        **{k: median[k] for k in ["index_time", "neighbour_time", "expand_time", "distance_evals", "tree_depth",
                                  "clusters", "noise_points", "threads", "simd", "index"]},
        "leaf_size": median["leaf_size"] if median["index"] == "balltree" else "",
        "time": statistics.median(times), "time_min": min(times), "time_max": max(times),
    }


def run_exam(binary: Path, path: Path, eps: float, reps: int) -> dict:
    """The exam version prints its CPU time (clock()) on the last line; it is single-threaded."""
    times = []
    for _ in range(reps):
        out = subprocess.run(
            [str(binary), "-ds", str(path), "-labels", str(path), "-k", "1", "-MinPts", str(MIN_PTS),
             "-Eps", str(eps), "-s"],
            capture_output=True, text=True, check=True, cwd=DATA,
        )
        times.append(float(out.stdout.split()[-1]))
    return {"index": "balltree", "simd": True, "threads": 1, "leaf_size": 300,
            "time": statistics.median(times), "time_min": min(times), "time_max": max(times)}


def run_sklearn(path: Path, eps: float, jobs: int, reps: int) -> dict:
    x = read_dataset(path)
    times, labels = [], None
    for _ in range(reps):
        start = time.perf_counter()
        labels = DBSCAN(eps=eps, min_samples=MIN_PTS, n_jobs=jobs).fit(x).labels_
        times.append(time.perf_counter() - start)
    return {"index": "auto", "threads": os.cpu_count() if jobs == -1 else jobs, "clusters": int(labels.max() + 1),
            "noise_points": int(np.count_nonzero(labels == -1)),
            "time": statistics.median(times), "time_min": min(times), "time_max": max(times)}


class Writer:
    def __init__(self, name: str):
        RESULTS.mkdir(exist_ok=True)
        self.name = name
        self.file = open(RESULTS / f"{name}.csv", "w", newline="")
        self.csv = csv.DictWriter(self.file, FIELDS, extrasaction="ignore")
        self.csv.writeheader()

    def add(self, kind: str, n: int, d: int, eps: float, program: str, result: dict, reps: int, opt: str = "-O2"):
        row = {"experiment": self.name, "dataset": kind, "n": n, "d": d, "eps": eps, "min_pts": MIN_PTS,
               "program": program, "opt": opt, "reps": reps, **result}
        self.csv.writerow(row)
        self.file.flush()
        print(f"  {program:14s} n={n:<6d} d={d:<4d} {row.get('index', ''):9s} simd={row.get('simd', '')!s:5s} "
              f"threads={row.get('threads', '')!s:2s} leaf={row.get('leaf_size', '')!s:4s} {row['time']:8.4f} s")

    def close(self):
        self.file.close()


def c_binaries() -> list[tuple[str, Path]]:
    return [(name, BIN / name) for name in ["dbscan64", "dbscan32"] if (BIN / name).exists()]


def variants(w: Writer, reps: int):
    """Every optimization step on the same inputs, from the exam version to OpenMP."""
    for kind, n, d in [("blobs", 20000, 16), ("exam", 5000, 50)]:
        path, eps = dataset(kind, n, d)
        exam = BIN / "dbscan64-exam"
        if exam.exists():
            w.add(kind, n, d, eps, "exam-2024", run_exam(exam, path, eps, reps), reps, opt="-O0")
        for name, binary in c_binaries():
            for index in ["linear", "balltree"]:
                for simd in [["--no-simd"], []]:
                    w.add(kind, n, d, eps, name, run_c(binary, path, eps, ["--index", index, *simd], reps), reps)
            omp = binary.with_name(name + "-omp")
            if omp.exists():
                for threads in [4, 8]:
                    w.add(kind, n, d, eps, omp.name, run_c(omp, path, eps, ["-t", str(threads)], reps), reps)
        for jobs in [1, -1]:
            w.add(kind, n, d, eps, "scikit-learn", run_sklearn(path, eps, jobs, reps), reps)


def threads(w: Writer, reps: int):
    """OpenMP scaling of the ball tree and of the linear search."""
    for index, n in [("balltree", 80000), ("linear", 20000)]:
        path, eps = dataset("blobs", n, 16)
        for t in range(1, (os.cpu_count() or 8) + 1):
            w.add("blobs", n, 16, eps, "dbscan64-omp",
                  run_c(BIN / "dbscan64-omp", path, eps, ["--index", index, "-t", str(t)], reps), reps)


def scaling(w: Writer, reps: int):
    """Running time as the number of points grows."""
    for n in [5000, 10000, 20000, 40000, 80000]:
        path, eps = dataset("blobs", n, 16)
        if n <= 40000:
            w.add("blobs", n, 16, eps, "dbscan64", run_c(BIN / "dbscan64", path, eps, ["--index", "linear"], reps), reps)
        w.add("blobs", n, 16, eps, "dbscan64", run_c(BIN / "dbscan64", path, eps, [], reps), reps)
        w.add("blobs", n, 16, eps, "dbscan64-omp", run_c(BIN / "dbscan64-omp", path, eps, ["-t", "8"], reps), reps)
        w.add("blobs", n, 16, eps, "scikit-learn", run_sklearn(path, eps, 1, reps), reps)


def leaf_size(w: Writer, reps: int):
    """Effect of the leaf size of the ball tree."""
    for n, d in [(40000, 16), (20000, 64)]:
        path, eps = dataset("blobs", n, d)
        for leaf in [1, 2, 4, 8, 16, 32, 64, 128, 300, 1000]:
            w.add("blobs", n, d, eps, "dbscan64",
                  run_c(BIN / "dbscan64", path, eps, ["--leaf-size", str(leaf)], reps), reps)


def dimensions(w: Writer, reps: int):
    """How much the ball tree prunes as the dimensions grow."""
    for d in [2, 4, 8, 16, 32, 64, 128]:
        path, eps = dataset("blobs", 20000, d)
        for index in ["linear", "balltree"]:
            w.add("blobs", 20000, d, eps, "dbscan64", run_c(BIN / "dbscan64", path, eps, ["--index", index], reps), reps)


def optimization(w: Writer, reps: int):
    """Hand-written SIMD against the C kernel, compiled with -O0 (as in the course) and -O2."""
    o0 = ROOT / "build" / "O0"
    subprocess.run(["make", "-s", "x64", "OPT=-O0", f"BIN={o0}"], cwd=ROOT, check=True)
    path, eps = dataset("blobs", 20000, 16)
    for opt, folder in [("-O0", o0), ("-O2", BIN)]:
        for index in ["linear", "balltree"]:
            for simd in [["--no-simd"], []]:
                w.add("blobs", 20000, 16, eps, "dbscan64",
                      run_c(folder / "dbscan64", path, eps, ["--index", index, *simd], reps), reps, opt=opt)


EXPERIMENTS = {"variants": variants, "threads": threads, "scaling": scaling, "leaf_size": leaf_size,
               "dimensions": dimensions, "optimization": optimization}


def environment() -> dict:
    cpu = next((line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name")), "")
    version = lambda cmd: subprocess.run(cmd, capture_output=True, text=True).stdout.splitlines()[0]  # noqa: E731
    return {"date": date.today().isoformat(), "cpu": cpu, "logical_cpus": os.cpu_count(),
            "os": platform.platform(), "gcc": version(["gcc", "--version"]),
            "nasm": version([os.environ.get("NASM", "nasm"), "-v"]),
            "python": platform.python_version(), "numpy": np.__version__, "scikit-learn": sklearn.__version__}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=EXPERIMENTS, help="run only these experiments")
    parser.add_argument("--reps", type=int, default=5, help="runs per measurement (default: 5)")
    args = parser.parse_args()

    if not (BIN / "dbscan64").exists():
        sys.exit("build the program first: make")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "environment.json").write_text(json.dumps(environment(), indent=2) + "\n")
    for name in args.only or EXPERIMENTS:
        print(f"\n{name}")
        writer = Writer(name)
        EXPERIMENTS[name](writer, args.reps)
        writer.close()


if __name__ == "__main__":
    main()
