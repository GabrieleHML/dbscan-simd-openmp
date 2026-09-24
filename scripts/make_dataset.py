#!/usr/bin/env python3
"""Generates a dataset for the C program.

blobs:   Gaussian clusters plus a share of uniform noise, the typical DBSCAN input.
uniform: points uniformly distributed in the unit hypercube, like the dataset used
         for the exam, where no density-based structure exists.
"""

import argparse

import numpy as np
from sklearn.datasets import make_blobs

from dsio import write_dataset


def blobs(n: int, d: int, centers: int, std: float, noise: float, rng: np.random.Generator) -> np.ndarray:
    n_noise = int(n * noise)
    points, _ = make_blobs(
        n_samples=n - n_noise,
        n_features=d,
        centers=centers,
        cluster_std=std,
        center_box=(-10 * std * centers ** (1 / d), 10 * std * centers ** (1 / d)),
        random_state=int(rng.integers(2**31)),
    )
    low, high = points.min(axis=0), points.max(axis=0)
    points = np.vstack([points, rng.uniform(low, high, size=(n_noise, d))])
    return points[rng.permutation(n)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", help="dataset file to write")
    parser.add_argument("--kind", choices=["blobs", "uniform"], default="blobs")
    parser.add_argument("-n", type=int, default=10_000, help="number of points (default: 10000)")
    parser.add_argument("-d", type=int, default=8, help="number of dimensions (default: 8)")
    parser.add_argument("--centers", type=int, default=5, help="blobs: number of clusters (default: 5)")
    parser.add_argument("--std", type=float, default=1.0, help="blobs: standard deviation of a cluster")
    parser.add_argument("--noise", type=float, default=0.05, help="blobs: share of noise points (default: 0.05)")
    parser.add_argument("--precision", type=int, choices=[32, 64], default=64, help="bits per value (default: 64)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.kind == "blobs":
        points = blobs(args.n, args.d, args.centers, args.std, args.noise, rng)
    else:
        points = rng.random((args.n, args.d))
    write_dataset(args.output, points, args.precision)


if __name__ == "__main__":
    main()
