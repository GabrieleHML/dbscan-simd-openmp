#!/usr/bin/env python3
"""Compares the labels written by the C program with those of scikit-learn's DBSCAN.

Both assign a border point to the first cluster that reaches it and number the
clusters in order of discovery, so on the same input the labels must be identical.
"""

import argparse
import sys

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

from dsio import read_dataset, read_labels


def sklearn_labels(points: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
    return DBSCAN(eps=eps, min_samples=min_pts).fit(points).labels_


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("labels", help="labels written by the C program with -o")
    parser.add_argument("-e", "--eps", type=float, required=True)
    parser.add_argument("-m", "--min-pts", type=int, required=True)
    args = parser.parse_args()

    ours = read_labels(args.labels)
    expected = sklearn_labels(read_dataset(args.dataset), args.eps, args.min_pts)
    mismatches = int(np.count_nonzero(ours != expected))
    print(f"points        {len(ours)}")
    print(f"clusters      {ours.max() + 1} (scikit-learn: {expected.max() + 1})")
    print(f"noise points  {np.count_nonzero(ours == -1)} (scikit-learn: {np.count_nonzero(expected == -1)})")
    print(f"mismatches    {mismatches}")
    print(f"ARI           {adjusted_rand_score(expected, ours):.6f}")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
