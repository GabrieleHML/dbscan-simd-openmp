"""Binary files shared with the C program (format described in src/dataset.h)."""

from pathlib import Path

import numpy as np

HEADER = np.dtype("<i4")


def write_matrix(path: str | Path, values: np.ndarray) -> None:
    """Writes a 2-D array: int32 columns, int32 rows, then the values in row-major order."""
    values = np.ascontiguousarray(values)
    rows, cols = values.shape
    with open(path, "wb") as f:
        np.array([cols, rows], dtype=HEADER).tofile(f)
        values.astype(values.dtype.newbyteorder("<"), copy=False).tofile(f)


def write_dataset(path: str | Path, points: np.ndarray, precision: int = 64) -> None:
    write_matrix(path, points.astype(np.float64 if precision == 64 else np.float32))


def read_dataset(path: str | Path) -> np.ndarray:
    """Returns the points as float64, whatever the precision stored in the file."""
    raw = Path(path).read_bytes()
    cols, rows = np.frombuffer(raw, dtype=HEADER, count=2)
    payload = len(raw) - 8
    dtype = {rows * cols * 4: np.float32, rows * cols * 8: np.float64}.get(payload)
    if dtype is None:
        raise ValueError(f"{path}: size does not match a {rows}x{cols} float32/float64 matrix")
    return np.frombuffer(raw, dtype=np.dtype(dtype).newbyteorder("<"), offset=8).reshape(rows, cols).astype(np.float64)


def read_labels(path: str | Path) -> np.ndarray:
    raw = Path(path).read_bytes()
    cols, rows = np.frombuffer(raw, dtype=HEADER, count=2)
    if cols != 1 or len(raw) != 8 + 4 * rows:
        raise ValueError(f"{path}: not a labels file")
    return np.frombuffer(raw, dtype=HEADER, offset=8).astype(np.int32)
