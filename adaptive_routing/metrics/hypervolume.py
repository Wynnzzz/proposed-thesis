"""Hypervolume eksak (minimisasi) untuk 3 objektif, dengan titik referensi. Titik yang tidak lebih baik dari referensi
pada SEMUA objektif diabaikan. Normalisasi min-max atas gabungan titik dev (bagian 7 DESIGN.md)."""
from __future__ import annotations
from typing import Sequence, Tuple
import numpy as np


def _hv2d(P: np.ndarray, ref: Tuple[float, float]) -> float:
    if len(P) == 0:
        return 0.0
    P = P[np.argsort(P[:, 0], kind="stable")]
    hv, best_y = 0.0, ref[1]
    for i, (x, y) in enumerate(P):
        best_y = min(best_y, y)
        nx = P[i + 1, 0] if i + 1 < len(P) else ref[0]
        hv += (nx - x) * (ref[1] - best_y)
    return float(hv)


def hypervolume(points, ref: Sequence[float]) -> float:
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    ref = np.asarray(ref, dtype=float)
    P = P[np.all(P < ref, axis=1)]
    if len(P) == 0:
        return 0.0
    P = P[np.argsort(P[:, 0], kind="stable")]
    hv = 0.0
    for i in range(len(P)):
        nx = P[i + 1, 0] if i + 1 < len(P) else ref[0]
        if nx > P[i, 0]:
            hv += (nx - P[i, 0]) * _hv2d(P[: i + 1, 1:], (ref[1], ref[2]))
    return float(hv)


def minmax_bounds(points) -> Tuple[np.ndarray, np.ndarray]:
    P = np.asarray(points, dtype=float)
    return P.min(axis=0), P.max(axis=0)


def normalize(points, lo, hi) -> np.ndarray:
    P, lo, hi = np.asarray(points, float), np.asarray(lo, float), np.asarray(hi, float)
    span = np.where(hi > lo, hi - lo, 1.0)
    return (P - lo) / span
