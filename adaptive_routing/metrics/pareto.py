"""Dominasi Pareto (minimisasi). Titik identik tidak saling mendominasi."""
from __future__ import annotations
from typing import List
import numpy as np


def dominates(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.all(a <= b) and np.any(a < b))


def pareto_indices(points) -> List[int]:
    P = np.asarray(points, dtype=float)
    out = []
    for i in range(len(P)):
        dom = np.all(P <= P[i], axis=1) & np.any(P < P[i], axis=1)
        if not dom.any():
            out.append(i)
    return out
