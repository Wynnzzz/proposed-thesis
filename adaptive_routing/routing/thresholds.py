"""Kandidat ambang tau1 dari distribusi P(True) pada DEV (kuantil + sentinel). Ruang diskret [D4]."""
from __future__ import annotations
from typing import Iterable, List
import numpy as np


def tau1_candidates(p_values: Iterable[float], levels: Iterable[float], include_sentinels: bool = True) -> List[float]:
    """Ambang = nilai P(True) teramati pada dev (metode 'higher'), sehingga keputusan p >= tau berubah hanya
    pada nilai yang memang ada. Sentinel: -inf (selalu terima) dan +inf (tidak pernah terima)."""
    vals = np.asarray([p for p in p_values if p is not None], dtype=float)
    taus = set()
    if vals.size:
        for q in levels:
            taus.add(float(np.quantile(vals, q, method="higher")))
    if include_sentinels:
        taus |= {float("-inf"), float("inf")}
    return sorted(taus)
