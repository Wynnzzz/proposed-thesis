"""Entropi disagreement dan ruang ambang diskret [D4].

Hanya fungsi murni di Tahap 2 (dipakai validasi konfigurasi); router memakainya di Tahap 3.
Log natural. Perbandingan ambang: d <= tau + CMP_TOL.
CMP_TOL=1e-4 diperlukan karena ambang dari konfigurasi dibulatkan (mis. 0.6365 vs eksak 0.636514);
jarak antar level entropi diskret jauh lebih besar (>0.1), sehingga toleransi ini tidak mengubah keputusan.
"""
from __future__ import annotations
import math
from typing import Iterable, List, Tuple

CMP_TOL = 1e-4   # toleransi perbandingan ambang
ZERO_TOL = 1e-12


def entropy_from_counts(counts: Iterable[int]) -> float:
    c = [x for x in counts if x > 0]
    n = sum(c)
    if n == 0:
        return 0.0
    h = -sum((x / n) * math.log(x / n) for x in c)
    return 0.0 if abs(h) < ZERO_TOL else h


def partitions(n: int, max_part: int | None = None) -> List[Tuple[int, ...]]:
    """Semua partisi n (pola distribusi jawaban antar n agen)."""
    max_part = n if max_part is None else max_part
    if n == 0:
        return [()]
    out: List[Tuple[int, ...]] = []
    for k in range(min(n, max_part), 0, -1):
        out += [(k,) + p for p in partitions(n - k, k)]
    return out


def unique_majority(counts: Tuple[int, ...]) -> bool:
    s = sorted(counts, reverse=True)
    return len(s) == 1 or s[0] > s[1]


def achievable_entropies(n: int) -> List[float]:
    """Nilai entropi berbeda yang mungkin untuk n agen, urut naik."""
    vals = sorted({round(entropy_from_counts(p), 12) for p in partitions(n)})
    return vals


def accepted_patterns(n: int, tau: float, require_unique_majority: bool = True):
    return frozenset(
        p for p in partitions(n)
        if entropy_from_counts(p) <= tau + CMP_TOL and (unique_majority(p) or not require_unique_majority)
    )


def effective_thresholds(n: int, require_unique_majority: bool = True) -> List[float]:
    """Ambang kanonik: satu per himpunan pola yang diterima (ambang ekuivalen dilebur)."""
    seen, out = {}, []
    for t in achievable_entropies(n):
        key = accepted_patterns(n, t, require_unique_majority)
        if key not in seen:
            seen[key] = t
            out.append(t)
    return out
