"""Disagreement berbasis entropi (Pers. II.6-II.7) dengan penanganan jawaban invalid [P2] dan pemutus seri [P3]."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .entropy import CMP_TOL, entropy_from_counts


@dataclass(frozen=True)
class Disagreement:
    d: float                        # entropi (log natural)
    counts: Tuple[int, ...]         # pola distribusi, urut menurun
    majority: Optional[str]         # jawaban mayoritas bila UNIK dan valid, selain itu None
    unique: bool
    n_invalid: int


def analyze(answers: Sequence[Optional[str]]) -> Disagreement:
    """Invalid = label unik per agen: menaikkan entropi dan tidak pernah membentuk mayoritas."""
    labels = [a if a is not None else f"__invalid_{i}" for i, a in enumerate(answers)]
    cnt = Counter(labels)
    counts = tuple(sorted(cnt.values(), reverse=True))
    top = cnt.most_common()
    unique = len(counts) == 1 or counts[0] > counts[1]
    maj = top[0][0] if unique else None
    if maj is not None and maj.startswith("__invalid_"):   # tidak mungkin untuk N>=2; dijaga eksplisit
        maj, unique = None, False
    return Disagreement(entropy_from_counts(counts), counts, maj, unique,
                        sum(1 for a in answers if a is None))


def accepts(dis: Disagreement, tau: float) -> bool:
    """Aturan Algoritma III.1 baris 5 dan 9: d <= tau (toleransi CMP_TOL) DAN mayoritas unik."""
    return dis.unique and dis.majority is not None and dis.d <= tau + CMP_TOL


def aggregate(answers: Sequence[Optional[str]], node_ids: Sequence[str],
              tie_order: Sequence[str]) -> Optional[str]:
    """Mayoritas atas jawaban valid; seri diputus oleh agen berperingkat tertinggi (tie_order).
    Tidak ada jawaban valid -> None (dihitung salah)."""
    valid = [(a, n) for a, n in zip(answers, node_ids) if a is not None]
    if not valid:
        return None
    cnt = Counter(a for a, _ in valid)
    best = max(cnt.values())
    tied = {a for a, c in cnt.items() if c == best}
    rank = {n: i for i, n in enumerate(tie_order)}
    big = len(rank) + 1

    def best_rank(a: str) -> Tuple[int, str]:
        return (min(rank.get(n, big) for x, n in valid if x == a), a)   # lalu leksikografis

    return min(tied, key=best_rank)
