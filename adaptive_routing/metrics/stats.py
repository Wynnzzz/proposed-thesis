"""Statistik: bootstrap persentil atas query, bootstrap berpasangan, McNemar eksak (binomial dua sisi)."""
from __future__ import annotations
from typing import Dict, Sequence
import numpy as np

from ..tracing.schema import PolicyOutcome
from .qa import primary_score

STATS = {"mean": lambda x: x.mean(axis=1), "median": lambda x: np.median(x, axis=1),
         "p95": lambda x: np.percentile(x, 95, axis=1)}


def _boot_idx(n: int, B: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def bootstrap_ci(values: Sequence[float], stat: str = "mean", B: int = 10000, level: float = 0.95, seed: int = 0):
    v = np.asarray(values, dtype=float)
    est = STATS[stat](v[None, :])[0]
    dist = STATS[stat](v[_boot_idx(len(v), B, seed)])
    a = (1 - level) / 2
    return float(est), float(np.quantile(dist, a)), float(np.quantile(dist, 1 - a))


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], stat: str = "mean", B: int = 10000,
                          level: float = 0.95, seed: int = 0) -> Dict:
    """Selisih a-b dengan resampling query yang SAMA untuk keduanya."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("a dan b harus berpasangan (panjang sama)")
    idx = _boot_idx(len(a), B, seed)
    dist = STATS[stat](a[idx]) - STATS[stat](b[idx])
    q = (1 - level) / 2
    return {"diff": float(STATS[stat](a[None, :])[0] - STATS[stat](b[None, :])[0]),
            "lo": float(np.quantile(dist, q)), "hi": float(np.quantile(dist, 1 - q))}


def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> Dict:
    """Uji McNemar eksak dua sisi. b = A benar & B salah; c = A salah & B benar."""
    ca, cb = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    if ca.shape != cb.shape:
        raise ValueError("data harus berpasangan")
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    n, k = b + c, min(b, c)
    if n == 0:
        return {"b": b, "c": c, "p": 1.0}
    from math import comb
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"b": b, "c": c, "p": float(p)}


def compare_policies(a: Sequence[PolicyOutcome], b: Sequence[PolicyOutcome], dataset: str, B: int = 10000,
                     level: float = 0.95, seed: int = 0) -> Dict:
    """Perbandingan berpasangan A vs B pada query yang sama. Menolak membandingkan latency_kind berbeda."""
    ka, kb = {o.latency_kind for o in a}, {o.latency_kind for o in b}
    if ka != kb or len(ka) != 1:
        raise ValueError(f"latency_kind berbeda atau bercampur: {sorted(ka)} vs {sorted(kb)}")
    ma, mb = {o.query_id: o for o in a}, {o.query_id: o for o in b}
    if set(ma) != set(mb):
        raise ValueError("himpunan query A dan B tidak sama")
    ids = sorted(ma)
    A, Bo = [ma[i] for i in ids], [mb[i] for i in ids]
    col = lambda outs, f: [f(o) for o in outs]
    sc = lambda o: primary_score(dataset, o.em or 0.0, o.f1)
    return {"n": len(ids), "latency_kind": ka.pop(),
            "score": paired_bootstrap_diff(col(A, sc), col(Bo, sc), "mean", B, level, seed),
            "cost": paired_bootstrap_diff(col(A, lambda o: o.cost_usd), col(Bo, lambda o: o.cost_usd), "mean", B, level, seed),
            "latency": paired_bootstrap_diff(col(A, lambda o: o.latency_s), col(Bo, lambda o: o.latency_s), "mean", B, level, seed),
            "mcnemar_correct": mcnemar_exact(col(A, lambda o: bool(o.correct)), col(Bo, lambda o: bool(o.correct)))}
