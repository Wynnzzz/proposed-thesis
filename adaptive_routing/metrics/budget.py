"""Estimator biaya dari pilot. Semua angka adalah EKSTRAPOLASI dari n pilot kecil, bukan jaminan."""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Sequence
import numpy as np

from ..tracing.schema import CallRecord


def paid_cost_per_query(calls: Sequence[CallRecord]) -> Dict[str, float]:
    """Biaya yang benar-benar dibayar per query (cache hit tidak dihitung)."""
    d = defaultdict(float)
    for c in calls:
        if not c.cache_hit:
            d[c.query_id] += c.cost_usd or 0.0
    return dict(d)


def project(per_query: Sequence[float], n_target: int, safety: float = 1.5) -> Dict[str, float]:
    v = np.asarray(list(per_query), float)
    if v.size == 0:
        raise ValueError("tidak ada data pilot")
    return {"n_pilot": int(v.size), "mean_per_query": float(v.mean()), "p95_per_query": float(np.percentile(v, 95)),
            "expected_total": float(v.mean() * n_target), "upper_total": float(max(v.mean(), np.percentile(v, 95)) * n_target * safety),
            "note": f"ekstrapolasi dari {v.size} query pilot, faktor aman {safety}"}


def reasoning_share(calls: Sequence[CallRecord]) -> Dict[str, float]:
    tot, rs = defaultdict(int), defaultdict(int)
    for c in calls:
        if c.output_tokens:
            tot[c.model_key] += c.output_tokens
            rs[c.model_key] += c.reasoning_tokens or 0
    return {k: rs[k] / tot[k] for k in tot}


def direct_test_estimate(rows_selected: Sequence[Dict], n_test: int, repeats: int = 1, safety: float = 1.5) -> Dict[str, float]:
    """Biaya eksekusi langsung test ~ rata-rata biaya replay dev per kebijakan x n_test x repeats."""
    per = {r["policy_id"]: r["cost"] * n_test * repeats for r in rows_selected}
    return {"per_policy": per, "expected_total": float(sum(per.values())), "upper_total": float(sum(per.values()) * safety)}
