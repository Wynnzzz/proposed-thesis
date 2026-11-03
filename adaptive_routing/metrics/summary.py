"""Ringkasan metrik satu kebijakan pada satu benchmark. Latensi: persentil memakai interpolasi linier (numpy default).
Melarang pencampuran latency_kind (composed vs measured) dalam satu ringkasan."""
from __future__ import annotations
from typing import Dict, Sequence
import numpy as np

from ..tracing.schema import PolicyOutcome
from .qa import primary_score


def summarize(outcomes: Sequence[PolicyOutcome], dataset: str) -> Dict:
    if not outcomes:
        raise ValueError("outcomes kosong")
    kinds = {o.latency_kind for o in outcomes}
    if len(kinds) != 1:
        raise ValueError(f"latency_kind bercampur: {sorted(kinds)}")
    score = np.array([primary_score(dataset, o.em or 0.0, o.f1) for o in outcomes])
    lat = np.array([o.latency_s for o in outcomes])
    cost = np.array([o.cost_usd for o in outcomes])
    s = {"n": len(outcomes), "latency_kind": kinds.pop(),
         "em": float(np.mean([o.em or 0.0 for o in outcomes])),
         "score": float(score.mean()), "error": float(1.0 - score.mean()),
         "cost_mean": float(cost.mean()), "cost_total": float(cost.sum()),
         "latency_mean": float(lat.mean()), "latency_median": float(np.median(lat)),
         "latency_p95": float(np.percentile(lat, 95)), "n_calls_mean": float(np.mean([o.n_calls for o in outcomes])),
         "none_rate": float(np.mean([o.final_answer is None for o in outcomes]))}
    if dataset == "hotpotqa":
        s["f1"] = float(np.mean([o.f1 or 0.0 for o in outcomes]))
    return s
