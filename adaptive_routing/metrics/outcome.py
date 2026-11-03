"""Hasil kebijakan per query: jawaban dinilai, biaya dan latensi dihitung dari pemanggilan yang dipakai."""
from __future__ import annotations
from ..data.types import Query
from ..routing.accounting import cost_latency
from ..tracing.schema import PolicyOutcome
from .qa import score


def make_outcome(q: Query, result, policy_id: str, latency_kind: str, overhead_s: float = 0.0,
                 measured_latency_s: float | None = None) -> PolicyOutcome:
    """latency_kind='composed_estimated' (replay): latensi dari catatan + overhead.
    latency_kind='measured_e2e' (langsung): measured_latency_s wajib diberikan dan dipakai apa adanya."""
    cost, composed = cost_latency(result.calls, overhead_s)
    if latency_kind == "measured_e2e":
        if measured_latency_s is None:
            raise ValueError("measured_e2e membutuhkan measured_latency_s")
        lat = measured_latency_s
    elif latency_kind == "composed_estimated":
        if measured_latency_s is not None:
            raise ValueError("composed_estimated tidak boleh menerima latensi terukur")
        lat = composed
    else:
        raise ValueError(latency_kind)
    em, f1 = score(q.dataset, result.final_answer, q.gold)
    return PolicyOutcome(q.query_id, policy_id, result.final_answer, em, f1, em == 1.0,
                         list(result.stages_run), len(result.calls), cost, lat, latency_kind)
