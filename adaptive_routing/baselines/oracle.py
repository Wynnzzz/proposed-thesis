"""B7 oracle (analisis batas atas, BUKAN baseline operasional): per query memilih hasil terbaik di antara kandidat
berdasarkan label gold. Urutan: skor utama tertinggi, lalu biaya terendah, lalu latensi terendah, lalu policy_id."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import replace
from typing import Dict, List

from ..metrics.qa import primary_score
from ..tracing.schema import PolicyOutcome


def oracle_outcomes(by_policy: Dict[str, List[PolicyOutcome]], dataset: str) -> List[PolicyOutcome]:
    per_q = defaultdict(list)
    for pid, outs in by_policy.items():
        for o in outs:
            per_q[o.query_id].append((pid, o))
    result = []
    for qid in sorted(per_q):
        cands = per_q[qid]
        if len({pid for pid, _ in cands}) != len(by_policy):
            raise ValueError(f"{qid}: tidak semua kandidat memiliki hasil (oracle butuh himpunan lengkap)")
        pid, o = min(cands, key=lambda c: (-primary_score(dataset, c[1].em or 0.0, c[1].f1), c[1].cost_usd,
                                           c[1].latency_s, c[0]))
        result.append(replace(o, policy_id="B7", stages_run=[f"oracle:{pid}"], latency_kind="composed_estimated"))
    return result
