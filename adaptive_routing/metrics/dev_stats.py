"""Statistik yang boleh dihitung HANYA dari dev: peringkat agen untuk pemutus seri (P3)."""
from __future__ import annotations
from typing import Dict, List, Sequence

from ..data.types import Query
from ..routing.nodes import agent_nodes
from ..tracing.schema import CallRecord
from .qa import primary_score, score


def rank_nodes_by_dev_accuracy(queries: Sequence[Query], calls: Sequence[CallRecord], S: str, N: int,
                               cfg_raw: dict) -> List[str]:
    if any(q.split != "dev" for q in queries):
        raise ValueError("peringkat agen untuk pemutus seri hanya boleh dihitung dari split dev")
    nodes = [n.node_id for n in agent_nodes(cfg_raw, S, N)]
    qmap = {q.query_id: q for q in queries}
    tot: Dict[str, float] = {n: 0.0 for n in nodes}
    cnt: Dict[str, int] = {n: 0 for n in nodes}
    for c in calls:
        if c.stage == "s2_agent" and c.node_id in tot and c.query_id in qmap:
            q = qmap[c.query_id]
            em, f1 = score(q.dataset, c.parsed_answer if c.parse_ok else None, q.gold)
            tot[c.node_id] += primary_score(q.dataset, em, f1)
            cnt[c.node_id] += 1
    mean = {n: (tot[n] / cnt[n] if cnt[n] else 0.0) for n in nodes}
    return sorted(nodes, key=lambda n: (-mean[n], nodes.index(n)))     # seri: urutan config
