"""Replay dev: rekonstruksi keputusan dan hasil SEMUA kebijakan dari trace lengkap. Latensi = composed_estimated.
Menolak data selain dev (D2: test tidak pernah dipakai untuk memilih apa pun)."""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Sequence

from ..baselines.registry import PolicySpec, run_spec
from ..data.types import Query
from ..metrics.outcome import make_outcome
from ..routing.sessions import TraceSession
from ..tracing.schema import CallRecord, PolicyOutcome


def replay_dev(queries: Sequence[Query], calls: Sequence[CallRecord], cfg_raw: dict, specs: Sequence[PolicySpec],
               tie_orders: Dict[tuple, List[str]], overhead_s: float = 0.0) -> Dict[str, List[PolicyOutcome]]:
    if any(q.split != "dev" for q in queries) or any(c.split != "dev" for c in calls):
        raise ValueError("replay hanya untuk split dev")
    by_q = defaultdict(list)
    for c in calls:
        by_q[c.query_id].append(c)
    nmax = max(cfg_raw["policy_space"]["N"])
    sessions: Dict[tuple, TraceSession] = {}
    out: Dict[str, List[PolicyOutcome]] = {}
    for spec in specs:
        S, N = (spec.theta.S, spec.theta.N) if spec.theta else ("heterogeneous", nmax)
        tie = tie_orders.get((S, N), [])
        res = []
        for q in queries:
            key = (q.query_id, S, N)
            if key not in sessions:
                sessions[key] = TraceSession(q, by_q[q.query_id], S, N, cfg_raw)
            r = run_spec(spec, sessions[key], tie)
            res.append(make_outcome(q, r, spec.policy_id, "composed_estimated", overhead_s))
        out[spec.policy_id] = res
    return out
