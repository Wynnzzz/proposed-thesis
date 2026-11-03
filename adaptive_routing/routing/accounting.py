"""Biaya dan latensi dari pemanggilan yang DIPAKAI oleh suatu kebijakan (bagian 5 DESIGN.md).

cost    = jumlah cost_usd (error = 0, tidak ditagih).
latency = sum per grup( max latency_s dalam grup ) + overhead untuk grup tahap 2 dan ronde debat.
Dipakai oleh replay (composed_estimated) dan sebagai pemeriksa silang eksekusi langsung.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Iterable, Tuple

OVERHEAD_PREFIXES = ("s2", "s3_r")


def cost_latency(calls: Iterable, overhead_s: float = 0.0) -> Tuple[float, float]:
    groups = defaultdict(list)
    cost = 0.0
    for c in calls:
        cost += c.cost_usd or 0.0
        groups[c.group_id].append(c.latency_s)
    lat = 0.0
    for g, ls in groups.items():
        lat += max(ls) + (overhead_s if g.startswith(OVERHEAD_PREFIXES) else 0.0)
    return cost, lat
