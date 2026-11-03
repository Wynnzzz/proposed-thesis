"""Definisi node agen untuk komposisi S dan jumlah N. Satu fungsi dipakai sesi live dan trace."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    model_key: str
    temperature: float
    sample_idx: Optional[int]


def agent_nodes(cfg_raw: Dict[str, Any], S: str, N: int) -> List[NodeSpec]:
    b = cfg_raw["backends"]
    if S == "heterogeneous":
        if N > len(b["agents"]):
            raise ValueError(f"N={N} melebihi jumlah agen heterogen ({len(b['agents'])})")
        return [NodeSpec(a["key"], a["key"], a["temperature"], None) for a in b["agents"][:N]]
    if S == "homogeneous":
        h = b["homogeneous_variant"]
        return [NodeSpec(f"{h['agent_key']}#{i}", h["agent_key"], h["temperature"], i) for i in range(N)]
    raise ValueError(f"S tidak dikenal: {S}")


def model_cfg(cfg_raw: Dict[str, Any], model_key: str) -> Dict[str, Any]:
    b = cfg_raw["backends"]
    for m in [b["slm"], *b["agents"], b["strong"]]:
        if m["key"] == model_key:
            return m
    raise KeyError(model_key)
