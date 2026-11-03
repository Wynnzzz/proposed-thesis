"""Seleksi kebijakan pada DEV per skenario yang dinyatakan sebelum replay, lalu dibekukan (hash isi) sebelum test.

Aturan: dalam tiap keluarga (A3, A1, A2, B4, B5), pilih kebijakan layak (batas biaya relatif B2 dan/atau batas latensi
relatif B4 acuan) dengan error terkecil; seri: biaya, latensi, policy_id. Bila tidak ada yang layak: None (batas TIDAK
dilonggarkan diam-diam). Argmin leksikografis demikian Pareto-optimal DALAM keluarganya (bukan jaminan global: kebijakan keluarga lain
dapat mendominasinya; frontier global dilaporkan terpisah)."""
from __future__ import annotations
import datetime as dt, hashlib, json
from typing import Any, Dict, List, Optional, Sequence

from ..baselines.registry import PolicySpec, spec_to_dict
from ..tracing.schema import PolicyOutcome
from .pareto import pareto_indices
from .summary import summarize

TOL = 1e-12


class SelectionError(ValueError):
    pass


def make_rows(specs: Sequence[PolicySpec], outcomes: Dict[str, List[PolicyOutcome]], dataset: str) -> List[Dict[str, Any]]:
    rows = []
    for sp in specs:
        s = summarize(outcomes[sp.policy_id], dataset)
        rows.append({"policy_id": sp.policy_id, "kind": sp.kind, "spec": spec_to_dict(sp),
                     "cost": s["cost_mean"], "latency": s["latency_mean"], "error": s["error"], "summary": s})
    return rows


def objective_matrix(rows):
    return [[r["cost"], r["latency"], r["error"]] for r in rows]


def frontier(rows) -> List[Dict[str, Any]]:
    return [rows[i] for i in pareto_indices(objective_matrix(rows))]


def references(rows: Sequence[Dict], cfg_raw: dict) -> Dict[str, Dict]:
    by = {r["policy_id"]: r for r in rows}
    rb = cfg_raw["protocol"]["selection"]["reference_b4"]
    b4 = f"B4|N={rb['N']}|R={rb['R']}|S={'het' if rb['S'] == 'heterogeneous' else 'hom'}"
    for need in ("B2", b4):
        if need not in by:
            raise SelectionError(f"acuan {need} tidak ada pada hasil replay")
    return {"B2": by["B2"], "B4_ref": by[b4]}


def feasible(row, scenario, refs) -> bool:
    c, l = scenario.get("max_cost_ratio_vs_B2"), scenario.get("max_latency_ratio_vs_B4")
    if c is not None and row["cost"] > c * refs["B2"]["cost"] + TOL:
        return False
    if l is not None and row["latency"] > l * refs["B4_ref"]["latency"] + TOL:
        return False
    return True


def select_scenarios(rows: Sequence[Dict], cfg_raw: dict) -> Dict[str, Dict[str, Optional[Dict]]]:
    sel = cfg_raw["protocol"]["selection"]
    refs = references(rows, cfg_raw)
    out: Dict[str, Dict[str, Optional[Dict]]] = {}
    for sc in sel["scenarios"]:
        out[sc["name"]] = {}
        for kind in sel["families"]:
            cands = [r for r in rows if r["kind"] == kind and feasible(r, sc, refs)]
            out[sc["name"]][kind] = (min(cands, key=lambda r: (r["error"], r["cost"], r["latency"], r["policy_id"]))
                                     if cands else None)
    return out


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode()


def freeze_selection(path: str, payload: Dict[str, Any]) -> str:
    body = dict(payload, frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    digest = hashlib.sha256(_canon(body)).hexdigest()
    with open(path, "w", encoding="utf-8") as f:
        json.dump({**body, "content_sha256": digest}, f, indent=2, ensure_ascii=False, default=str)
    return digest


def verify_selection(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    digest = d.pop("content_sha256", None)
    if digest != hashlib.sha256(_canon(d)).hexdigest():
        raise SelectionError("selected_policies.json berubah setelah dibekukan (hash tidak cocok)")
    return d
