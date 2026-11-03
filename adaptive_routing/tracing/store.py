"""Penyimpanan trace per run: manifest.json, queries.jsonl, calls.jsonl (bagian 4 DESIGN.md)."""
from __future__ import annotations
import dataclasses, json, os
from typing import List, Sequence, Tuple

from ..data.io import read_queries, write_queries
from ..data.types import Query
from .schema import CallRecord, TraceManifest, append_jsonl, read_jsonl


class TraceError(ValueError):
    pass


def write_trace(path: str, manifest: TraceManifest, queries: Sequence[Query], calls: Sequence[CallRecord]) -> None:
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(manifest), f, indent=2, ensure_ascii=False, default=str)
    write_queries(os.path.join(path, "queries.jsonl"), queries)
    cp = os.path.join(path, "calls.jsonl")
    open(cp, "w").close()
    append_jsonl(cp, calls)


def read_trace(path: str) -> Tuple[dict, List[Query], List[CallRecord]]:
    with open(os.path.join(path, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    qs = read_queries(os.path.join(path, "queries.jsonl"))
    calls = [CallRecord.from_dict(d) for d in read_jsonl(os.path.join(path, "calls.jsonl"))]
    return man, qs, calls


def validate_trace(man: dict, queries: Sequence[Query], calls: Sequence[CallRecord], expect_split: str = "dev") -> None:
    if man["split"] != expect_split:
        raise TraceError(f"manifest.split={man['split']!r}, diharapkan {expect_split!r}")
    if any(q.split != expect_split for q in queries):
        raise TraceError(f"queries memuat split selain {expect_split}")
    if any(c.split != expect_split for c in calls):
        raise TraceError(f"calls memuat split selain {expect_split}")
    ids = {q.query_id for q in queries}
    if set(man["query_ids"]) != ids:
        raise TraceError("manifest.query_ids tidak cocok dengan queries.jsonl")
    stray = {c.query_id for c in calls} - ids
    if stray:
        raise TraceError(f"calls merujuk query tak dikenal: {sorted(stray)[:3]}")
