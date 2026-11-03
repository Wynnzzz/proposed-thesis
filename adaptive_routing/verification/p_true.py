"""P(True) dari logprobs (Pers. II.5): p = P(Yes | q, y, p_v).

Mendukung dua format yang diketahui (OpenAI 'content' dan Together 'tokens'); format lain
dikembalikan sebagai None (tidak ada tebakan) dan dicatat. Dukungan nyata di backend BELUM
diuji; lihat scripts/probe_backends.py.
"""
from __future__ import annotations
import math
from typing import Any, Dict, Optional, Tuple

_STRIP = "▁Ġ \t\n\r.,!:;\"'`*"


def first_token_alternatives(raw: Any) -> Optional[Dict[str, float]]:
    """{token: logprob} untuk token pertama, atau None bila format tidak dikenali."""
    if not raw:
        return None
    content = raw.get("content") if isinstance(raw, dict) else None
    if isinstance(content, list) and content:
        first = content[0]
        alts = {a["token"]: a["logprob"] for a in first.get("top_logprobs", []) if "token" in a}
        if "token" in first and "logprob" in first:
            alts.setdefault(first["token"], first["logprob"])
        return alts or None
    if isinstance(raw, dict) and raw.get("tokens") and raw.get("token_logprobs"):
        top = raw.get("top_logprobs") or [None]
        alts = dict(top[0]) if top[0] else {}
        alts.setdefault(raw["tokens"][0], raw["token_logprobs"][0])
        return alts or None
    return None


def _norm(tok: str) -> str:
    return tok.strip(_STRIP).lower()


def p_true_from_alternatives(alts: Optional[Dict[str, float]], normalize: str = "yes_over_yes_plus_no"
                             ) -> Tuple[Optional[float], Dict[str, Any]]:
    """Kembalikan (p, meta). p None bila logprobs tidak ada atau Yes dan No sama-sama di luar top-k."""
    meta: Dict[str, Any] = {}
    if not alts:
        return None, {"p_true_status": "no_logprobs"}
    y = sum(math.exp(v) for k, v in alts.items() if _norm(k) == "yes")
    n = sum(math.exp(v) for k, v in alts.items() if _norm(k) == "no")
    meta.update(p_yes_raw=y, p_no_raw=n, p_true_partial=bool((y == 0) != (n == 0)))
    if y == 0 and n == 0:
        meta["p_true_status"] = "yes_no_absent"
        return None, meta
    meta["p_true_status"] = "ok"
    if normalize == "raw_yes":
        return y, meta
    return y / (y + n), meta
