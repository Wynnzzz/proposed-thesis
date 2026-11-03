"""Evaluator jawaban. MMLU: akurasi (pencocokan huruf). HotpotQA: EM dan F1 gaya resmi HotpotQA/SQuAD
(normalisasi, F1 token; untuk yes/no/noanswer tidak ada kredit parsial bila tidak identik).
Implementasi ditulis ulang dari definisi metrik, bukan disalin; verifikasi terhadap skrip evaluasi resmi dianjurkan."""
from __future__ import annotations
from collections import Counter
from typing import Any, Optional, Tuple

from ..routing.answers import normalize_text

SPECIAL = {"yes", "no", "noanswer"}


def f1_tokens(pred: str, gold: str) -> float:
    if pred in SPECIAL and pred != gold:
        return 0.0
    if gold in SPECIAL and pred != gold:
        return 0.0
    p, g = pred.split(), gold.split()
    common = Counter(p) & Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(p), ns / len(g)
    return 2 * prec * rec / (prec + rec)


def score(dataset: str, pred: Optional[str], gold: Any) -> Tuple[float, Optional[float]]:
    """Kembalikan (em, f1). MMLU: em = akurasi, f1 = None. pred HotpotQA harus sudah dinormalisasi (parse)."""
    if dataset == "mmlu":
        return (1.0 if pred is not None and pred == str(gold).strip().upper() else 0.0), None
    if dataset == "hotpotqa":
        g = normalize_text(str(gold))
        if pred is None or not g:
            return 0.0, 0.0
        return (1.0 if pred == g else 0.0), f1_tokens(pred, g)
    raise ValueError(f"dataset tidak dikenal: {dataset}")


def primary_score(dataset: str, em: float, f1: Optional[float]) -> float:
    """Skor utama untuk objektif error (P4): MMLU akurasi; HotpotQA F1. error = 1 - primary_score."""
    return em if dataset == "mmlu" else float(f1 or 0.0)
