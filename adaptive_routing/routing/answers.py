"""Ekstraksi dan normalisasi jawaban (P10, P11).

Format ketat: baris "Final answer: X". Kegagalan parse = jawaban INVALID (None); tidak ada tebakan.
MMLU : satu huruf A-D. HotpotQA: string pendek ternormalisasi gaya SQuAD (huruf kecil, tanpa
tanda baca dan artikel). Pengelompokan mayoritas HotpotQA = kecocokan string ternormalisasi,
sehingga sinonim dihitung berbeda (keterbatasan yang dilaporkan).
"""
from __future__ import annotations
import re, string
from typing import Optional

_FINAL = re.compile(r"final\s*answer\s*[:：]\s*(.+)", re.IGNORECASE)
_LETTER = re.compile(r"^[\s\*\(\[\"'`]*([ABCD])(?:[\s\*\)\]\.\:\"'`]*$|[\.\):]\s)", re.IGNORECASE)
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


def extract_final(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = _FINAL.findall(text)
    if not m:
        return None
    last = m[-1].strip()
    return last or None


def normalize_text(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def parse_mmlu(text: Optional[str]) -> Optional[str]:
    raw = extract_final(text)
    if raw is None:
        return None
    m = _LETTER.match(raw)
    return m.group(1).upper() if m else None


def parse_hotpotqa(text: Optional[str]) -> Optional[str]:
    raw = extract_final(text)
    if raw is None:
        return None
    raw = raw.strip(" *`\"'")
    n = normalize_text(raw)
    return n or None


def parse_answer(dataset: str, text: Optional[str]) -> Optional[str]:
    if dataset == "mmlu":
        return parse_mmlu(text)
    if dataset == "hotpotqa":
        return parse_hotpotqa(text)
    raise ValueError(f"dataset tidak dikenal: {dataset}")
