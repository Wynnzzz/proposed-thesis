"""Pembangun pesan untuk agen (jawaban independen, verifikasi, revisi)."""
from __future__ import annotations
from typing import Dict, List

from ..data.types import Query
from ..prompts import templates as T


def system_msg(dataset: str) -> Dict[str, str]:
    return {"role": "system", "content": T.SYSTEM.format(fmt=T.FORMAT[dataset])}


def answer_messages(q: Query) -> List[Dict[str, str]]:
    return [system_msg(q.dataset), {"role": "user", "content": T.ANSWER_USER.format(body=q.body)}]


def verify_messages(q: Query, answer: str) -> List[Dict[str, str]]:
    # verifikasi memakai jawaban final yang diekstrak (y-topi pada Pers. II.5), bukan seluruh penalaran
    return [{"role": "user", "content": T.VERIFY_USER.format(body=q.body, answer=answer)}]
