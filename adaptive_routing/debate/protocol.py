"""Mekanisme debat: tiap agen melihat respons LENGKAP agen lain pada ronde sebelumnya, lalu merevisi (Du dkk.).
Penghentian dan agregasi diatur router (routing/policy.py) agar satu sumber keputusan."""
from __future__ import annotations
from typing import Dict, List, Sequence

from ..data.types import Query
from ..prompts import templates as T
from ..agents.agent import system_msg


def revision_messages(q: Query, idx: int, prev_texts: Sequence[str], node_ids: Sequence[str]) -> List[Dict[str, str]]:
    others = "\n\n".join(f"[Agent {j + 1}]\n{t.strip() or '(no response)'}"
                         for j, t in enumerate(prev_texts) if j != idx)
    own = prev_texts[idx].strip() or "(no response)"
    return [system_msg(q.dataset),
            {"role": "user", "content": T.REVISION_USER.format(body=q.body, own=own, others=others,
                                                              fmt=T.FORMAT[q.dataset])}]
