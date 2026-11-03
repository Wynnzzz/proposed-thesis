"""Router tiga tahap (Algoritma III.1): SATU-SATUNYA sumber keputusan untuk eksekusi langsung dan replay.

Perbandingan: P(True) >= tau1 menerima jawaban SLM; entropi <= tau (toleransi CMP_TOL) DAN mayoritas unik
menerima/menghentikan pada tahap 2 dan 3. Jawaban SLM invalid atau P(True) tak terdefinisi memaksa eskalasi [P2].
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .disagreement import accepts, aggregate, analyze


@dataclass(frozen=True)
class Theta:
    tau1: float                 # boleh -inf (selalu terima) atau +inf (tidak pernah terima)
    tau2: float
    tau3: float
    N: int
    R: int
    S: str                      # heterogeneous | homogeneous

    def __post_init__(self):
        if self.R < 1:
            raise ValueError("R minimal 1")


@dataclass
class PolicyResult:
    final_answer: Optional[str]
    stopped_at: str             # s1 | s2 | s3_r{r} | s3_end
    stages_run: List[str] = field(default_factory=list)
    calls: List = field(default_factory=list)
    disagreement: List[float] = field(default_factory=list)   # entropi tiap tahap multi-agen yang dievaluasi
    escalation_reason: Optional[str] = None


def stage1_gate(theta: Theta, session, res: PolicyResult) -> bool:
    """Tahap 1. True bila jawaban SLM diterima (res terisi); False bila harus eskalasi (alasan dicatat)."""
    s1 = session.stage1()
    res.stages_run.append("s1")
    res.calls += s1.calls
    if s1.answer is None:
        res.escalation_reason = "slm_invalid"
    elif s1.p_true is None:
        res.escalation_reason = "p_true_undefined"
    elif s1.p_true >= theta.tau1:
        res.final_answer, res.stopped_at = s1.answer, "s1"
        return True
    else:
        res.escalation_reason = "low_p_true"
    return False


def escalate(theta: Theta, session, tie_order: Sequence[str], res: PolicyResult, gates: bool = True) -> PolicyResult:
    """Tahap 2 dan 3. gates=False: tanpa penghentian dini berbasis disagreement (B4, A1): R ronde penuh."""
    s2 = session.stage2()
    res.stages_run.append("s2")
    res.calls += s2.calls
    dis = analyze(s2.answers)
    res.disagreement.append(dis.d)
    if gates and accepts(dis, theta.tau2):
        res.final_answer, res.stopped_at = dis.majority, "s2"
        return res
    last = s2
    for r in range(1, theta.R + 1):
        rr = session.revision(r)
        res.stages_run.append(f"s3_r{r}")
        res.calls += rr.calls
        last = rr
        dis = analyze(rr.answers)
        res.disagreement.append(dis.d)
        if gates and accepts(dis, theta.tau3):
            res.final_answer, res.stopped_at = dis.majority, f"s3_r{r}"
            return res
    res.final_answer = aggregate(last.answers, last.node_ids, tie_order)
    res.stopped_at = "s3_end"
    return res


def run_policy(theta: Theta, session, tie_order: Sequence[str]) -> PolicyResult:
    """A3: kebijakan penuh (Algoritma III.1)."""
    res = PolicyResult(None, "s1")
    if stage1_gate(theta, session, res):
        return res
    return escalate(theta, session, tie_order, res, gates=True)
