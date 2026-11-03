"""Baseline B1-B5 (DESIGN.md bagian 6). Setiap fungsi memakai antarmuka sesi yang sama dengan router,
sehingga berjalan identik pada LiveSession (langsung) dan TraceSession (replay)."""
from __future__ import annotations
from typing import Sequence

from ..routing.disagreement import aggregate
from ..routing.policy import PolicyResult, Theta, escalate


def run_b1(session) -> PolicyResult:
    r = session.slm_answer()
    return PolicyResult(r.answer, "b1", ["b1"], list(r.calls))


def run_b2(session) -> PolicyResult:
    r = session.strong()
    return PolicyResult(r.answer, "b2", ["b2"], list(r.calls))


def run_b3(session, k: int) -> PolicyResult:
    r = session.sc(k)
    order = [f"slm#{i}" for i in range(k)]          # seri: sampel berindeks terkecil
    return PolicyResult(aggregate(r.answers, r.node_ids, order), "b3", ["b3"], list(r.calls))


def run_b4(theta: Theta, session, tie_order: Sequence[str]) -> PolicyResult:
    """MAD tetap: tahap 2 lalu R ronde pada SEMUA query, tanpa verifikasi dan tanpa penghentian dini."""
    res = PolicyResult(None, "s3_end")
    res.escalation_reason = "fixed_mad"
    return escalate(theta, session, tie_order, res, gates=False)


def run_b5(tau: float, session) -> PolicyResult:
    """Cascade P(True): SLM + verifikasi; bila p < tau (atau invalid/tak terdefinisi) eskalasi ke model kuat."""
    res = PolicyResult(None, "b5_s1", ["s1"])
    s1 = session.stage1()
    res.calls += s1.calls
    if s1.answer is not None and s1.p_true is not None and s1.p_true >= tau:
        res.final_answer = s1.answer
        return res
    res.escalation_reason = ("slm_invalid" if s1.answer is None else
                             "p_true_undefined" if s1.p_true is None else "low_p_true")
    st = session.strong()
    res.stages_run.append("b2")
    res.calls += st.calls
    res.final_answer, res.stopped_at = st.answer, "b5_strong"
    return res
