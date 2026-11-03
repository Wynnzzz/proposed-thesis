"""Ablasi A1-A2 (A3 = routing.policy.run_policy). Definisi operasional: DESIGN.md bagian 6 [P5 PROVISIONAL]."""
from __future__ import annotations
from typing import Sequence

from ..routing.policy import PolicyResult, Theta, escalate, stage1_gate


def run_a1(theta: Theta, session, tie_order: Sequence[str]) -> PolicyResult:
    """Tanpa gerbang disagreement: gerbang P(True) (tau1) tetap; bila eskalasi, tahap 2 lalu R ronde debat penuh."""
    res = PolicyResult(None, "s1")
    if stage1_gate(theta, session, res):
        return res
    return escalate(theta, session, tie_order, res, gates=False)


def run_a2(theta: Theta, session, tie_order: Sequence[str]) -> PolicyResult:
    """Tanpa verifikasi awal: semua query ke tahap 2 (gerbang tau2) lalu tahap 3 (gerbang tau3). tau1 tidak dipakai."""
    res = PolicyResult(None, "s2")
    res.escalation_reason = "no_stage1"
    return escalate(theta, session, tie_order, res, gates=True)
