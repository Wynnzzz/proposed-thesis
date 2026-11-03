"""Pengumpulan trace lengkap per query (D2): tahap 1, tahap 2, semua ronde debat 1..rounds,
dan (opsional) pemanggilan baseline B2 dan B3."""
from __future__ import annotations


def collect_full_trace(session, rounds: int, baselines: bool = False, k_sc: int = 5) -> None:
    session.stage1()
    session.stage2()
    session.revision(rounds)          # memicu ronde 1..rounds secara berurutan
    if baselines:
        session.strong()
        session.sc(k_sc)
