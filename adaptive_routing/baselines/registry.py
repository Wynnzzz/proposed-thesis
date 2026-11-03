"""Registri kebijakan: spesifikasi deklaratif + enumerasi dari config. Dipakai replay (Tahap 5) dan runner langsung (Tahap 6)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ..ablations.variants import run_a1, run_a2
from ..routing.entropy import effective_thresholds
from ..routing.policy import PolicyResult, Theta, run_policy
from .policies import run_b1, run_b2, run_b3, run_b4, run_b5

KINDS = ("B1", "B2", "B3", "B4", "B5", "A1", "A2", "A3")


def _f(x: float) -> str:
    return "-inf" if x == float("-inf") else "+inf" if x == float("inf") else f"{x:.4g}"


@dataclass(frozen=True)
class PolicySpec:
    kind: str
    theta: Optional[Theta] = None        # B4, A1, A2, A3
    tau: Optional[float] = None          # B5
    k: Optional[int] = None              # B3

    @property
    def policy_id(self) -> str:
        t = self.theta
        if self.kind in ("B1", "B2"):
            return self.kind
        if self.kind == "B3":
            return f"B3|k={self.k}"
        if self.kind == "B5":
            return f"B5|t={_f(self.tau)}"
        S = "het" if t.S == "heterogeneous" else "hom"
        if self.kind == "B4":
            return f"B4|N={t.N}|R={t.R}|S={S}"
        if self.kind == "A1":
            return f"A1|t1={_f(t.tau1)}|N={t.N}|R={t.R}|S={S}"
        if self.kind == "A2":
            return f"A2|t2={_f(t.tau2)}|t3={_f(t.tau3)}|N={t.N}|R={t.R}|S={S}"
        return f"A3|t1={_f(t.tau1)}|t2={_f(t.tau2)}|t3={_f(t.tau3)}|N={t.N}|R={t.R}|S={S}"


def _enc(x):
    return None if x is None else ("inf" if x == float("inf") else "-inf" if x == float("-inf") else x)


def _dec(x):
    return None if x is None else float(x) if isinstance(x, str) else x


def spec_to_dict(spec: "PolicySpec") -> dict:
    t = spec.theta
    return {"kind": spec.kind, "tau": _enc(spec.tau), "k": spec.k,
            "theta": None if t is None else {"tau1": _enc(t.tau1), "tau2": _enc(t.tau2), "tau3": _enc(t.tau3),
                                              "N": t.N, "R": t.R, "S": t.S}}


def spec_from_dict(d: dict) -> "PolicySpec":
    t = d.get("theta")
    theta = None if t is None else Theta(_dec(t["tau1"]), _dec(t["tau2"]), _dec(t["tau3"]), t["N"], t["R"], t["S"])
    return PolicySpec(d["kind"], theta, _dec(d.get("tau")), d.get("k"))


def run_spec(spec: PolicySpec, session, tie_order: Sequence[str]) -> PolicyResult:
    k = spec.kind
    if k == "B1": return run_b1(session)
    if k == "B2": return run_b2(session)
    if k == "B3": return run_b3(session, spec.k)
    if k == "B4": return run_b4(spec.theta, session, tie_order)
    if k == "B5": return run_b5(spec.tau, session)
    if k == "A1": return run_a1(spec.theta, session, tie_order)
    if k == "A2": return run_a2(spec.theta, session, tie_order)
    if k == "A3": return run_policy(spec.theta, session, tie_order)
    raise ValueError(f"kind tidak dikenal: {k}")


def build_specs(cfg_raw: Dict[str, Any], tau1_values: Sequence[float]) -> List[PolicySpec]:
    """Enumerasi seluruh kebijakan yang dievaluasi pada dev. tau1_values dari routing.thresholds.tau1_candidates (dev)."""
    ps = cfg_raw["policy_space"]
    dedupe = ps["tau_disagreement"]["dedupe_equivalent"]
    umaj = ps["tau_disagreement"]["require_unique_majority"]
    specs: List[PolicySpec] = [PolicySpec("B1"), PolicySpec("B2"), PolicySpec("B3", k=cfg_raw["baselines"]["b3"]["k"])]
    specs += [PolicySpec("B5", tau=t) for t in tau1_values]
    for N in ps["N"]:
        from ..routing.entropy import achievable_entropies
        taus = effective_thresholds(N, umaj) if dedupe else achievable_entropies(N)
        for S in ps["S"]:
            for R in ps["R"]:
                specs.append(PolicySpec("B4", Theta(float("inf"), taus[0], taus[0], N, R, S)))
                for t1 in tau1_values:
                    specs.append(PolicySpec("A1", Theta(t1, taus[0], taus[0], N, R, S)))
                for t2 in taus:
                    for t3 in taus:
                        specs.append(PolicySpec("A2", Theta(float("inf"), t2, t3, N, R, S)))
                        for t1 in tau1_values:
                            specs.append(PolicySpec("A3", Theta(t1, t2, t3, N, R, S)))
    ids = [s.policy_id for s in specs]
    assert len(ids) == len(set(ids)), "policy_id harus unik"
    return specs
