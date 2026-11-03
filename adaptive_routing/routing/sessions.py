"""Sesi per query yang menyediakan hasil tiap tahap kepada router.

LiveSession  : memanggil LLM secara LAZY (hanya tahap yang diminta) dan mencatat CallRecord.
               Dipakai untuk eksekusi langsung (test) dan untuk pengumpulan trace lengkap (dev).
TraceSession : membaca CallRecord yang sudah tersimpan (replay). Tidak memanggil model.
Keduanya memenuhi antarmuka yang sama sehingga routing/policy.py adalah SATU-SATUNYA sumber keputusan.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ..agents.agent import answer_messages, verify_messages
from ..data.types import Query
from ..debate.protocol import revision_messages
from ..llm.base import CallContext
from ..llm.recording import RecordingLLM
from ..tracing.schema import CallRecord
from ..verification.p_true import first_token_alternatives, p_true_from_alternatives
from .answers import parse_answer
from .nodes import NodeSpec, agent_nodes, model_cfg


class TraceMissing(KeyError):
    """Trace tidak memuat pemanggilan yang dibutuhkan kebijakan (mis. N atau S tidak dikumpulkan)."""


@dataclass
class Stage1Result:
    answer: Optional[str]
    p_true: Optional[float]
    calls: List[CallRecord]


@dataclass
class RoundResult:
    answers: List[Optional[str]]
    node_ids: List[str]
    texts: List[str]
    calls: List[CallRecord]


class LiveSession:
    def __init__(self, q: Query, rec: RecordingLLM, cfg_raw: Dict[str, Any], S: str, N: int, runner):
        self.q, self.rec, self.cfg, self.S, self.N, self.runner = q, rec, cfg_raw, S, N, runner
        self.nodes: List[NodeSpec] = agent_nodes(cfg_raw, S, N)
        self._s1: Optional[Stage1Result] = None
        self._a: Optional[CallRecord] = None
        self._strong: Optional[Stage1Result] = None
        self._sc: Dict[int, RoundResult] = {}
        self._rounds: Dict[int, RoundResult] = {}

    # --- util
    def _ctx(self, stage, group, node, model_key, rnd=None, sidx=None) -> CallContext:
        return CallContext(self.q.query_id, self.q.dataset, self.q.split, stage, group, node, model_key, rnd, sidx)

    @staticmethod
    def _params(m: Dict[str, Any], temperature: Optional[float] = None) -> Dict[str, Any]:
        return {"temperature": m.get("temperature") if temperature is None else temperature,
                "max_tokens": m["max_tokens"], "extra_body": m.get("extra_body") or {},
                "reasoning_effort": m.get("reasoning_effort")}

    def _set_parse(self, rec: CallRecord) -> None:
        if rec.error:
            rec.parsed_answer, rec.parse_ok = None, False
        else:
            rec.parsed_answer = parse_answer(self.q.dataset, rec.text)
            rec.parse_ok = rec.parsed_answer is not None

    # --- tahap 1
    def _answer_call(self) -> CallRecord:
        if self._a is None:
            slm = self.cfg["backends"]["slm"]
            a_ctx = self._ctx("s1_answer", "s1a", "slm", slm["key"])
            (a,) = self.runner.run([lambda: self.rec.call(a_ctx, slm, answer_messages(self.q), self._params(slm))])
            self._set_parse(a)
            self._a = a
        return self._a

    def slm_answer(self) -> Stage1Result:
        """B1: hanya jawaban SLM, tanpa verifikasi."""
        a = self._answer_call()
        return Stage1Result(a.parsed_answer if a.parse_ok else None, None, [a])

    def strong(self) -> Stage1Result:
        """B2: satu pemanggilan model kuat."""
        if self._strong is None:
            m = self.cfg["backends"]["strong"]
            ctx = self._ctx("b2_strong", "b2", m["key"], m["key"])
            (c,) = self.runner.run([lambda: self.rec.call(ctx, m, answer_messages(self.q), self._params(m))])
            self._set_parse(c)
            self._strong = Stage1Result(c.parsed_answer if c.parse_ok else None, None, [c])
        return self._strong

    def sc(self, k: int) -> RoundResult:
        """B3: k sampel independen SLM (temperatur dari config) secara paralel."""
        if k not in self._sc:
            slm = self.cfg["backends"]["slm"]
            temp = self.cfg["baselines"]["b3"]["temperature"]
            thunks = []
            for i in range(k):
                ctx = self._ctx("b3_sample", "b3", f"slm#{i}", slm["key"], None, i)
                thunks.append(lambda ctx=ctx: self.rec.call(ctx, slm, answer_messages(self.q), self._params(slm, temp)))
            calls = self.runner.run(thunks)
            for c in calls:
                self._set_parse(c)
            self._sc[k] = RoundResult([c.parsed_answer if c.parse_ok else None for c in calls],
                                      [f"slm#{i}" for i in range(k)], [c.text for c in calls], calls)
        return self._sc[k]

    def stage1(self) -> Stage1Result:
        if self._s1 is not None:
            return self._s1
        slm = self.cfg["backends"]["slm"]
        a = self._answer_call()
        calls, p = [a], None
        if a.parse_ok:                                   # verifikasi dilewati bila jawaban invalid (tidak ada yang diverifikasi)
            v = self.cfg["verification"]
            lp = dict(slm["logprobs"], top_k=v["top_k"])
            vp = self._params(slm, 0.0)
            vp["max_tokens"], vp["logprobs"] = v["max_tokens"], lp
            v_ctx = self._ctx("s1_verify", "s1v", "slm", slm["key"])
            (vr,) = self.runner.run([lambda: self.rec.call(v_ctx, slm, verify_messages(self.q, a.parsed_answer), vp)])
            alts = None if vr.error else first_token_alternatives(vr.meta.get("raw_logprobs"))
            vr.logprobs_first_token = alts
            p, meta = p_true_from_alternatives(alts, v["normalize"])
            vr.p_true = p
            vr.meta.update(meta)
            calls.append(vr)
        self._s1 = Stage1Result(a.parsed_answer if a.parse_ok else None, p, calls)
        return self._s1

    # --- tahap 2 dan revisi
    def _run_round(self, rnd: int) -> RoundResult:
        stage, group = ("s2_agent", "s2") if rnd == 0 else ("s3_revision", f"s3_r{rnd}")
        thunks = []
        for i, n in enumerate(self.nodes):
            m = model_cfg(self.cfg, n.model_key)
            msgs = (answer_messages(self.q) if rnd == 0 else
                    revision_messages(self.q, i, self._rounds[rnd - 1].texts, [x.node_id for x in self.nodes]))
            ctx = self._ctx(stage, group, n.node_id, n.model_key, rnd or None, n.sample_idx)
            thunks.append(lambda m=m, msgs=msgs, ctx=ctx, n=n: self.rec.call(ctx, m, msgs, self._params(m, n.temperature)))
        calls = self.runner.run(thunks)
        for c in calls:
            self._set_parse(c)
        return RoundResult([c.parsed_answer if c.parse_ok else None for c in calls],
                           [n.node_id for n in self.nodes], [c.text for c in calls], calls)

    def stage2(self) -> RoundResult:
        if 0 not in self._rounds:
            self._rounds[0] = self._run_round(0)
        return self._rounds[0]

    def revision(self, r: int) -> RoundResult:
        if r < 1:
            raise ValueError("ronde revisi dimulai dari 1")
        for k in range(0, r):                            # ronde r membutuhkan ronde sebelumnya
            if k not in self._rounds:
                self._rounds[k] = self._run_round(k)
        if r not in self._rounds:
            self._rounds[r] = self._run_round(r)
        return self._rounds[r]


class TraceSession:
    def __init__(self, q: Query, calls: Sequence[CallRecord], S: str, N: int, cfg_raw: Dict[str, Any]):
        self.q, self.S, self.N = q, S, N
        self.nodes = agent_nodes(cfg_raw, S, N)
        self._strong_key = cfg_raw["backends"]["strong"]["key"]
        self._idx: Dict[tuple, CallRecord] = {}
        for c in calls:
            if c.query_id != q.query_id:
                continue
            k = (c.stage, c.node_id, c.round if c.round else 0)
            if k not in self._idx or (self._idx[k].cache_hit and not c.cache_hit):
                self._idx[k] = c

    def _get(self, stage: str, node: str, rnd: int = 0) -> CallRecord:
        try:
            return self._idx[(stage, node, rnd)]
        except KeyError:
            raise TraceMissing(f"{self.q.query_id}: tidak ada {stage}/{node}/r{rnd} pada trace") from None

    def slm_answer(self) -> Stage1Result:
        a = self._get("s1_answer", "slm")
        return Stage1Result(a.parsed_answer if a.parse_ok else None, None, [a])

    def strong(self) -> Stage1Result:
        c = self._get("b2_strong", self._strong_key)
        return Stage1Result(c.parsed_answer if c.parse_ok else None, None, [c])

    def sc(self, k: int) -> RoundResult:
        calls = [self._get("b3_sample", f"slm#{i}") for i in range(k)]
        return RoundResult([c.parsed_answer if c.parse_ok else None for c in calls],
                           [f"slm#{i}" for i in range(k)], [c.text for c in calls], calls)

    def stage1(self) -> Stage1Result:
        a = self._get("s1_answer", "slm")
        calls, p = [a], None
        if a.parse_ok:
            v = self._get("s1_verify", "slm")
            calls.append(v)
            p = v.p_true
        return Stage1Result(a.parsed_answer if a.parse_ok else None, p, calls)

    def _round(self, rnd: int) -> RoundResult:
        stage = "s2_agent" if rnd == 0 else "s3_revision"
        calls = [self._get(stage, n.node_id, rnd) for n in self.nodes]
        return RoundResult([c.parsed_answer if c.parse_ok else None for c in calls],
                           [n.node_id for n in self.nodes], [c.text for c in calls], calls)

    def stage2(self) -> RoundResult:
        return self._round(0)

    def revision(self, r: int) -> RoundResult:
        return self._round(r)
