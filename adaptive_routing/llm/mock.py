"""Backend mock deterministik untuk tes (BUKAN hasil eksperimen). Keluaran diturunkan dari hash
(seed, query, node, tahap, ronde), sehingga tidak bergantung urutan pemanggilan."""
from __future__ import annotations
import random, re
from typing import Any, Dict, List

from ..tracing.schema import stable_hash
from .base import CallContext, LLMResponse

LETTERS = "ABCD"


class SimClock:
    def __init__(self):
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class MockLLM:
    def __init__(self, gold: Dict[str, str], skill: Dict[str, float], latency: Dict[str, float],
                 clock: SimClock | None = None, seed: int = 7, invalid_rate: float = 0.04,
                 logprob_format: str = "openai_content"):
        self.gold, self.skill, self.latency = gold, skill, latency
        self.clock, self.seed, self.invalid_rate, self.fmt = clock or SimClock(), seed, invalid_rate, logprob_format
        self.n_calls = 0

    def _rng(self, ctx: CallContext) -> random.Random:
        return random.Random(stable_hash([self.seed, ctx.query_id, ctx.node_id, ctx.stage, ctx.round, ctx.sample_idx]))

    def _answer_set(self, ctx: CallContext):
        g = self.gold[ctx.query_id]
        return (list(LETTERS), g) if ctx.dataset == "mmlu" else ([g, "alpha", "beta", "gamma"], g)

    def chat(self, model_cfg: Dict[str, Any], messages: List[Dict[str, str]], params: Dict[str, Any],
             ctx: CallContext) -> LLMResponse:
        self.n_calls += 1
        rng = self._rng(ctx)
        lat = self.latency.get(ctx.model_key, 1.0) * (0.5 + rng.random())
        in_tok = sum(len(m["content"]) for m in messages) // 4 + 5
        raw_lp, out_tok = None, 0
        if ctx.stage == "s1_verify":
            proposed = re.search(r"Proposed answer: (.+)", messages[-1]["content"]).group(1).strip()
            opts, gold = self._answer_set(ctx)
            ok = proposed.lower() == gold.lower()
            p = rng.uniform(0.55, 0.95) if ok else rng.uniform(0.05, 0.8)
            text, out_tok = ("Yes" if p >= 0.5 else "No"), 1
            import math
            alts = {"Yes": math.log(p), "No": math.log(1 - p)}
            if self.fmt == "none":
                raw_lp = None                                  # backend tanpa logprobs
            elif self.fmt == "openai_content":
                raw_lp = {"content": [{"token": text, "logprob": alts[text],
                                       "top_logprobs": [{"token": k, "logprob": v} for k, v in alts.items()]}]}
            else:
                raw_lp = {"tokens": [text], "token_logprobs": [alts[text]], "top_logprobs": [dict(alts)]}
        else:
            opts, gold = self._answer_set(ctx)
            skill = self.skill.get(ctx.model_key, 0.5) + (0.05 if ctx.stage == "s3_revision" else 0.0)
            ans = gold if rng.random() < skill else rng.choice([o for o in opts if o != gold])
            if rng.random() < self.invalid_rate:
                text = "I think it could be several things, hard to say."
            else:
                text = f"Reasoning ...\nFinal answer: {ans}"
            out_tok = len(text) // 4 + 20 + int(rng.random() * 30)
        self.clock_start = self.clock.now()
        return LLMResponse(text=text, finish_reason="stop", input_tokens=in_tok, output_tokens=out_tok,
                           raw_logprobs=raw_lp, latency_s=lat, t_submit_utc="mock",
                           t_start_mono=self.clock.now(), t_end_mono=self.clock.now() + lat)
