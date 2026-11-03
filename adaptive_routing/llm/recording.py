"""RecordingLLM: membungkus klien, mengubah respons menjadi CallRecord (usage API, harga, biaya), menegakkan
budget guard dan cache. Token tidak pernah diperkirakan: usage hilang -> UsageMissingError (run berhenti)."""
from __future__ import annotations
import copy
from typing import Any, Callable, Dict, List, Optional

from ..tracing.schema import CallRecord, request_key, stable_hash
from .base import CallContext, LLMClient, LLMError
from .pricing import PriceBook


class BudgetExceeded(RuntimeError):
    pass


class UsageMissingError(RuntimeError):
    pass


class BudgetGuard:
    def __init__(self, cap_usd: float, abort_fraction: float = 0.9):
        self.cap, self.frac, self.spent = cap_usd, abort_fraction, 0.0

    def check(self) -> None:
        if self.spent >= self.cap * self.frac:
            raise BudgetExceeded(f"anggaran habis: terpakai ${self.spent:.4f} dari batas ${self.cap * self.frac:.4f}")

    def add(self, usd: float) -> None:
        self.spent += usd


class RecordingLLM:
    def __init__(self, client: LLMClient, pricebook: PriceBook, run_id: str,
                 sink: Optional[Callable[[CallRecord], None]] = None, budget: Optional[BudgetGuard] = None,
                 cache_enabled: bool = False, allow_estimated: bool = False):
        self.client, self.pb, self.run_id = client, pricebook, run_id
        self.records: List[CallRecord] = []
        self.sink = sink
        self.budget, self.cache_enabled, self.allow_estimated = budget, cache_enabled, allow_estimated
        self._cache: Dict[str, CallRecord] = {}

    def _emit(self, rec: CallRecord) -> CallRecord:
        self.records.append(rec)
        if self.sink:
            self.sink(rec)
        return rec

    def call(self, ctx: CallContext, model_cfg: Dict[str, Any], messages: List[Dict[str, str]],
             params: Dict[str, Any]) -> CallRecord:
        # Kunci cache = prompt+riwayat+parameter identik DAN identitas logis pemanggilan identik
        # (tahap, node, ronde, sampel). Tanpa identitas logis, ronde debat yang konvergen akan saling
        # memakai hasil ronde lain dan menggabungkan sampel independen (ditemukan oleh tes kesetaraan).
        key = request_key(model_cfg["model_id"], messages,
                          {**params, "_id": [ctx.stage, ctx.node_id, ctx.round, ctx.sample_idx]})
        if self.cache_enabled and key in self._cache:
            rec = copy.deepcopy(self._cache[key])
            for f in ("stage", "group_id", "node_id", "round", "sample_idx", "model_key"):
                setattr(rec, f, getattr(ctx, f))
            rec.cache_hit = True
            rec.call_id = stable_hash([self.run_id, ctx.query_id, ctx.stage, ctx.node_id, ctx.round, ctx.sample_idx, "hit"])
            return self._emit(rec)
        if self.budget:
            self.budget.check()
        rec = CallRecord(
            call_id=stable_hash([self.run_id, ctx.query_id, ctx.stage, ctx.node_id, ctx.round, ctx.sample_idx]),
            run_id=self.run_id, query_id=ctx.query_id, dataset=ctx.dataset, split=ctx.split, stage=ctx.stage,
            group_id=ctx.group_id, node_id=ctx.node_id, round=ctx.round, sample_idx=ctx.sample_idx,
            model_key=ctx.model_key, provider=model_cfg["provider"], model_id=model_cfg["model_id"],
            request_key=key, history_hash=stable_hash(messages[:-1]), params=params)
        try:
            resp = self.client.chat(model_cfg, messages, params, ctx)
        except LLMError as e:
            rec.error, rec.cost_usd = str(e), 0.0          # tidak ditagih; dihitung sebagai jawaban invalid
            return self._emit(rec)
        rec.t_submit_utc, rec.t_start_mono, rec.t_end_mono = resp.t_submit_utc, resp.t_start_mono, resp.t_end_mono
        rec.latency_s, rec.attempt = resp.latency_s, resp.attempts
        rec.input_tokens, rec.output_tokens = resp.input_tokens, resp.output_tokens
        rec.reasoning_tokens, rec.cached_input_tokens = resp.reasoning_tokens, resp.cached_input_tokens
        rec.text, rec.finish_reason = resp.text, resp.finish_reason
        rec.meta["raw_logprobs"] = resp.raw_logprobs
        try:
            rec.finalize_cost(self.pb, model_cfg["pricing_key"])
        except ValueError as e:
            raise UsageMissingError(str(e)) from e
        rec.validate(self.allow_estimated)
        if self.budget:
            self.budget.add(rec.cost_usd)
        if self.cache_enabled:
            self._cache[key] = copy.deepcopy(rec)
        return self._emit(rec)
