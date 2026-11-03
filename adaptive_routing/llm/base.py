"""Antarmuka LLM. Klien (HTTP nyata atau mock) hanya mengembalikan fakta; pencatatan ada di RecordingLLM."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


@dataclass(frozen=True)
class CallContext:
    query_id: str
    dataset: str
    split: str
    stage: str
    group_id: str
    node_id: str
    model_key: str
    round: Optional[int] = None
    sample_idx: Optional[int] = None


@dataclass
class LLMResponse:
    text: str
    finish_reason: Optional[str]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    reasoning_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    raw_logprobs: Any = None
    latency_s: float = 0.0
    t_submit_utc: str = ""
    t_start_mono: float = 0.0
    t_end_mono: float = 0.0
    attempts: int = 1


class LLMError(RuntimeError):
    """Kegagalan pemanggilan setelah seluruh retry (dicatat sebagai error, bukan disembunyikan)."""


class LLMClient(Protocol):
    def chat(self, model_cfg: Dict[str, Any], messages: List[Dict[str, str]],
             params: Dict[str, Any], ctx: CallContext) -> LLMResponse: ...
