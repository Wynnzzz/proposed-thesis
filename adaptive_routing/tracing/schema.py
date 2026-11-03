"""Skema trace [D2]: catatan per pemanggilan, per query, per hasil kebijakan, dan manifest.

Prinsip: trace menyimpan FAKTA (teks respons, usage dari API, waktu). Biaya dan latensi
kebijakan DIREKONSTRUKSI dari fakta ini; tidak ada angka yang diperkirakan dari panjang teks
(berbeda dengan MasRouter yang memakai tiktoken).
"""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional

STAGES = {
    "s1_answer", "s1_verify",          # Tahap 1: jawaban SLM, verifikasi P(True)
    "s2_agent",                        # Tahap 2: agen paralel independen
    "s3_revision",                     # Tahap 3: revisi debat (round >= 1)
    "b2_strong", "b3_sample",          # baseline B2, B3
    "b4_agent", "b4_revision",         # baseline B4 (MAD tetap), bila terpisah dari trace utama
}
TOKEN_SOURCES = {"api_usage", "estimated"}   # 'estimated' hanya boleh di mode mock
LATENCY_KINDS = {"composed_estimated", "measured_e2e"}


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]


def request_key(model_id: str, messages: List[dict], params: dict) -> str:
    """Kunci cache: model, prompt (termasuk riwayat dialog), dan parameter harus identik [Subbab III.3.4]."""
    return stable_hash({"m": model_id, "msgs": messages, "p": params})


@dataclass
class CallRecord:
    call_id: str
    run_id: str
    query_id: str
    dataset: str                    # mmlu | hotpotqa
    split: str                      # dev | test
    stage: str
    group_id: str                   # kelompok paralel: 's1', 's2', 's3_r1', ...  (replay: latensi = max dalam kelompok)
    node_id: str                    # slm | agent_qwen | ... | strong_gpt
    round: Optional[int] = None
    sample_idx: Optional[int] = None
    model_key: str = ""
    provider: str = ""
    model_id: str = ""
    request_key: str = ""
    history_hash: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    t_submit_utc: str = ""          # waktu dinding, ISO-8601 UTC
    t_start_mono: float = 0.0       # perf_counter, hanya bermakna dalam satu proses
    t_end_mono: float = 0.0
    latency_s: float = 0.0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None      # token keluaran yang DITAGIH (termasuk reasoning bila API melaporkan demikian)
    reasoning_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    token_source: str = "api_usage"
    price_in_per_mtok: Optional[float] = None    # snapshot harga saat pemanggilan
    price_out_per_mtok: Optional[float] = None
    cost_usd: Optional[float] = None
    pricing_verified: bool = False
    text: str = ""
    finish_reason: Optional[str] = None
    logprobs_first_token: Optional[Dict[str, float]] = None   # {token: logprob} alternatif token pertama (verifikasi)
    parsed_answer: Optional[str] = None
    parse_ok: Optional[bool] = None
    p_true: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)   # mis. p_yes_raw, p_no_raw, p_true_partial
    error: Optional[str] = None
    attempt: int = 1
    cache_hit: bool = False

    def finalize_cost(self, pricebook, pricing_key: str) -> None:
        if self.input_tokens is None or self.output_tokens is None:
            raise ValueError(f"{self.call_id}: token usage tidak tersedia; biaya tidak boleh diperkirakan")
        p = pricebook.get(pricing_key)
        self.price_in_per_mtok, self.price_out_per_mtok = p.input_per_mtok, p.output_per_mtok
        self.cost_usd = pricebook.cost_usd(pricing_key, self.input_tokens, self.output_tokens)
        self.pricing_verified = p.verified_on is not None

    def validate(self, allow_estimated: bool = False) -> None:
        errs = []
        if self.stage not in STAGES:
            errs.append(f"stage tidak dikenal: {self.stage}")
        if self.token_source not in TOKEN_SOURCES:
            errs.append(f"token_source tidak dikenal: {self.token_source}")
        if self.token_source == "estimated" and not allow_estimated:
            errs.append("token_source='estimated' ditolak di luar mode mock")
        if self.error is None:
            if self.input_tokens is None or self.output_tokens is None:
                errs.append("pemanggilan sukses wajib memiliki input_tokens dan output_tokens")
            if self.latency_s < 0 or self.t_end_mono < self.t_start_mono:
                errs.append("waktu tidak konsisten")
            if self.cost_usd is None:
                errs.append("cost_usd belum dihitung (finalize_cost)")
        if self.split not in ("dev", "test"):
            errs.append("split harus dev atau test")
        if errs:
            raise ValueError(f"CallRecord {self.call_id}: " + "; ".join(errs))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CallRecord":
        return cls(**d)


@dataclass
class QueryRecord:
    query_id: str
    dataset: str
    split: str
    gold: Any                       # MMLU: huruf; HotpotQA: jawaban + alias bila ada
    source_index: int               # indeks pada split resmi
    meta: Dict[str, Any] = field(default_factory=dict)   # mis. subject


@dataclass
class PolicyOutcome:
    """Hasil satu kebijakan pada satu query. latency_kind membedakan replay dan eksekusi langsung."""
    query_id: str
    policy_id: str
    final_answer: Optional[str]
    em: Optional[float]
    f1: Optional[float]
    correct: Optional[bool]
    stages_run: List[str]
    n_calls: int
    cost_usd: float
    latency_s: float
    latency_kind: str

    def __post_init__(self):
        if self.latency_kind not in LATENCY_KINDS:
            raise ValueError(f"latency_kind harus salah satu dari {sorted(LATENCY_KINDS)}")


@dataclass
class TraceManifest:
    run_id: str
    mode: str
    config_hash: str
    git_commit: Optional[str]
    created_utc: str
    dataset: str
    split: str
    n_queries: int
    query_ids: List[str]
    models: Dict[str, str]          # key -> model_id (versi tercatat)
    pricing_snapshot: Dict[str, Any]
    sampling_seed: int
    notes: str = ""


def append_jsonl(path: str, records: Iterable[Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            d = r.to_dict() if hasattr(r, "to_dict") else asdict(r)
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
