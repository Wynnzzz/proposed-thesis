"""Klien HTTP kompatibel OpenAI (Together, OpenAI) berbasis stdlib.

Belum diuji terhadap backend nyata (sandbox tanpa jaringan); diuji terhadap server palsu lokal.
Gaya logprobs: together_int -> {"logprobs": k}; openai_bool -> {"logprobs": true, "top_logprobs": k}.
Gaya 'auto' DITOLAK: harus ditetapkan dari probe_report.
"""
from __future__ import annotations
import datetime as dt, json, os, time, urllib.error, urllib.request
from typing import Any, Dict, List

from .base import CallContext, LLMError, LLMResponse

RETRY_HTTP = {408, 409, 425, 429, 500, 502, 503, 504}


class HTTPClient:
    def __init__(self, timeout_s: float = 120, max_retries: int = 3, backoff_s: float = 2.0,
                 env: Dict[str, str] | None = None, sleep=time.sleep):
        self.timeout_s, self.max_retries, self.backoff_s = timeout_s, max_retries, backoff_s
        self.env, self.sleep = os.environ if env is None else env, sleep

    @staticmethod
    def build_body(model_cfg: Dict[str, Any], messages: List[Dict[str, str]], params: Dict[str, Any]) -> dict:
        body: Dict[str, Any] = {"model": model_cfg["model_id"], "messages": messages}
        if params.get("temperature") is not None:
            body["temperature"] = params["temperature"]
        mt = params["max_tokens"]
        body["max_completion_tokens" if model_cfg["provider"] == "openai" else "max_tokens"] = mt
        if params.get("reasoning_effort"):
            body["reasoning_effort"] = params["reasoning_effort"]
        lp = params.get("logprobs")
        if lp:
            style = lp.get("style", "auto")
            if style == "together_int":
                body["logprobs"] = lp["top_k"]
            elif style == "openai_bool":
                body["logprobs"], body["top_logprobs"] = True, lp["top_k"]
            else:
                raise LLMError("gaya logprobs belum ditetapkan ('auto'); jalankan probe dan isi config")
        body.update(params.get("extra_body") or {})
        return body

    def chat(self, model_cfg, messages, params, ctx: CallContext) -> LLMResponse:
        key = self.env.get(model_cfg["api_key_env"], "")
        body = self.build_body(model_cfg, messages, params)
        url = model_cfg["base_url"].rstrip("/") + "/chat/completions"
        t_submit = dt.datetime.now(dt.timezone.utc).isoformat()
        t0 = time.perf_counter()
        last_err = "unknown"
        for attempt in range(1, self.max_retries + 2):
            req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json",
                                                  **({"Authorization": f"Bearer {key}"} if key else {})})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                    data = json.loads(r.read())
                t1 = time.perf_counter()
                return self._parse(data, t_submit, t0, t1, attempt)
            except urllib.error.HTTPError as e:
                last_err = f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"
                if e.code not in RETRY_HTTP:
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last_err = f"{type(e).__name__}: {e}"
            if attempt <= self.max_retries:
                self.sleep(self.backoff_s * (2 ** (attempt - 1)))
        raise LLMError(last_err)

    @staticmethod
    def _parse(data: dict, t_submit: str, t0: float, t1: float, attempts: int) -> LLMResponse:
        ch = data["choices"][0]
        u = data.get("usage") or {}
        det = u.get("completion_tokens_details") or {}
        pdet = u.get("prompt_tokens_details") or {}
        return LLMResponse(
            text=(ch.get("message") or {}).get("content") or "",
            finish_reason=ch.get("finish_reason"),
            input_tokens=u.get("prompt_tokens"), output_tokens=u.get("completion_tokens"),
            reasoning_tokens=det.get("reasoning_tokens"), cached_input_tokens=pdet.get("cached_tokens"),
            raw_logprobs=ch.get("logprobs"), latency_s=t1 - t0, t_submit_utc=t_submit,
            t_start_mono=t0, t_end_mono=t1, attempts=attempts)
