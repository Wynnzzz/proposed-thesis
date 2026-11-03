#!/usr/bin/env python3
"""Probe kompatibilitas backend untuk eksperimen tesis (Tahap 1).

Tujuan: menguji APA YANG SEBENARNYA dikembalikan tiap backend, tanpa asumsi:
  - apakah logprobs didukung, dalam format apa, dan apakah token Yes/No muncul
  - apakah usage (token masuk/keluar) dikembalikan oleh API
  - apakah model reasoning menghabiskan token pada pemikiran tersembunyi
  - latensi satu pemanggilan

Skrip ini TIDAK mengganti metode apa pun. Jika logprobs tidak didukung, hasilnya
dilaporkan apa adanya di probe_report.json.

Env:  TOGETHER_API_KEY, OPENAI_API_KEY (dari environment atau file .env di folder kerja),
      TOGETHER_BASE_URL (opsional, untuk uji lokal)
Pakai: python scripts/probe_backends.py [--only NAMA ...] [--include-alt] [--out probe_report.json]

Keputusan D1: SLM lewat Together AI (tanpa vLLM). Target 'slm_alt_*' hanya JALAN bila diminta
(--include-alt atau --only); skrip tidak pernah mengganti model SLM secara otomatis. Jika
slm_together gagal logprobs, laporkan hasilnya dan tinjau alternatif sebelum mengubah metodologi.
Hanya stdlib (tanpa dependensi).
"""
import argparse, json, math, os, time, urllib.request, urllib.error


def load_dotenv(path=".env"):
    """Baca KEY=VALUE dari .env (tanpa dependensi). Variabel yang sudah ada di environment TIDAK ditimpa."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_dotenv()

PROMPT = ("Question: What is the capital of France?\nA. Berlin\nB. Paris\nC. Rome\nD. Madrid\n"
          "Proposed answer: B\nIs the proposed answer correct? Reply with exactly one word: Yes or No.")

# Catatan: id model berasal dari hasil pencarian (sumber sekunder); belum diverifikasi ke /v1/models.
TB = os.getenv("TOGETHER_BASE_URL", "https://api.together.ai/v1")
TARGETS = [
    dict(name="slm_together", base=TB, key_env="TOGETHER_API_KEY",
         model="meta-llama/Llama-3.2-3B-Instruct-Turbo", logprobs=True, style="together"),
    dict(name="agent1_qwen", base=TB, key_env="TOGETHER_API_KEY",
         model="Qwen/Qwen3.5-9B", logprobs=False, style="together",
         extra={"reasoning": {"enabled": False}}),
    dict(name="agent2_llama3_8b_lite", base=TB, key_env="TOGETHER_API_KEY",
         model="meta-llama/Meta-Llama-3-8B-Instruct-Lite", logprobs=False, style="together"),
    dict(name="agent3_glm", base=TB, key_env="TOGETHER_API_KEY",
         model="zai-org/GLM-5.3-Flash", logprobs=False, style="together"),
    dict(name="strong_gpt", base="https://api.openai.com/v1", key_env="OPENAI_API_KEY",
         model="gpt-6.1-terra", logprobs=False, style="openai_new"),
    # Uji GRATIS lokal (hanya untuk debugging pipeline; BUKAN pengganti SLM Together untuk eksperimen):
    dict(name="slm_alt_ollama", base=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"), key_env="OLLAMA_API_KEY",
         model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"), logprobs=True, style="openai", optional=True, no_key=True),
    # Kandidat alternatif SLM (hanya diuji bila diminta; tidak otomatis menggantikan SLM):
    dict(name="slm_alt_qwen35_9b", base=TB, key_env="TOGETHER_API_KEY", model="Qwen/Qwen3.5-9B",
         logprobs=True, style="together", extra={"reasoning": {"enabled": False}}, optional=True),
]


def post(base, key, body, timeout=120):
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {key}"} if key else {})})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode(errors="replace")[:500]}, time.perf_counter() - t0
    except Exception as e:  # jaringan / koneksi
        return None, {"error": f"{type(e).__name__}: {e}"}, time.perf_counter() - t0


def parse_logprobs(choice):
    """Kembalikan (format, daftar top-alternatif token pertama {token: logprob}) atau (None, None)."""
    lp = choice.get("logprobs")
    if not lp:
        return None, None
    content = lp.get("content")
    if isinstance(content, list) and content:            # gaya OpenAI
        first = content[0]
        alts = {a["token"]: a["logprob"] for a in first.get("top_logprobs", [])}
        alts.setdefault(first["token"], first["logprob"])
        return "openai_content", alts
    if "tokens" in lp and "token_logprobs" in lp:        # gaya Together lama
        alts = dict(lp["top_logprobs"][0]) if lp.get("top_logprobs") and lp["top_logprobs"][0] else {}
        alts.setdefault(lp["tokens"][0], lp["token_logprobs"][0])
        return "together_tokens", alts
    return "unknown", None


def p_yes(alts):
    """P(Yes) dinormalisasi atas varian Yes/No yang tampak; None jika tidak ada."""
    norm = lambda t: t.strip().lower()
    y = sum(math.exp(v) for k, v in alts.items() if norm(k) == "yes")
    n = sum(math.exp(v) for k, v in alts.items() if norm(k) == "no")
    return (y / (y + n)) if (y + n) > 0 else None


def probe(t):
    key = os.getenv(t["key_env"], "")
    out = {"name": t["name"], "model": t["model"], "base": t["base"]}
    if not key and not t.get("no_key"):
        out["status"] = "SKIPPED: env %s kosong" % t["key_env"]
        return out
    body = {"model": t["model"], "messages": [{"role": "user", "content": PROMPT}], "temperature": 0}
    body["max_completion_tokens" if t["style"] == "openai_new" else "max_tokens"] = 16
    body.update(t.get("extra", {}))
    attempts = []
    variants = [None]
    if t["logprobs"]:  # coba dua gaya parameter; catat mana yang diterima
        variants = [{"logprobs": 5}, {"logprobs": True, "top_logprobs": 5}] if t["style"] == "together" \
            else [{"logprobs": True, "top_logprobs": 5}, {"logprobs": 5}]
    chosen = None
    for v in variants:
        b = {**body, **(v or {})}
        status, resp, dt = post(t["base"], key, b)
        att = {"logprobs_param": v, "http": status, "latency_s": round(dt, 3)}
        if status == 200:
            ch = resp["choices"][0]
            att["content"] = ch["message"].get("content")
            att["finish_reason"] = ch.get("finish_reason")
            att["usage"] = resp.get("usage")
            fmt, alts = parse_logprobs(ch)
            att["logprobs_format"] = fmt
            if alts:
                att["top_alternatives"] = dict(sorted(alts.items(), key=lambda kv: -kv[1])[:5])
                att["p_yes_normalized"] = p_yes(alts)
            if t["logprobs"] and not fmt:
                att["note"] = "HTTP 200 tetapi logprobs tidak ada di respons"
            if chosen is None and (fmt or not t["logprobs"]):
                chosen = att
        else:
            att["error"] = resp.get("error")
        attempts.append(att)
        if chosen:
            break
    out["attempts"] = attempts
    if t["logprobs"]:
        out["logprobs_ok"] = bool(chosen and chosen.get("logprobs_format") and chosen.get("p_yes_normalized") is not None)
    else:
        out["call_ok"] = bool(chosen)
    u = (chosen or {}).get("usage") or {}
    out["usage_returned"] = bool(u.get("prompt_tokens") is not None and u.get("completion_tokens") is not None)
    out["reasoning_tokens"] = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
    out["empty_content"] = bool(chosen and not (chosen.get("content") or "").strip())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--include-alt", action="store_true")
    ap.add_argument("--out", default="probe_report.json")
    a = ap.parse_args()
    report = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "results": []}
    for t in TARGETS:
        if a.only and t["name"] not in a.only:
            continue
        if t.get("optional") and not (a.include_alt or a.only):
            continue
        r = probe(t)
        report["results"].append(r)
        flag = r.get("logprobs_ok", r.get("call_ok", r.get("status")))
        print(f"{t['name']:<24} ok={flag} usage={r.get('usage_returned')} reasoning_tokens={r.get('reasoning_tokens')} empty={r.get('empty_content')}")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("Laporan lengkap:", a.out)
