"""Pemuat dan validator konfigurasi terpusat.

validate()        -> pemeriksaan struktural/metodologis (aman untuk mode mock).
validate_live()   -> tambahan untuk run yang memanggil API nyata:
                     probe dikonfirmasi, harga terverifikasi, kunci API ada.
Semua pelanggaran dikumpulkan lalu dilaporkan sekaligus (ConfigError).
"""
from __future__ import annotations
import copy, hashlib, json, os
from typing import Any, Dict, List
import yaml

from .llm.pricing import PriceBook
from .routing.entropy import achievable_entropies

RUN_MODES = {"mock", "trace_dev", "replay_dev", "direct_test"}
LIVE_MODES = {"trace_dev", "direct_test"}
PROVIDERS = {"together", "openai", "mock"}


class ConfigError(ValueError):
    pass


class Config:
    def __init__(self, raw: Dict[str, Any]):
        self.raw = raw

    @classmethod
    def from_file(cls, path: str, overrides: Dict[str, Any] | None = None) -> "Config":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        for dotted, v in (overrides or {}).items():
            cur = raw
            parts = dotted.split(".")
            for p in parts[:-1]:
                cur = cur[p]
            cur[parts[-1]] = v
        return cls(raw)

    def __getitem__(self, k):
        return self.raw[k]

    @property
    def mode(self) -> str:
        return self.raw["experiment"]["run_mode"]

    def hash(self) -> str:
        blob = json.dumps(self.raw, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def all_backends(self) -> List[dict]:
        b = self.raw["backends"]
        return [b["slm"], *b["agents"], b["strong"]]

    def pricing_keys(self) -> List[str]:
        return [m["pricing_key"] for m in self.all_backends()]

    # ------------------------------------------------------------------ validate
    def validate(self, pricebook: PriceBook | None = None) -> None:
        e: List[str] = []
        r = self.raw
        if self.mode not in RUN_MODES:
            e.append(f"experiment.run_mode harus salah satu dari {sorted(RUN_MODES)}")

        # --- backend [D1]
        b = r["backends"]
        slm = b["slm"]
        if slm["provider"] != "together":
            e.append("[D1] SLM harus melalui provider 'together' (tanpa vLLM lokal)")
        if not slm.get("logprobs", {}).get("required", False):
            e.append("[D1] SLM wajib logprobs.required=true untuk P(True)")
        keys = [m["key"] for m in self.all_backends()]
        if len(keys) != len(set(keys)):
            e.append("backends: key model harus unik")
        for m in self.all_backends():
            if m["provider"] not in PROVIDERS:
                e.append(f"provider tidak dikenal: {m['provider']}")
            if "pricing_key" not in m:
                e.append(f"{m['key']}: pricing_key wajib")
        fams = [a["model_id"].split("/")[0] for a in b["agents"]]
        if len(set(fams)) < 2:
            e.append("agen heterogen membutuhkan lebih dari satu keluarga/organisasi model")

        # --- ruang kebijakan
        ps = r["policy_space"]
        for n in ps["N"]:
            if n not in (3, 5):
                e.append(f"policy_space.N hanya 3 atau 5, ditemukan {n}")
            if "heterogeneous" in ps["S"] and n > len(b["agents"]):
                e.append(f"N={n} melebihi jumlah agen heterogen terdefinisi ({len(b['agents'])})")
        if any(x not in (1, 2, 3) for x in ps["R"]):
            e.append("policy_space.R harus subset {1,2,3}")
        if r["protocol"]["trace"]["full_trace_rounds"] < max(ps["R"]):
            e.append("trace.full_trace_rounds harus >= max(R) agar replay dapat memakai prefiks debat")
        if ps["tau1"]["compare"] != "ge" or ps["tau_disagreement"]["compare"] != "le":
            e.append("aturan perbandingan baku: P(True) >= tau1; entropi <= tau (Algoritma III.1)")

        # --- entropi [D4]
        en = r["entropy"]
        if en["log_base"] != "e":
            e.append("[D4] log_base harus 'e' (natural)")
        for n, levels in en["levels"].items():
            ach = achievable_entropies(int(n))
            if len(levels) != len(ach) or any(abs(a - c) > en["tolerance"] for a, c in zip(sorted(levels), ach)):
                e.append(f"[D4] entropy.levels[{n}] tidak sesuai nilai yang mungkin {[round(x, 4) for x in ach]}")
        for n in ps["N"]:
            if str(n) not in {str(k) for k in en["levels"]}:
                e.append(f"entropy.levels belum mendefinisikan N={n}")

        # --- invalid
        ah = r["answer_handling"]
        if ah["invalid_as"] != "singleton_distinct":
            e.append("answer_handling.invalid_as hanya 'singleton_distinct' yang diimplementasikan")

        # --- data [D3]
        d = r["datasets"]
        s = d["sizes"]
        if not (s["pilot"] <= s["dev"]):
            e.append("sizes.pilot harus <= sizes.dev (pilot adalah subset dev)")
        if not d["pilot_is_subset_of_dev"]:
            e.append("[D3] pilot harus subset dev agar test tidak disentuh")
        if d["mmlu"]["dev_split"] != "validation" or d["mmlu"]["test_split"] != "test":
            e.append("[D3] MMLU: dev=validation resmi, test=test resmi")
        if d["hotpotqa"]["pool_split"] != "validation":
            e.append("[D3] HotpotQA: dev dan test diambil disjoint dari validation")

        # --- protokol [D2]
        p = r["protocol"]
        if p["trace"]["collect_splits"] != ["dev"]:
            e.append("[D2] trace lengkap hanya boleh dikumpulkan pada split dev")
        if p["replay"]["splits"] != ["dev"]:
            e.append("[D2] replay hanya pada dev")
        if p["replay"]["latency_kind"] != "composed_estimated":
            e.append("[D2] latensi replay wajib berlabel composed_estimated")
        if p["selection"]["split"] != "dev" or not p["selection"]["declared_before_replay"]:
            e.append("[D2] pemilihan konfigurasi hanya pada dev dan kriteria dinyatakan sebelum replay")
        if p["direct"]["splits"] != ["test"] or p["direct"]["latency_kind"] != "measured_e2e":
            e.append("[D2] eksekusi langsung hanya pada test dan berlabel measured_e2e")
        if p["direct"]["cache_enabled"]:
            e.append("[D2] eksekusi langsung tidak boleh memakai cache (latensi harus nyata)")
        if p["stats"]["binary_paired_test"] != "mcnemar":
            e.append("[D3] uji biner berpasangan: mcnemar")

        # --- verifikasi
        v = r["verification"]
        if v["normalize"] not in ("yes_over_yes_plus_no", "raw_yes"):
            e.append("verification.normalize harus yes_over_yes_plus_no atau raw_yes")
        if v["top_k"] < 2:
            e.append("verification.top_k minimal 2 agar Yes dan No dapat muncul")
        if r["llm"]["max_retries"] < 0:
            e.append("llm.max_retries tidak boleh negatif")

        # --- seleksi
        sel = r["protocol"]["selection"]
        for k in sel["families"]:
            if k not in ("A3", "A1", "A2", "B4", "B5"):
                e.append(f"selection.families tidak dikenal: {k}")
        rb = sel["reference_b4"]
        if rb["N"] not in ps["N"] or rb["S"] not in ps["S"] or rb["R"] not in ps["R"]:
            e.append("selection.reference_b4 harus berada dalam policy_space (N, S, R)")
        for sc in sel["scenarios"]:
            if sc.get("max_cost_ratio_vs_B2") is None and sc.get("max_latency_ratio_vs_B4") is None:
                e.append(f"skenario {sc.get('name')}: minimal satu batas wajib")

        # --- baseline
        bl = r["baselines"]
        if bl["b3"]["k"] < 3:
            e.append("baselines.b3.k minimal 3")
        if bl["oracle"]["latency_kind"] != "composed_estimated":
            e.append("B7 (oracle) wajib berlabel composed_estimated")

        # --- anggaran
        if not (0 < r["budget"]["pilot_cap_usd"]):
            e.append("budget.pilot_cap_usd harus > 0")

        # --- harga: setiap pricing_key harus ada (tidak diam-diam 0)
        if pricebook is not None:
            for k in self.pricing_keys():
                try:
                    pricebook.get(k)
                except ValueError as ex:
                    e.append(str(ex))
        if e:
            raise ConfigError("Konfigurasi tidak valid:\n- " + "\n- ".join(e))

    def validate_live(self, pricebook: PriceBook, env: Dict[str, str] | None = None) -> None:
        self.validate(pricebook)
        env = os.environ if env is None else env
        e: List[str] = []
        if self.mode not in LIVE_MODES:
            e.append(f"run_mode '{self.mode}' bukan mode live {sorted(LIVE_MODES)}")
        for m in self.all_backends():
            if not m.get("probe_confirmed", False):
                e.append(f"{m['key']}: probe_confirmed=false (jalankan scripts/probe_backends.py dan tinjau hasilnya)")
            if not env.get(m["api_key_env"]):
                e.append(f"{m['key']}: env {m['api_key_env']} kosong")
        if self.raw["backends"]["slm"]["logprobs"].get("style", "auto") == "auto":
            e.append("slm.logprobs.style masih 'auto': tetapkan together_int atau openai_bool dari probe_report")
        bad = pricebook.unverified(self.pricing_keys())
        if bad:
            e.append("tarif belum terverifikasi/kedaluwarsa: " + ", ".join(sorted(set(bad))))
        if e:
            raise ConfigError("Run live ditolak:\n- " + "\n- ".join(e))
