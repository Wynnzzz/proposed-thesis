"""Buku harga. Biaya = (token_masuk*in + token_keluar*out) / 1e6 (Pers. II.1 tesis)."""
from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import yaml


class PricingError(ValueError):
    pass


@dataclass(frozen=True)
class Price:
    key: str
    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: Optional[float]
    verified_on: Optional[dt.date]
    source: str


class PriceBook:
    def __init__(self, raw: dict):
        self.max_age_days = int(raw.get("max_age_days", 30))
        self.output_includes_reasoning = bool(raw.get("output_includes_reasoning", True))
        self._p: Dict[str, Price] = {}
        for k, v in (raw.get("models") or {}).items():
            vo = v.get("verified_on")
            if isinstance(vo, str):
                vo = dt.date.fromisoformat(vo)
            self._p[k] = Price(k, float(v["input_per_mtok"]), float(v["output_per_mtok"]),
                               v.get("cached_input_per_mtok"), vo, v.get("source", ""))

    @classmethod
    def from_file(cls, path: str) -> "PriceBook":
        with open(path, encoding="utf-8") as f:
            return cls(yaml.safe_load(f))

    def get(self, key: str) -> Price:
        if key not in self._p:  # berbeda dari MasRouter: model tak dikenal TIDAK bernilai 0 diam-diam
            raise PricingError(f"Tarif tidak ditemukan untuk '{key}'")
        return self._p[key]

    def cost_usd(self, key: str, input_tokens: int, output_tokens: int) -> float:
        p = self.get(key)
        return (input_tokens * p.input_per_mtok + output_tokens * p.output_per_mtok) / 1e6

    def unverified(self, keys: Iterable[str], today: Optional[dt.date] = None):
        today = today or dt.date.today()
        bad = []
        for k in keys:
            p = self.get(k)
            if p.verified_on is None or (today - p.verified_on).days > self.max_age_days:
                bad.append(k)
        return bad

    def require_verified(self, keys: Iterable[str], today: Optional[dt.date] = None) -> None:
        bad = self.unverified(keys, today)
        if bad:
            raise PricingError("Tarif belum terverifikasi atau kedaluwarsa: " + ", ".join(sorted(bad)))

    def snapshot(self, keys: Iterable[str]) -> dict:
        return {k: {"input_per_mtok": self.get(k).input_per_mtok, "output_per_mtok": self.get(k).output_per_mtok,
                    "verified_on": str(self.get(k).verified_on) if self.get(k).verified_on else None,
                    "source": self.get(k).source} for k in keys}
