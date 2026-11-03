#!/usr/bin/env python3
"""Analisis dev dari trace lengkap (replay semua kebijakan, Pareto, hypervolume, seleksi, pembekuan).
Pakai: python scripts/analyze_dev.py --trace results/traces/mmlu/dev/RUN --trace results/traces/hotpotqa/dev/RUN [--out results]
Hanya membaca trace DEV; menolak trace split lain."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_routing.config import Config
from adaptive_routing.runners.analyze_dev import analyze_dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--trace", action="append", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    rep = analyze_dev(a.trace, Config.from_file(a.config), a.out, make_plots=not a.no_plots)
    for ds, r in rep["datasets"].items():
        print(f"[{ds}] n_dev={r['n_dev']} kebijakan={r['n_policies']} frontier={r['frontier_size']}")
        for sc, byk in r["selected"].items():
            print(f"  {sc}: " + ", ".join(f"{k}={v}" for k, v in byk.items()))
        b = r["budget_direct_test_extrapolated"]
        print(f"  ekstrapolasi biaya test langsung ~ ${b['expected_total']:.3f} (batas atas ${b['upper_total']:.3f}); dari pilot kecil, bukan jaminan")
    print("selected_policies.json sha256:", rep["selected_policies_sha256"])
    if not rep["config_unchanged_since_trace"]:
        print("PERINGATAN: config berubah sejak trace dikumpulkan; direct runner akan menolaknya tanpa override eksplisit")


if __name__ == "__main__":
    main()
