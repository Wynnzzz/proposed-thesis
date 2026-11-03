#!/usr/bin/env python3
"""Unduh dan proses MMLU dan HotpotQA (distractor) ke data/ . DIJALANKAN PENGGUNA (butuh internet + paket `datasets`).

BELUM diuji terhadap Hugging Face nyata (sandbox tanpa jaringan). Logika konversi/sampling diuji pada data sintetis
(tests/test_data.py). Bila skema atau jumlah baris tidak sesuai harapan, skrip berhenti (skema) atau mencatat peringatan
di manifest (jumlah).

Pakai: python scripts/prepare_data.py [--config configs/default.yaml] [--out data] [--only mmlu hotpotqa]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_routing.config import Config
from adaptive_routing.data.io import write_queries
from adaptive_routing.data.prepare import build_hotpot, build_mmlu, make_manifest

EXPECTED = {"mmlu": {"validation": 1531, "test": 14042}, "hotpotqa": {"validation": 7405}}   # dari ingatan, BELUM diverifikasi


def load_split(hf_id, config, split):
    try:
        import datasets
    except ImportError:
        sys.exit("Paket 'datasets' belum terpasang: pip install datasets")
    ds = datasets.load_dataset(hf_id, config, split=split)
    prov = {"datasets_version": datasets.__version__, "hf_split": split, "fingerprint": getattr(ds, "_fingerprint", None),
            "dataset_version": str(getattr(ds.info, "version", None)), "num_rows": len(ds)}
    return [dict(r) for r in ds], prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="data")
    ap.add_argument("--only", nargs="*", default=["mmlu", "hotpotqa"])
    a = ap.parse_args()
    cfg = Config.from_file(a.config)
    d, sizes, seed = cfg["datasets"], cfg["datasets"]["sizes"], cfg["datasets"]["sample_seed"]
    for name in a.only:
        out = os.path.join(a.out, name); os.makedirs(out, exist_ok=True)
        if name == "mmlu":
            val, pv = load_split(d["mmlu"]["hf_id"], d["mmlu"]["hf_config"], d["mmlu"]["dev_split"])
            tst, pt = load_split(d["mmlu"]["hf_id"], d["mmlu"]["hf_config"], d["mmlu"]["test_split"])
            dev, test, info = build_mmlu(val, tst, sizes, seed)
            counts = {"validation": len(val), "test": len(tst)}
            prov = {"validation": pv, "test": pt}
        else:
            val, pv = load_split(d["hotpotqa"]["hf_id"], d["hotpotqa"]["hf_config"], d["hotpotqa"]["pool_split"])
            dev, test, info = build_hotpot(val, sizes, seed)
            counts, prov = {"validation": len(val)}, {"validation": pv}
        write_queries(os.path.join(out, "dev.jsonl"), dev)
        write_queries(os.path.join(out, "test.jsonl"), test)
        write_queries(os.path.join(out, "pilot.jsonl"), dev[:sizes["pilot"]])
        man = make_manifest(name, d[name], sizes, seed, dev, test, counts, prov, info, EXPECTED[name])
        with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(man, f, indent=2, ensure_ascii=False)
        print(f"{name}: dev={len(dev)} test={len(test)} pilot={sizes['pilot']} peringatan_jumlah={man['count_warnings'] or 'tidak ada'}")


if __name__ == "__main__":
    main()
