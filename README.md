# Adaptive Pareto Routing untuk MAS berbasis LLM (kode tesis)

Lihat `docs/DESIGN.md` (desain), `PROJECT_STATUS.md` (progres, keputusan, kendala).

```
python -m unittest discover -s tests -t .          # tes unit (tanpa jaringan)
python scripts/probe_backends.py --out probe_report.json   # uji logprobs/usage (butuh API key)
```
Run live ditolak oleh `Config.validate_live()` sampai probe dikonfirmasi dan harga terverifikasi.

```
python scripts/prepare_data.py            # unduh + proses MMLU/HotpotQA ke data/ (butuh internet + paket datasets)
```
```
python scripts/analyze_dev.py --trace results/traces/mmlu/dev/RUN --trace results/traces/hotpotqa/dev/RUN   # analisis dev
```
