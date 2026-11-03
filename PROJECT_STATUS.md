# PROJECT_STATUS.md

Proyek: Optimisasi Routing Adaptif Berbasis Pareto (biaya, latensi, akurasi) pada MAS berbasis LLM
Sumber utama: `USULAN_TESIS_UTS_28426003.pdf`. Referensi: MasRouter (ZIP, diperiksa langsung).
Checkpoint terakhir: 2026-11-03 (Tahap 5)

## Ringkasan Tahap

| Tahap | Nama | Status |
|---|---|---|
| 1 | Repository Analysis + validasi model/backend | SELESAI (uji API nyata belum dilakukan; lihat bagian "Belum teruji") |
| 2 | Project Design | SELESAI (`docs/DESIGN.md`); validasi eksplisit P1-P11 belum diterima |
| 3 | Core Implementation | **SELESAI pada backend MOCK dan server HTTP palsu**; jalur live belum teruji (lihat bagian bawah) |
| 4 | Benchmark and Baselines | **SELESAI pada data sintetis dan backend mock**; unduhan dataset nyata belum teruji |
| 5 | Optimization and Metrics | **SELESAI pada trace mock**; belum pernah dijalankan pada trace API nyata |
| 6-7 | Execution, Validation | Belum |

## Keputusan pengguna (2026-11-01)

- **D1** SLM = Llama-3.2-3B-Instruct-Turbo via Together AI; tanpa vLLM/GPU lokal. Logprobs aktual wajib; bila tidak ada, laporkan alternatif sebelum mengubah metodologi.
- **D2** Satu trace lengkap per query pada dev; replay offline; latensi replay berlabel `composed_estimated`; kebijakan terpilih dijalankan langsung pada test (`measured_e2e`); test tidak dipakai untuk memilih konfigurasi.
- **D3** MMLU dev = `validation` resmi, test = `test` resmi, stratifikasi per subjek; HotpotQA `validation` dibagi dev/test disjoint; pilot 20, dev 100, test 300; CI + McNemar/paired bootstrap.
- **D4** Log natural; N=3: 0; 0,6365; 1,0986; ruang ambang diskret.

## Tahap 5: hasil

File baru: `metrics/{summary,stats,pareto,hypervolume,selection,budget,plots}.py`, `tracing/store.py`, `runners/{replay,analyze_dev}.py`, `scripts/analyze_dev.py`; tes `test_metrics_core.py`, `test_analysis.py`. Perubahan: `baselines/registry.py` (serialisasi spesifikasi, inf-aman), blok config `protocol.selection.{families,reference_b4}` dan `protocol.stats.bootstrap_seed`, validasi terkait.

Hasil validasi: **134 tes lulus** (`python -W error -m unittest discover -s tests -t .`), tanpa ResourceWarning. Verifikasi independen: hypervolume 3D cocok dengan inklusi-eksklusi brute-force (25 kasus acak, titik kembar, koordinat pertama sama); McNemar eksak cocok dengan `scipy.stats.binomtest` (20 kasus) dan nilai referensi manual (p=0,03857); frontier dicek terhadap definisi dominasi pada 200 titik acak. Jalur penuh pada trace mock (36 query): replay -> seleksi -> pembekuan; kebijakan A3 terpilih dari berkas beku dieksekusi langsung (mock) dan sama dengan hasil replay tersimpan (jawaban, EM, jumlah panggilan, biaya, latensi); manipulasi berkas beku terdeteksi; trace split test ditolak di semua jalur; skenario mustahil menghasilkan None (tidak dilonggarkan); perubahan config sejak trace tercatat. CLI `scripts/analyze_dev.py` berjalan end-to-end pada trace mock dan plot diperiksa visual.

Koreksi diri: tes pertama gagal karena saya mengklaim argmin leksikografis Pareto-optimal secara global. Yang benar: Pareto-optimal dalam keluarganya (A1 terpilih dapat didominasi A3). Kode benar; docstring dan tes diperbaiki.

Catatan batas (tidak ada klaim keberhasilan):
- Semua angka dari tahap ini berasal dari **mock**, bukan hasil eksperimen. Belum ada trace API nyata.
- Dengan dev 100 query dan sekitar 300 konfigurasi A3, seleksi rentan overfit; CI bootstrap error kebijakan terpilih ikut dilaporkan di berkas beku. Dengan pilot 20 query, seleksi praktis hanya untuk uji alur.
- Proyeksi biaya adalah ekstrapolasi dari pilot kecil (faktor aman 1,5), bukan jaminan.
- Latensi pada seluruh analisis dev berlabel `composed_estimated` dan tidak boleh dibandingkan langsung dengan latensi `measured_e2e` di test (dijaga `compare_policies`).
- P12 (B4 acuan: N=3, het, R=2) adalah default sementara.

## Tahap 4: hasil

File baru: `data/{types(Tahap 3),render,sampling,io,prepare}.py`, `scripts/prepare_data.py`, `metrics/{qa,outcome,dev_stats}.py`, `routing/thresholds.py`, `baselines/{policies,registry,oracle}.py`, `ablations/variants.py`; tes `test_{qa,data,baselines}.py` dan perluasan `test_equivalence.py`. Perubahan: `routing/policy.py` direfaktor (`stage1_gate`, `escalate`) agar A3, A1, A2, B4 memakai logika tahap 2-3 yang sama; `LiveSession`/`TraceSession` menambah `slm_answer`, `strong`, `sc`; blok config `baselines`.

Hasil validasi: **105 tes lulus** (`python -W error -m unittest discover -s tests -t .`), tanpa jaringan. Tes utama: untuk semua spesifikasi kebijakan (B1-B5, A1-A3; 157 kebijakan x 30 query pada mock) replay dari trace == eksekusi langsung (jawaban, tahap, EM/F1, jumlah panggilan, biaya, latensi composed == terukur di jam simulasi). Oracle diuji tidak pernah lebih buruk dari komponen mana pun. Sampler diuji deterministik, disjoint, alokasi proporsional; pelanggaran skema data menimbulkan `DataSchemaError`.

Dua kegagalan tes di tahap ini adalah kesalahan ekspektasi tes (data sintetis kurang miring; subjek terbanyak tidak dijamin paling sering muncul), bukan bug kode; tes diperbaiki.

Catatan batas (tidak ada klaim keberhasilan):
- **Unduhan HF nyata belum diuji.** Nama kolom (`cais/mmlu`: question, subject, choices, answer; `hotpotqa/hotpot_qa` distractor: id, question, answer, type, level, context.title/sentences) dan jumlah baris harapan (MMLU validation 1531, test 14042; HotpotQA validation 7405) berasal dari ingatan, bukan verifikasi. Skema yang tidak cocok menghentikan skrip; jumlah baris yang berbeda dicatat di `manifest.json` (`count_warnings`).
- Evaluator HotpotQA ditulis ulang dari definisi metrik (bukan disalin); belum dibandingkan dengan skrip evaluasi resmi.
- Dengan 100 soal dev dari 57 subjek MMLU, banyak subjek kecil mendapat alokasi 0 (dicatat di manifest).
- Panggilan B2 tanpa parameter temperatur dan dengan `reasoning_effort` dari config; kompatibilitas parameter pada GPT-6.1-terra belum diuji.
- B6 (router terlatih) tetap di luar lingkup.

## Tahap 3: hasil

Permintaan "lanjutkan tahap 3" diterima tanpa jawaban P1-P11, hasil probe, atau verifikasi harga. **P1-P11 dipakai sebagai default sementara** (semua dapat diubah lewat config). Jalur live tetap terkunci oleh `Config.validate_live()` (probe_confirmed, gaya logprobs bukan `auto`, harga terverifikasi, API key).

File baru: `llm/{base,http_client,recording,mock}.py`, `routing/{answers,disagreement,nodes,runner,sessions,policy,accounting,collect}.py`, `verification/p_true.py`, `agents/agent.py`, `debate/protocol.py`, `prompts/templates.py`, `data/types.py`; tes `test_{answers,disagreement,p_true,policy,equivalence,recording,http_client}.py`, `tests/helpers.py`. Config: blok `llm` dan `verification` ditambahkan.

Hasil validasi: **73 tes `unittest` lulus** (`python -W error -m unittest discover -s tests -t .`), semuanya tanpa jaringan. Tes kunci (`test_equivalence.py`): untuk 36 query x 60 theta (satu grid) replay dari trace lengkap == eksekusi langsung pada mock deterministik: jawaban akhir, tahap berhenti, alasan eskalasi, biaya, dan latensi composed == latensi terukur (jam simulasi, overhead 0); eksekusi langsung hanya memanggil tahap yang dibutuhkan. Cabang ekstrem (banyak jawaban invalid; backend tanpa logprobs) juga dicakup dan dibuktikan tercapai.

Bug yang ditemukan tes dan sudah diperbaiki:
1. `vr.p_true, vr.meta.update(meta)` bukan penugasan: P(True) tidak tersimpan di trace sehingga replay memaksa eskalasi sementara eksekusi langsung tidak. Ditangkap oleh tes kesetaraan.
2. Cache menggabungkan sampel independen antar ronde debat yang konvergen (prompt identik). Kunci cache kini memuat identitas logis (tahap, node, ronde, sampel); cache hanya berbagi tahap 1 antara trace het dan hom.

Keputusan implementasi (tambahan pada DESIGN.md bagian 2b): verifikasi dilewati bila jawaban SLM invalid; Yes/No yang hilang dari top-k dianggap 0 dan diberi flag `p_true_partial`; latensi tercatat mencakup retry; kegagalan API menjadi catatan error tak-bertagih (jawaban invalid), sedangkan usage hilang dari respons sukses menghentikan run (`UsageMissingError`); BudgetGuard menghentikan run pada 90% cap.

Catatan batas:
- Mock membangkitkan jawaban dari hash; **bukan** hasil eksperimen dan tidak boleh dikutip sebagai hasil.
- `HTTPClient` hanya diuji terhadap server lokal palsu (parameter per provider, retry, parse usage). Perilaku Together/OpenAI nyata (format logprobs, field reasoning, `max_completion_tokens` pada GPT-6.1-terra, konten kosong pada model reasoning) belum diketahui.
- Konteks debat: respons penuh agen lain dimasukkan; batas konteks Llama-3-8B-Lite (8K) belum diperiksa pada HotpotQA, kegagalan akan tercatat sebagai error/invalid.
- Pemutus seri memakai `tie_order` yang harus diisi dari akurasi agen di dev (belum ada; Tahap 5).
- `requirements.txt` belum di-pin.

## Tahap 2: hasil

File dibuat (proyek `adaptive-routing-thesis/`):
- `configs/default.yaml` (konfigurasi terpusat; penanda [D1..D4], [PROVISIONAL], [PROBE]), `configs/pricing.yaml` (semua harga `verified_on: null`).
- `adaptive_routing/config.py` (`Config.validate()`, `validate_live()`; menolak pelanggaran D1-D4 dan run live tanpa probe/harga terverifikasi/API key).
- `adaptive_routing/llm/pricing.py` (model tak dikenal = error, bukan 0).
- `adaptive_routing/routing/entropy.py` (entropi, partisi, ruang ambang diskret, pelebur ambang ekuivalen).
- `adaptive_routing/tracing/schema.py` (`CallRecord`, `QueryRecord`, `PolicyOutcome`, `TraceManifest`, JSONL, `request_key`).
- `docs/DESIGN.md` (desain final), `docs/MODIFICATIONS.md` (tidak ada kode MasRouter disalin), `README.md`, `requirements.txt`, `.env.example`, `.gitignore`.
- `tests/` (28 tes `unittest`; semua lulus di sandbox tanpa jaringan).
- `scripts/probe_backends.py` diperbarui: tanpa vLLM; `slm_alt_qwen35_9b` hanya jalan bila diminta (`--include-alt`), tidak pernah mengganti SLM otomatis.
- Direktori kosong dengan penanda tahap: `llm, agents, verification, routing, debate, data, baselines, ablations, metrics, tracing, runners, prompts`.

Temuan penting:
1. **Batas ambang entropi.** Entropi eksak 2-1-0 = 0,636514 > 0,6365 (angka bulat D4). Dengan `d <= tau` tanpa toleransi, pola 2-1-0 ditolak pada tau=0,6365. Diperbaiki dengan toleransi perbandingan 1e-4 (`CMP_TOL`), ada tes regresi. Jarak antar level >0,1 sehingga keputusan tidak berubah.
2. **Untuk N=3, ruang efektif tau2/tau3 hanya 2 nilai** (0 = unanimous, 0,6365 = cukup mayoritas) karena syarat mayoritas unik membuat 1,0986 ekuivalen dengan 0,6365. Ambang ekuivalen dilebur. Perlu disebut di tesis.
3. Ukuran ruang theta pada N=3: sekitar 312 konfigurasi (13 tau1 x 2 x 2 x 3 R x 2 S); dengan 100 query dev, risiko overfit seleksi dicatat di DESIGN.md.
4. Debat Tahap 3 dan B4 membutuhkan debat pada semua query di trace (B4 dan A1 menjalankannya tanpa gerbang), jadi tidak ada penghematan dengan melewati debat pada query unanimous.

## Probe backend (dilaporkan pengguna)
- 2026-11-03: `probe_backends.py --only slm_alt_ollama` pada mesin pengguna: `ok=True usage=True reasoning_tokens=None empty=False` (Ollama lokal, llama3.2:3b; untuk tugas kuliah, **bukan** untuk tesis). Isi `probe_ollama.json` belum saya lihat; hanya baris ringkasan. Probe Together dan OpenAI **belum dijalankan**.
- Pengguna menanyakan penggantian B2 (OpenAI) ke model Together agar hanya satu penyedia; menunggu keputusan (lihat jawaban: perlu dicatat sebagai deviasi dari rancangan; kandidat belum diverifikasi).

## Belum teruji (tidak ada klaim keberhasilan)
- Pemanggilan API apa pun (Together, OpenAI): belum dijalankan; sandbox tanpa internet/API key.
- Dukungan dan format logprobs Llama-3.2-3B-Instruct-Turbo di Together: **belum diuji**. Dokumentasi Together hanya mencontohkan Qwen3.5-9B.
- Id model, status deprecation, dan harga resmi: belum diverifikasi.
- `probe_backends.py` hanya diuji terhadap server palsu lokal (dua format logprobs, fallback parameter).
- Pengunduhan MMLU/HotpotQA dan jumlah baris tiap split: belum diverifikasi; jalankan `python scripts/prepare_data.py` dan periksa `data/*/manifest.json`.

## Yang harus dijalankan pengguna sebelum run live
1. Tinjau `docs/DESIGN.md`; jawab titik P1-P11 (setuju atau ubah). Sementara ini dipakai sebagai default.
2. `python scripts/probe_backends.py --out probe_report.json` (butuh `TOGETHER_API_KEY`, `OPENAI_API_KEY`); kirim `probe_report.json` tanpa kunci.
3. Verifikasi harga ke halaman resmi Together dan OpenAI, isi `verified_on` di `configs/pricing.yaml`.
4. Periksa daftar deprecation Together untuk model yang dipakai.

## Pekerjaan berikutnya
Tahap 6: runner eksperimen: `collect_trace` (dev, pilot dulu, BudgetGuard, cache), `run_direct` (test; memverifikasi `selected_policies.json`, tanpa cache, sekuensial, latensi terukur), konfigurasi Ollama untuk tugas, laporan akhir. Sebelum run live: `probe_backends.py` untuk Together (dan B2), verifikasi harga, `prepare_data.py`, serta konfirmasi P1-P12 dan keputusan penggantian B2.

## 1. Analisis Repository MasRouter (diverifikasi dari ZIP)

Ukuran: sekitar 4.100 baris Python. Lisensi Apache-2.0. Tidak ada `requirements.txt`.

Struktur nyata:
- `MAR/Agent` (agent.py: kelas `Agent`, `FinalRefer`), `MAR/Graph` (graph.py, node.py: graf agen dengan koneksi spasial/temporal), `MAR/LLM` (gpt_chat.py, price.py, llm_profile.py, registry), `MAR/MasRouter/mas_router.py` (373 baris, controller network terlatih), `MAR/Prompts`, `MAR/Roles` (JSON role + `FinalNode/mmlu.json`), `MAR/Tools`, `MAR/Utils`.
- `Experiments/run_{gsm8k,humaneval,math,mbpp,mmlu}.py`, `Datasets/*_dataset.py` (tidak ada HotpotQA).
- Dependensi terdeteksi dari impor: torch, sentence_transformers, openai, tiktoken, tenacity, pandas, numpy, loguru, pyyaml, python-dotenv, aiohttp, langchain_community, wikipedia, bs4, pytube, seaborn.

Temuan relevan (semua dari pembacaan kode):
1. **Biaya diestimasi, bukan dari API.** `price.py` menghitung token memakai tiktoken (encoder gpt-4o) atas teks prompt dan respons; `usage` dari API tidak dipakai. Model yang tidak ada di `MODEL_PRICE` mengembalikan biaya 0 secara diam-diam. Akumulasi memakai singleton global (`Cost`, `PromptTokens`). **Tidak sesuai** dengan Pers. II.1 tesis (token aktual per pemanggilan). Tidak dipakai.
2. **Tidak ada latensi, logprobs, atau pencatatan per pemanggilan.** `ALLChat.gen` bahkan tidak meneruskan `max_tokens`/`temperature` ke API. Klien dibuat ulang tiap pemanggilan.
3. **Inkonsistensi env:** README memakai `URL`/`KEY`; `gpt_chat.py` memakai `URL`/`KEY` (ALLChat) tetapi `BASE_URL`/`API_KEY` (GPTChat, memanggil endpoint non-standar milik penulis).
4. **"Debate" di MasRouter bukan debat Du dkk.** Implementasinya hanya prompt "berikan jawaban yang berbeda atau berlawanan dengan agen lain" (`Prompts/reasoning.py`). Tidak ada revisi iteratif setelah melihat argumen agen lain. **Tidak setara** dengan Tahap 3 tesis; Tahap 3 dibangun sendiri mengikuti Algoritma III.1.
5. **Loader MMLU:** path `datasets/MMLU/data/...` (huruf kecil) tidak cocok dengan folder `Datasets/` pada filesystem case-sensitive; `Datasets/MMLU/download.py` diimpor tetapi tidak ada di repo. Ekstraksi jawaban `postprocess_answer` mengambil huruf pertama respons (rapuh). Tidak dipakai; dibuat loader baru.
6. **Protokol evaluasi MasRouter:** `run_mmlu.py` melatih router pada split `dev` lalu menguji pada `test`, melewati batch awal; router dilatih dengan REINFORCE dan penalti biaya. Ini inti MasRouter dan bertentangan dengan Batasan Masalah no. 4 tesis (tidak melatih router).
7. Debat/graf MasRouter bergantung pada `torch` (matriks adjacency terlatih). Tidak diperlukan untuk metode tesis.

Keputusan penggunaan ulang:
- **Tidak dipakai:** MasRouter controller, price.py, gpt_chat.py, dataset loaders, graph/node, Tools.
- **Dipakai sebagai referensi konsep saja:** role JSON `Roles/Commonsense` dan `FinalNode/mmlu.json` (inspirasi prompt agen, dicatat atribusinya bila disalin), pola `.env` URL/KEY.
- **Opsional B6:** MasRouter sebagai baseline router terlatih hanya jika diminta kemudian; butuh pelatihan, torch, dan sentence_transformers, serta profil LLM baru. Disarankan ditunda atau dikeluarkan dari lingkup awal (B6 di tesis memang "opsional").
- Atribusi: bila ada kode atau prompt disalin, sertakan `NOTICE` Apache-2.0 dan catat perubahan di `docs/MODIFICATIONS.md`.

## 2. Validasi model dan backend (hasil pencarian web; BELUM ada pemanggilan API nyata)

| Peran | Model (id) | Backend | Keberadaan | Harga acuan USD/1M (in/out) |
|---|---|---|---|---|
| SLM [D1] | meta-llama/Llama-3.2-3B-Instruct-Turbo | Together | Terdaftar di dokumentasi Together | 0,06 / 0,06 (agregator; satu agregator mencatat N/A) |
| Agen 1 | Qwen/Qwen3.5-9B | Together | Terdaftar | 0,10 / 0,15 (agregator) |
| Agen 2 | meta-llama/Meta-Llama-3-8B-Instruct-Lite | Together | Terdaftar (INT4, konteks 8K) | 0,10 / 0,10 (agregator) |
| Agen 3 | zai-org/GLM-5.3-Flash | Together | Terdaftar; dirilis 2026-08-26; konteks besar | 0,15 / 0,50 (beberapa agregator) |
| Strong / B2 | gpt-6.1-terra | OpenAI | Dirilis 2026-10-26 | 1,00 / 5,00 (pengumuman OpenAI dikutip media; cache 0,05) |

Catatan:
- **Semua harga berasal dari sumber sekunder.** Halaman harga resmi tidak dibuka. Harus diverifikasi manual ke halaman resmi sebelum eksperimen (sesuai instruksi) dan dimasukkan ke `configs/pricing.yaml` beserta tanggal verifikasi.
- **Logprobs:** dokumentasi Together menyebut `logprobs: 1` (integer) dengan contoh pada Qwen/Qwen3.5-9B; dukungan untuk Llama-3.2-3B-Instruct-Turbo **tidak terkonfirmasi**. Format Together berbeda dari OpenAI (`logprobs` integer vs `logprobs: true` + `top_logprobs`). vLLM mendukung gaya OpenAI. Dukungan dan format **harus diuji** dengan `scripts/probe_backends.py` sebelum P(True) diimplementasikan.
- **Model reasoning:** GLM-5.3-Flash dan GPT-6.1-terra adalah model reasoning (Terra punya enam level reasoning effort; Qwen3.5-9B punya sakelar `reasoning.enabled`). Token reasoning memengaruhi biaya dan latensi; konfigurasi (mode non-thinking atau effort rendah) harus ditetapkan dan dicatat seragam. Status default thinking pada GLM-5.3-Flash di Together belum terkonfirmasi.
- GPT-6.1-terra: pemanggilan tool diarahkan ke Responses API saat reasoning aktif (sumber: LiteLLM). Untuk B2 cukup Chat Completions; kompatibilitas parameter (`max_completion_tokens`) diuji oleh probe.
- Risiko deprecation: Together menerbitkan daftar deprecation (Llama 3.1 8B Turbo sudah dijadwalkan 2026-03-06). Status Llama-3-8B-Lite dan Llama-3.2-3B-Turbo **belum diperiksa**; periksa `docs.together.ai/docs/deprecations` sebelum pilot.
- Heterogenitas agen: Llama (Meta), Qwen (Alibaba), GLM (Z.ai) berasal dari tiga keluarga berbeda; SLM Llama 3.2 satu keluarga dengan Agen 2. Dicatat sebagai ancaman validitas (kesalahan berkorelasi, Tabel III.1).

## Temuan pada dokumen tesis (Tahap 1)
1. Rujukan silang salah: persamaan biaya/latensi disebut "Subbab II.1.6" (seharusnya II.1.7); dominasi Pareto "II.1.7" (seharusnya II.1.8).
2. Heading II.1.5 dan II.1.6 sama; isi II.1.5 menduplikasi II.1.4. Rumus disagreement di II.1.5 kosong (rumus ada di Pers. II.6-II.7).
3. Pers. III.1/III.2 memakai `D_q > tau2` saja, sedangkan Algoritma III.1 juga mensyaratkan "mayoritas unik". Implementasi mengikuti Algoritma III.1.
4. Tabel III.3 menyebut `S` homogen/heterogen, sedangkan model terpilih adalah tiga keluarga berbeda; varian homogen perlu definisi (misal 3 salinan SLM atau Qwen).


## Error / catatan
- GitHub menolak akses otomatis; teratasi dengan ZIP dari pengguna. `pdftotext` gagal pada PDF MasRouter; teks dari OCR.
- pytest tidak tersedia di sandbox; tes memakai `unittest` (kompatibel dengan pytest).
