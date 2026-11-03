# DESIGN.md: Desain final untuk tinjauan (Tahap 2)

Status: **menunggu validasi pengguna. Tidak ada eksperimen berbiaya yang dijalankan.**
Sumber metode: usulan tesis (Algoritma III.1, Pers. II.1-II.9, III.1-III.2, Tabel III.3-III.5).

## 1. Keputusan

| ID | Keputusan | Sumber |
|---|---|---|
| D1 | SLM = Llama-3.2-3B-Instruct-Turbo via Together AI (tanpa vLLM). Logprobs aktual wajib; bila tidak tersedia, laporkan alternatif sebelum mengubah metodologi | Pengguna |
| D2 | Satu trace lengkap per query pada dev; semua (tau, R, S) dievaluasi offline. Latensi replay berlabel `composed_estimated`. Kebijakan terpilih (dan pembanding) dijalankan langsung pada test: `measured_e2e`. Test tidak dipakai untuk memilih apa pun | Pengguna |
| D3 | MMLU: dev = split `validation` resmi, test = split `test` resmi, stratifikasi per subjek, seed tetap. HotpotQA distractor: `validation` dibagi dev/test disjoint (seed tetap). Ukuran awal: pilot 20, dev 100, test 300. CI dilaporkan; uji berpasangan McNemar (biner) atau paired bootstrap (kontinu) | Pengguna |
| D4 | Log natural. N=3: 0 (3-0-0), 0,6365 (2-1-0), 1,0986 (1-1-1). Ambang dievaluasi pada ruang diskret, bukan kontinu | Pengguna |

Semua tertanam di `configs/default.yaml` dan dijaga oleh `Config.validate()` (28 tes unit lulus).

## 2. Aturan keputusan (Algoritma III.1) dan perbandingan ambang

1. Tahap 1: SLM menjawab (`s1_answer`), lalu verifikasi (`s1_verify`, berurutan karena bergantung pada jawaban). `p = P(Yes) / (P(Yes)+P(No))` dari alternatif token pertama (top-k=5). **Terima jika `p >= tau1`** (inklusif).
2. Tahap 2 (jika tidak diterima): N agen heterogen paralel dan independen. Entropi `d` (log natural) atas distribusi jawaban ternormalisasi. **Terima jika `d <= tau2` DAN mayoritas unik**; kembalikan mayoritas.
3. Tahap 3: untuk r=1..R, tiap agen merevisi setelah melihat respons lengkap agen lain ronde sebelumnya. **Berhenti jika `d_r <= tau3` DAN mayoritas unik.** Bila sampai R tanpa berhenti: mayoritas dengan aturan pemutus seri.
4. Perbandingan entropi memakai toleransi 1e-4 (`CMP_TOL`). Ditemukan oleh tes: entropi eksak 2-1-0 adalah 0,636514, sedikit di atas 0,6365; tanpa toleransi pola 2-1-0 ditolak pada tau=0,6365. Jarak antar level >0,1, jadi toleransi tidak mengubah keputusan.
5. **Ruang tau diskret.** tau2 dan tau3 hanya mengambil nilai dari entropi yang mungkin untuk N (N=3: 3 nilai). Karena syarat mayoritas unik, tau=1,0986 menerima pola yang sama dengan tau=0,6365 (1-1-1 tidak punya mayoritas unik), sehingga **ruang efektif N=3 hanya dua pilihan: 0 (harus bulat/unanimous) atau 0,6365 (cukup mayoritas)**. Ambang ekuivalen dilebur (`dedupe_equivalent`). Untuk N=5 ruang lebih kaya dan entropi tidak monoton terhadap mayoritas unik (mis. 2-1-1-1 entropi 1,332 bermayoritas unik, 2-2-1 entropi 1,055 tidak), sehingga syarat mayoritas unik memberi informasi tambahan.
6. tau1: kuantil level {0,0.1,...,1.0} dari distribusi P(True) **pada dev**, ditambah sentinel `always_accept` dan `never_accept`.
7. **Jawaban invalid** [PROVISIONAL]: setiap jawaban invalid menjadi label unik (tidak pernah membentuk mayoritas, menaikkan entropi; N tetap karena biaya sudah terjadi). SLM invalid memaksa eskalasi. Tidak ada jawaban valid di akhir dihitung salah. Jumlah invalid per model dilaporkan.
8. Pemutus seri [PROVISIONAL]: urutan agen menurut akurasi pada **dev**, ditetapkan sebelum test; cadangan: urutan leksikografis label.
9. Ukuran P(True): tesis (Pers. II.5) mendefinisikan `P(Yes | q, y, p_v)`; saya memakai normalisasi atas Yes/No [PROVISIONAL]. Logprobs mentah disimpan, jadi versi tanpa normalisasi dapat dihitung ulang sebagai sensitivitas. Bila Yes/No tidak muncul pada top-k, p tidak terdefinisi, eskalasi dipaksa, dan jumlah kasusnya dilaporkan.

## 2b. Arsitektur router (Tahap 3, diimplementasikan)

- `routing/policy.py::run_policy(theta, session, tie_order)` adalah **satu-satunya** tempat keputusan Algoritma III.1. Router meminta tahap lewat antarmuka sesi:
  - `LiveSession` memanggil LLM secara *lazy* (hanya tahap yang diminta) dan mencatat `CallRecord`. Dipakai untuk eksekusi langsung (test) dan pengumpulan trace lengkap (dev, `routing/collect.py`).
  - `TraceSession` membaca `CallRecord` tersimpan (replay). Trace yang tidak memuat panggilan yang dibutuhkan menimbulkan `TraceMissing`, bukan nilai default.
- Verifikasi (`s1_verify`) **dilewati bila jawaban SLM invalid** (tidak ada jawaban untuk diverifikasi); keputusan tetap eskalasi. Ini merevisi kalimat "tahap 1 selalu 2 pemanggilan" pada bagian 5.
- Cache (`RecordingLLM`) hanya berlaku bila prompt, riwayat, parameter **dan identitas logis** (tahap, node, ronde, sampel) identik. Tujuan praktisnya: berbagi tahap 1 antara trace het dan hom. Pemanggilan yang di-cache tetap dicatat dengan biayanya (untuk akuntansi kebijakan) tetapi tidak dihitung dua kali pada anggaran. Tes kesetaraan menemukan bahwa cache tanpa identitas logis menggabungkan sampel independen antar ronde debat yang konvergen.
- P(True): logprobs token pertama (top-k=5); token Yes/No diagregasi atas varian (`Yes`, ` yes`, `▁Yes`...). Bila hanya satu sisi muncul di top-k, sisi yang hilang dianggap 0 dan hasilnya diberi `p_true_partial=True` (artinya probabilitasnya di bawah peringkat ke-5); bila keduanya hilang atau logprobs tidak ada, `p=None` dan eskalasi dipaksa. Jumlah kasus dilaporkan.
- Backend LLM: `HTTPClient` (stdlib, kompatibel OpenAI) dengan retry untuk 408/409/425/429/5xx; latensi tercatat = total waktu termasuk retry dan backoff; gaya logprobs `auto` ditolak sampai ditetapkan dari probe.

## 3. Alur data

```
datasets (HF) --prepare--> data/{mmlu,hotpotqa}/{pilot,dev,test}.jsonl  (versi, sumber, split, seed, preprocessing dicatat)
                                         |
        dev only                         v                              test only
 +-----------------------+    +-----------------------+      +--------------------------+
 | TRACE COLLECTOR       |    | selected_policies.json|      | DIRECT RUNNER            |
 | satu trace lengkap    |    | (dibekukan, hash +    |----> | kebijakan terpilih + B1-5|
 | per query:            |    |  config_hash, sebelum |      | A1, A2 dieksekusi nyata, |
 |  SLM jawab+verifikasi |    |  test)                |      | tanpa cache, sekuensial, |
 |  3 agen paralel       |    +-----------^-----------+      | latensi measured_e2e     |
 |  debat 3 ronde        |                |                  +------------+-------------+
 |  B2, B3 (k sampel)    |                |                               |
 +-----------+-----------+                |                               v
             | calls.jsonl                |                  results/test_direct/*.jsonl
             v                            |                               |
 +-----------------------+   Pareto filter pada dev                       v
 | REPLAY ENGINE         |---(3 objektif)--+              Metrik, CI, McNemar / paired bootstrap,
 | rekonstruksi keputusan|   pilih per skenario            frontier + hypervolume (deskriptif)
 | untuk semua theta     |   yang dinyatakan sebelumnya
 | latensi composed_est. |
 +-----------------------+
 results/dev_replay/*     (terpisah dari results/test_direct/*)
```

## 4. Struktur trace (`adaptive_routing/tracing/schema.py`)

Berkas per run: `results/traces/{dataset}/{split}/{run_id}/{manifest.json, queries.jsonl, calls.jsonl}`.

- `manifest.json`: run_id, mode, config_hash, git_commit, tanggal, daftar query_id, model_id per peran (versi), **snapshot harga** (dengan status verifikasi), seed sampling.
- `queries.jsonl` (`QueryRecord`): query_id, dataset, split, gold, source_index, meta (mis. subject).
- `calls.jsonl` (`CallRecord`), satu baris per pemanggilan, memuat:
  identitas (query_id, stage, `group_id` kelompok paralel, node_id, round, sample_idx); model (model_key, provider, model_id); kunci cache (`request_key` = hash model+prompt+riwayat+parameter); parameter; **waktu** (`t_submit_utc`, `t_start_mono`, `t_end_mono`, `latency_s`); **usage dari API** (input_tokens, output_tokens ditagih, reasoning_tokens, cached_input_tokens, `token_source`); **snapshot harga**, `cost_usd`, `pricing_verified`; **respons** (teks penuh, finish_reason, `logprobs_first_token`, parsed_answer, parse_ok, p_true); error, attempt, cache_hit.
- Aturan keras: token tidak pernah diperkirakan dari teks (beda dengan MasRouter). `token_source='estimated'` ditolak di luar mode mock; pemanggilan sukses tanpa usage membuat `finalize_cost` gagal.
- Hasil kebijakan (`PolicyOutcome`) wajib memiliki `latency_kind` bernilai `composed_estimated` atau `measured_e2e`.

## 5. Perhitungan biaya dan latensi

Untuk query q, kebijakan theta, trace T_q (hanya tahap yang **dieksekusi menurut keputusan hasil rekonstruksi**):

- **Biaya** `C(q;theta) = sum cost_usd` atas semua pemanggilan yang dieksekusi, dengan `cost_usd = (tok_in*p_in + tok_out*p_out)/1e6` (Pers. II.1), token dari `usage` API. Tahap 1 selalu (2 pemanggilan); tahap 2 bila tidak diterima tahap 1; tiap ronde debat hingga ronde berhenti. Ronde di trace yang tidak dieksekusi tidak dihitung.
- **Latensi replay (composed_estimated)**:
  `L(q;theta) = L(a0) + L(v0) + 1[tahap2]*(max_i L_i + o) + sum_{r<=r_stop} (max_i L_{i,r} + o)`,
  dengan L dari `latency_s` tercatat, max atas pemanggilan paralel dalam satu `group_id`, `o` = `coordination_overhead_s` (default 0, jadi **batas bawah** yang tidak diukur). Dilaporkan sebagai mean, median, P95 dan **tidak** disebut end-to-end.
- **Latensi langsung (measured_e2e)**: `perf_counter` dari query diterima hingga jawaban akhir; query dijalankan sekuensial (tanpa konkurensi antar query), tanpa cache; `latency_repeats` kali.
- Pareto memakai: mean biaya, mean latensi, tingkat kesalahan (MMLU: 1-accuracy; HotpotQA: 1-F1 [PROVISIONAL]; EM dilaporkan juga).
- Keterbatasan yang dicatat: latensi per pemanggilan saat trace dikumpulkan mencakup antrian API; hasil replay tidak menangkap overhead koordinasi, sehingga kebijakan multi-agen terlihat lebih cepat di replay daripada di eksekusi langsung (ini diuji pada test).

## 6. Baseline dan ablasi (dari trace yang sama bila memungkinkan)

| Kode | Definisi operasional | Catatan |
|---|---|---|
| B1 | jawaban SLM | |
| B2 | satu pemanggilan model kuat | |
| B3 | self-consistency SLM, k sampel (temperatur 0,7), mayoritas | k=5 [PROVISIONAL] |
| B4 | tahap 2 + R ronde debat pada **semua** query (R=1..3; het/hom) | |
| B5 | cascade P(True): SLM, lalu B2 bila p < tau_B5 | tau_B5 dari grid yang sama |
| B6 | router terlatih | **di luar lingkup awal** (opsional di tesis) |
| B7 | oracle: per query pilih kebijakan termurah yang benar di antara B1-B4; analisis batas atas, bukan operasional | berlabel composed |
| A1 | tanpa gerbang disagreement: setelah gagal tau1, tahap 2 lalu R ronde debat tanpa penghentian dini | [PROVISIONAL] interpretasi |
| A2 | tanpa verifikasi awal: semua query ke tahap 2 (+gerbang tau2) lalu tahap 3 | |
| A3 | kebijakan penuh | |

### 6b. Definisi yang diimplementasikan (Tahap 4)

- **B1**: hanya pemanggilan jawaban SLM (tanpa verifikasi). **B2**: satu pemanggilan model kuat (tanpa temperatur; parameter reasoning dari config). **B3**: k=5 sampel SLM paralel (T=0,7), mayoritas; seri = sampel berindeks terkecil.
- **B4**: tahap 2 + R ronde pada semua query, tanpa tahap 1 dan tanpa penghentian dini; jawaban akhir = mayoritas dengan pemutus seri.
- **B5**: SLM + verifikasi; bila `p < tau`, jawaban SLM invalid, atau P(True) tak terdefinisi, eskalasi ke B2; jawaban akhir = jawaban model kuat. Grid tau = grid tau1 (tau=+inf: selalu model kuat).
- **A1**: gerbang tau1 tetap; bila eskalasi, tahap 2 lalu R ronde penuh tanpa gerbang entropi. **A2**: tanpa tahap 1; tahap 2 (gerbang tau2) lalu tahap 3 (gerbang tau3). **A3** = router penuh.
- **B7 (oracle)**: per query, hasil terbaik di antara himpunan kandidat yang diberikan (skor utama tertinggi, lalu biaya, lalu latensi); wajib himpunan lengkap; berlabel `composed_estimated`; per dataset.
- Skor utama (P4): MMLU = akurasi; HotpotQA = F1; error = 1 - skor. EM tetap dilaporkan; `correct` HotpotQA = EM=1 (dipakai McNemar).
- Ambang tau1 = kuantil P(True) **dev** pada nilai teramati ('higher') + sentinel -inf/+inf; tau2 dan tau3 dari entropi yang mungkin (ambang ekuivalen dilebur). Pemutus seri = peringkat akurasi agen pada dev (fungsi menolak split selain dev).
- Ukuran ruang pada N=3 dengan 4 nilai tau1: 96 A3, 24 A1, 24 A2, 6 B4, 4 B5, ditambah B1-B3 (`build_specs`).

## 7. Protokol evaluasi

1. **Sampling** (seed tetap; diimplementasikan di `data/sampling.py` dan `data/prepare.py`; manifest mencatat alokasi per subjek, subjek beralokasi 0, tumpang-tindih pertanyaan validation vs test, versi/fingerprint dataset): MMLU: alokasi proporsional per subjek (largest remainder) dari `validation` (dev) dan `test` (test); dengan 100 soal dev beberapa subjek kecil mendapat 0. HotpotQA: permutasi seed atas `validation` (7.405 per perkiraan, diverifikasi saat unduh); 100 pertama = dev, 300 berikutnya = test, disjoint. Pilot = 20 pertama dev (test tidak tersentuh). Soal yang diambil dicatat dalam manifest.
2. **Trace dev** (pilot dahulu, lalu dev penuh): satu trace lengkap per query. Anggaran dijaga `BudgetGuard` (cap pilot USD 20, henti pada 90%).
3. **Replay dev**: enumerasi theta (tau1: 11 kuantil + 2 sentinel; tau2 dan tau3: ruang efektif N=3; R: 1-3; S: het/hom) dan baseline. Pareto filter (dominasi) per benchmark atas tiga objektif.
4. **Pemilihan** pada dev dengan skenario yang dinyatakan **sebelum replay** (disimpan di config, di-hash): minimalkan error dengan batas relatif (biaya <= 50% atau 25% biaya B2; latensi <= 50% latensi B4); seri: biaya lalu latensi. Hasil dibekukan di `selected_policies.json` (hash + config_hash).
5. **Test langsung**: kebijakan terpilih **dan** pembanding (B1-B5, A1, A2) dijalankan langsung supaya latensi sebanding (tidak membandingkan measured dengan composed). B7 dari hasil langsung berlabel composed. Direct runner menolak jalan bila `selected_policies.json` tidak ada atau hash tidak cocok.
6. **Statistik**: bootstrap query 10.000 kali, CI 95%; McNemar (akurasi MMLU, EM biner); paired bootstrap (F1, biaya, latensi). Komparasi primer dinyatakan di muka: H1 adaptif vs B4, H2 A3 vs A1, H3 frontier 3 objektif vs proyeksi biaya-akurasi; lainnya eksploratori.
7. **Hypervolume**: normalisasi min-max atas gabungan titik dev, titik referensi 1,1; pada test deskriptif saja.
8. **Keterbatasan statistik** (wajib dilaporkan): n=300 memberi setengah-lebar CI 95% sekitar 5,7 poin persentase (akurasi 50%); n=100 sekitar 9,8 poin. Seleksi pada dev rentan overfit (sekitar 300 kandidat pada 100 query), jadi pilihan dibatasi pada grid kasar dan titik Pareto. Selisih kecil tidak diklaim signifikan; hasil negatif (mis. B2 unggul) dilaporkan.
9. **Pemisahan hasil**: `results/dev_replay/` vs `results/test_direct/`; tidak ada kode yang menulis ke keduanya.

## 7b. Analisis dev yang diimplementasikan (Tahap 5)

`python scripts/analyze_dev.py --trace <dir trace dev> [--trace ...] --out results` melakukan, per benchmark:
1. Membaca trace (manifest, queries, calls), menolak split selain dev dan config_hash yang berbeda antar trace.
2. tau1 = kuantil P(True) dev + sentinel; `build_specs`; peringkat agen (pemutus seri) dari dev.
3. Replay SEMUA kebijakan (latensi `composed_estimated`, overhead dari config) -> `results/dev_replay/{ds}/outcomes.jsonl`.
4. Ringkasan per kebijakan (error, biaya, latensi mean/median/P95), frontier 3 objektif (dominasi; titik identik sama-sama tetap), hypervolume eksak 3D (normalisasi min-max atas gabungan titik dev, referensi 1,1) untuk frontier global dan tiap keluarga.
5. Seleksi per skenario dan per keluarga (A3, A1, A2, B4, B5): kebijakan layak dengan error terkecil; seri: biaya, latensi, policy_id. Batas: biaya <= rasio x biaya B2; latensi <= rasio x latensi B4 acuan (`reference_b4`, [PROVISIONAL P12]: N=3, het, R=2). **Tidak layak -> None, batas tidak dilonggarkan.** Hasil Pareto-optimal dalam keluarganya (bukan jaminan global).
6. Pembekuan `results/selected_policies.json` dengan hash SHA-256 atas isinya (`verify_selection` menolak berkas yang diubah) beserta: config_hash saat analisis dan saat trace (`config_unchanged_since_trace`), run_id trace, hash daftar query dev, skenario yang dinyatakan, tau1 kandidat, peringkat agen, nilai acuan, dan metrik dev + CI bootstrap error kebijakan terpilih. Direct runner (Tahap 6) wajib memverifikasi berkas ini dan menolak bila config berubah sejak trace tanpa override eksplisit.
7. Plot frontier (3 proyeksi), proyeksi biaya trace dev dan test langsung (ekstrapolasi dari pilot, faktor aman 1,5, bukan jaminan), porsi token reasoning per model.

Statistik (`metrics/stats.py`): bootstrap persentil atas query (seed tetap), selisih berpasangan dengan resampling query yang sama, McNemar eksak dua sisi (binomial, diverifikasi terhadap `scipy.stats.binomtest`), dan `compare_policies` yang menolak latency_kind berbeda (tidak membandingkan composed dengan measured).

## 8. Titik yang menunggu konfirmasi Anda

| ID | Usulan | Alternatif |
|---|---|---|
| P1 | Temperatur 0 (SLM dan agen), 0,7 untuk varian homogen dan B3 | semua 0,7 / seed |
| P2 | Invalid = label unik; SLM invalid eskalasi | dikeluarkan dari N |
| P3 | Pemutus seri = peringkat akurasi agen di dev | acak ber-seed |
| P4 | Error HotpotQA pada Pareto = 1 - F1 | 1 - EM |
| P5 | Definisi A1, A2, B7 (bagian 6) | A1 = eskalasi langsung ke model kuat |
| P6 | Skenario seleksi relatif terhadap B2 dan B4 | batas absolut USD/detik |
| P7 | Test langsung mencakup pembanding; latency_repeats=1 | 3 pengulangan (biaya x3 untuk kebijakan terpilih) |
| P8 | Pilot = subset dev | pilot terpisah |
| P9 | P(True) ternormalisasi Yes/(Yes+No) | P(Yes) mentah |
| P10 | HotpotQA: normalisasi gaya SQuAD (huruf kecil, tanpa tanda baca/artikel); pengelompokan mayoritas = kecocokan string ternormalisasi (sinonim dihitung berbeda) | |
| P12 | B4 acuan untuk batas latensi: N=3, het, R=2 | B4 dengan R maksimum |
| P11 | Format jawaban ketat "Final answer: X"; gagal parse = invalid | |
