# China Imports Port Clustering App

Aplikasi analitik berbasis **Streamlit** untuk mengelompokkan kombinasi **pelabuhan × kelompok komoditas SITC** di Indonesia berdasarkan karakteristik aktivitas impor dari Tiongkok dan indikator makroekonomi provinsi.

Aplikasi menggunakan metode **K-Prototypes** untuk menangani campuran fitur numerik dan kategorikal, kemudian menggunakan **Random Forest surrogate** dan **SHAP** untuk menginterpretasikan karakteristik setiap klaster.

---

## Daftar Isi

1. [Introduksi](#1-introduksi)
2. [Tujuan Aplikasi](#2-tujuan-aplikasi)
3. [Fitur Utama](#3-fitur-utama)
4. [Halaman Aplikasi](#4-halaman-aplikasi)
5. [Data Input](#5-data-input)
6. [Struktur Direktori dan Manfaatnya](#6-struktur-direktori-dan-manfaatnya)
7. [Alur Kerja Kode](#7-alur-kerja-kode)
8. [Alur Preprocessing](#8-alur-preprocessing)
9. [Alur Modeling](#9-alur-modeling)
10. [Visualisasi dan Label Tampilan](#10-visualisasi-dan-label-tampilan)
11. [Output dan Unduhan](#11-output-dan-unduhan)
12. [Menjalankan Aplikasi](#12-menjalankan-aplikasi)
13. [Konfigurasi Saat Ini](#13-konfigurasi-saat-ini)
14. [Batasan dan Catatan Implementasi](#14-batasan-dan-catatan-implementasi)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Introduksi

Aktivitas impor suatu pelabuhan tidak hanya dipengaruhi oleh volume transaksi, tetapi juga oleh jenis komoditas dan kondisi ekonomi wilayah sekitarnya. Analisis per pelabuhan saja dapat menyembunyikan perbedaan pola antar komoditas, sedangkan analisis per komoditas saja tidak menunjukkan lokasi pelabuhan yang terlibat.

Aplikasi ini membentuk observasi pada level **pelabuhan × SITC**, yaitu kombinasi antara nama pelabuhan dan kelompok komoditas SITC Rev. 4. Setiap observasi kemudian dilengkapi dengan:

- nilai impor dari pelabuhan tersebut untuk kelompok SITC terkait;
- PDRB provinsi;
- nilai investasi Tiongkok di provinsi;
- jumlah industri mikro dan kecil provinsi;
- koordinat geografis pelabuhan.

Dengan pendekatan tersebut, aplikasi tidak hanya menjawab “pelabuhan mana yang memiliki impor besar”, tetapi juga “kombinasi pelabuhan dan jenis komoditas seperti apa yang memiliki karakteristik serupa”.

Aplikasi ditujukan untuk eksplorasi data, analisis klaster, dan dokumentasi hasil penelitian pada satu lingkungan pengguna lokal. Data diproses di dalam sesi Streamlit dan tidak disimpan ke database eksternal.

---

## 2. Tujuan Aplikasi

Tujuan utama aplikasi:

1. **Mengotomatiskan prapemrosesan data impor pelabuhan**, termasuk konversi HS 2022 ke SITC Rev. 4, agregasi, penggabungan data provinsi, dan penggabungan geolokasi.
2. **Membentuk klaster pelabuhan × SITC** menggunakan K-Prototypes dengan jumlah klaster yang dapat dipilih pada rentang K = 2–8.
3. **Mengevaluasi kualitas klaster** menggunakan Elbow dan Silhouette campuran K-Prototypes.
4. **Menginterpretasikan klaster** melalui statistik deskriptif, distribusi dan normalized entropy SITC, scatter plot, boxplot, violinplot, heatmap Z-score, dan SHAP.
5. **Memetakan sebaran pelabuhan** berdasarkan klaster dan kategori SITC.
6. **Menyediakan ekspor dataset** dari setiap tahap pipeline agar hasil dapat diaudit dan direproduksi.
7. **Memisahkan evaluasi model utama K-Prototypes dari evaluasi model surrogate Random Forest**, sehingga pembaca tidak mencampur kualitas klaster dengan fidelity model interpretasi.

---

## 3. Fitur Utama

### Input dan validasi

- Empat berkas input wajib.
- Validasi nama kolom secara case-insensitive.
- Deteksi berkas kosong, kolom hilang, dan kegagalan pembacaan.
- Deteksi HS6 yang dipetakan ke lebih dari satu SITC.
- Deteksi pelabuhan tanpa data geolokasi.
- Deteksi provinsi tanpa indikator makro.
- Deteksi duplikasi nama pelabuhan pada data geolokasi.
- Validasi rentang koordinat latitude dan longitude.

### Konfigurasi klaster

- Slider K dengan rentang **2–8**.
- Nilai awal/default **K = 4**.
- Tombol **Cari K Optimal (Silhouette)** menguji K = 2–6 dan memilih rata-rata silhouette tertinggi.
- Selama pencarian, slider dinonaktifkan dan tombol menampilkan indikator loading; setelah selesai slider ditempatkan pada K terbaik.
- Perubahan data atau K mendeteksi hasil analisis lama sebagai tidak valid.

### Prapemrosesan

- Ekstraksi kode HS6.
- Konversi HS 2022 ke SITC Rev. 4.
- Agregasi nilai impor per pelabuhan × SITC.
- Penggabungan provinsi dan koordinat pelabuhan.
- Pembersihan duplikasi.
- Imputasi median untuk nilai numerik kosong.
- IQR capping dengan multiplier 3.
- Integrasi geolokasi sebagai sumber resmi provinsi dan koordinat pelabuhan.

### Modeling dan evaluasi

- K-Prototypes dengan `init="Huang"`, `n_init=5`, `random_state=42`, dan `n_jobs=-1`.
- Elbow curve untuk K = 2–8.
- Silhouette campuran K-Prototypes.
- Plot silhouette per observasi.
- Unduhan nilai silhouette seluruh data dan observasi silhouette negatif.
- Random Forest surrogate untuk mereplikasi label klaster.
- Metrik fidelity, balanced accuracy, macro F1, Cohen’s kappa, classification report, dan confusion matrix.

### Interpretasi

- Statistik deskriptif fitur numerik per klaster.
- Modus fitur kategorikal SITC per klaster.
- Distribusi proporsi SITC per klaster.
- Tingkat keragaman kategori SITC per klaster (normalized entropy).
- Donut chart distribusi kategori SITC pada analisis deskriptif setiap klaster.
- Peta statis persebaran pelabuhan per klaster berdasarkan kategori SITC yang dipilih.
- Perbandingan rata-rata fitur numerik.
- Boxplot kuartil.
- Violinplot kepadatan distribusi.
- Heatmap Z-score profil klaster.
- Analisis deskriptif naratif otomatis per klaster yang merangkum fitur numerik, SITC, dan persebaran geografis.
- SHAP global importance.
- SHAP importance per kelas klaster.
- SHAP beeswarm untuk setiap kelas klaster.
- Daftar penjelasan kode SITC 0–9 pada caption tabel modus SITC dan peta.

### Peta dan ekspor

- Filter kategori SITC pada peta.
- Peta sebaran pelabuhan dengan warna klaster.
- Identifikasi pelabuhan yang memiliki koordinat identik.
- Pratinjau dan ekspor dataset dari setiap tahap preprocessing.
- Ekspor dataset berlabel klaster.
- Ekspor subset satu klaster.

---

## 4. Halaman Aplikasi

### 4.1 Beranda (`pages/1_Beranda.py`)

Halaman pembuka yang menjelaskan:

- judul penelitian dan konteks aplikasi;
- ringkasan tiga langkah penggunaan: input data, proses klasterisasi, dan eksplorasi hasil;
- tombol **Mulai Analisis** untuk masuk ke halaman Input Data;
- keterangan bahwa pemrosesan dilakukan secara lokal.

Halaman ini tidak menjalankan preprocessing atau modeling.

### 4.2 Input Data (`pages/2_Input_Data.py`)

Halaman untuk menyiapkan seluruh input dan menjalankan pipeline.

Komponen utama:

1. **Upload empat berkas wajib**
   - `port_imports.csv`
   - `provinces_data.csv`
   - `HS2022_to_SITC_Conversion.xlsx`
   - `port_geolocation.csv`

2. **Status verifikasi berkas**
   - nama berkas;
   - jumlah baris dan kolom;
   - daftar kolom wajib;
   - pesan validasi atau error.

3. **Reset Input Data**
   - menghapus berkas dan hasil analisis dari sesi;
   - mempertahankan nilai K yang sedang dipilih.

4. **Konfigurasi K**
   - K default = 4;
   - rentang K = 2–8;
   - tombol **Cari K Optimal (Silhouette)** menguji K = 2–6;
   - slider dan tombol proses dinonaktifkan selama pencarian; slider ditempatkan pada K terbaik setelah selesai;
   - tabel skor silhouette per K ditampilkan dalam expander.

5. **Tombol proses**
   - menjalankan `run_preprocessing()`;
   - dilanjutkan `run_modeling()`;
   - menampilkan progress dan status;
   - **dinonaktifkan selama pemrosesan berjalan** (flag `analysis_running`) dan aktif kembali setelah selesai atau gagal;
   - menyimpan hasil ke `st.session_state["analysis_result"]`.

6. **Ringkasan hasil**
   - jumlah observasi final;
   - jumlah pelabuhan;
   - cakupan provinsi;
   - jumlah klaster;
   - proporsi tiap klaster;
   - silhouette dan fidelity surrogate;
   - rincian conversion, duplikasi, dan IQR capping.

7. **Navigasi hasil**
   - tombol **Lihat Hasil Analisis** menuju Dashboard;
   - tombol kembali ke Beranda.

### 4.3 Dashboard (`pages/3_Dashboard.py`)

Dashboard menggunakan sidebar internal untuk memilih lima tampilan. Halaman ini membaca hasil analisis dari session state, sehingga pengguna tidak perlu menjalankan ulang pipeline hanya untuk berpindah tampilan.

#### a. Ringkasan

- Total observasi.
- Jumlah pelabuhan unik.
- Cakupan provinsi.
- Jumlah klaster.
- Bar chart ukuran klaster.
- Pie chart proporsi klaster.
- Tabel ukuran, jumlah pelabuhan, jumlah provinsi, dan proporsi klaster.
- Analisis deskriptif karakteristik klaster: paragraf terpisah untuk setiap klaster yang mencakup karakteristik numerik, kategorikal SITC, geografis, dan identitas utama klaster.
- Setiap judul fitur numerik pada bagian A dilengkapi badge Z-score (`↑↑` jauh di atas, `↑` di atas, `↔` mendekati, `↓` di bawah, `↓↓` jauh di bawah) dengan warna lembut yang konsisten; status netral "Tidak dapat dibandingkan" muncul bila standar deviasi global nol atau nilai tidak valid.
- Setiap kartu narasi memakai aksen garis vertikal di sisi kiri yang warnanya diambil dari `cluster_color_map()`, sehingga konsisten dengan bar chart, pie chart, dan scatter plot klaster.
- Bagian B setiap kartu menampilkan donut chart distribusi kategori SITC klaster, dengan warna kategori yang konsisten antar-klaster.
- Bagian C setiap kartu menampilkan peta statis persebaran pelabuhan; kategori SITC dipilih melalui radio button di sisi kanan peta dan hanya berisi kategori yang dimiliki klaster tersebut.
- Geographic extent peta memakai batas minimum 350 km dan tetap sama untuk semua kategori SITC dalam klaster yang sama, sehingga pergantian kategori hanya mengubah marker.

#### b. Profil Klaster

Terdapat tiga tab:

1. **Profil Statistik**
   - distribusi observasi per klaster;
   - statistik deskriptif fitur numerik;
   - modus SITC per klaster beserta daftar arti kode SITC 0–9;
   - tabel dan stacked bar proporsi SITC;
   - bar chart normalized entropy keragaman kategori SITC;
   - perbandingan rata-rata fitur numerik;
   - boxplot kuartil;
   - violinplot kepadatan;
   - heatmap Z-score profil klaster.

2. **Sebaran Klaster**
   - scatter nilai impor vs nilai investasi Tiongkok;
   - scatter nilai impor vs jumlah industri mikro kecil;
   - scatter nilai impor vs PDRB;
   - seluruh scatter memakai nilai asli dan warna klaster.

3. **Analisis SHAP**
   - global feature importance;
   - heatmap kontribusi fitur per kelas klaster;
   - beeswarm per kelas klaster;
   - seluruh nilai SHAP dihitung pada data uji dan tidak difilter berdasarkan anggota klaster.

#### c. Evaluasi

Terpisah menjadi dua tab:

1. **Model Utama (K-Prototypes)**
   - metrik K, rata-rata silhouette, dan ukuran sampel evaluasi;
   - Elbow curve;
   - tabel cost per K;
   - plot silhouette per observasi;
   - ringkasan silhouette per klaster;
   - unduhan CSV silhouette;
   - konfigurasi parameter K-Prototypes.

2. **Model Surrogate SHAP (Random Forest)**
   - metrik fidelity;
   - classification report per klaster;
   - confusion matrix;
   - penjelasan bahwa metrik ini mengukur fidelity surrogate, bukan kualitas klaster.

#### d. Peta

- Pemilihan satu kategori SITC.
- Peta sebaran pelabuhan dengan warna klaster.
- Tooltip berisi provinsi, SITC, dan nilai impor.
- Deteksi pelabuhan dengan koordinat identik.
- Caption berisi jumlah pelabuhan dan daftar arti kode SITC.

#### e. Data

- Ringkasan jumlah baris setiap tahap preprocessing.
- Detail IQR capping dan imputasi median.
- Katalog dataset:
  1. input mentah;
  2. hasil konversi HS–SITC;
  3. hasil agregasi pelabuhan × SITC;
  4. hasil penggabungan provinsi;
  5. dataset final preprocessing;
  6. dataset berlabel klaster.
- Pratinjau jumlah baris.
- Unduhan CSV per tahap.
- Pratinjau dan unduhan subset per klaster.

---

## 5. Data Input

### 5.1 `port_imports.csv`

Data transaksi impor pada level pelabuhan.

| Kolom | Keterangan |
|---|---|
| `port` | Nama pelabuhan |
| `hs_code` | Kode HS 2022, dapat berisi teks dan kode dalam tanda kurung |
| `usd_value` | Nilai impor dalam USD |

### 5.2 `provinces_data.csv`

Indikator makro provinsi.

| Kolom | Keterangan |
|---|---|
| `Provinsi` | Nama provinsi |
| `pdrb` | Produk Domestik Regional Bruto |
| `china_investment_us_ribu` | Nilai investasi Tiongkok dalam ribuan USD |
| `micro_small_industries` | Jumlah industri mikro dan kecil |

### 5.3 `HS2022_to_SITC_Conversion.xlsx`

Tabel konversi HS 2022 ke SITC Rev. 4.

| Kolom | Keterangan |
|---|---|
| `From HS 2022` | Kode HS 2022 |
| `To SITC Rev. 4` | Kode SITC Rev. 4 |

Jika satu kode HS6 memiliki lebih dari satu SITC, pipeline berhenti agar nilai impor tidak dihitung ganda.

### 5.4 `port_geolocation.csv`

Sumber resmi provinsi dan koordinat pelabuhan.

| Kolom | Keterangan |
|---|---|
| `port` | Nama pelabuhan, harus unik |
| `Provinsi` | Nama provinsi |
| `latitude` | Koordinat lintang |
| `longitude` | Koordinat bujur |

Pipeline menolak data jika terdapat pelabuhan duplikat, nilai kosong, atau koordinat di luar rentang.

### 5.5 Template dan Data Referensi

Untuk pengguna yang belum memiliki berkas input, setiap komponen upload pada
halaman **Input Data** menyediakan tombol unduh berikon di bawah area upload.
Berkas dibentuk saat runtime dari folder `dataset_input/` (di-cache dengan
`st.cache_data`), sehingga isinya selalu mengikuti dataset yang dipakai aplikasi.

| Berkas | Label di halaman | Isi unduhan |
|---|---|---|
| `port_imports.csv` | Unduh Template | Header + 5 baris contoh dengan pelabuhan berbeda (format `hs_code` dan `usd_value` asli) |
| `provinces_data.csv` | Unduh Data Referensi | Data referensi lengkap 38 provinsi |
| `HS2022_to_SITC_Conversion.xlsx` | Unduh Tabel Konversi | Berkas konversi asli utuh (lembar HS22–SITC4 dan korelasi) |
| `port_geolocation.csv` | Unduh Data Referensi | Data referensi lengkap 197 pelabuhan |

Catatan:

- `port_imports.csv` sengaja diringkas menjadi sampel karena berkas aslinya ±12 MB.
- Tidak ada pemetaan HS–SITC fiktif; berkas konversi yang diunduh adalah berkas asli.
- Bila `dataset_input/` tidak tersedia (mis. deployment tanpa data), tombol tetap
  berfungsi dan mengunduh header kolom wajib saja sebagai kerangka kosong.
- Tombol unduh bersifat independen, tidak memicu ulang pipeline, dan tidak
  mengubah proses upload, validasi, maupun prapemrosesan yang sudah ada.

---

## 6. Struktur Direktori dan Manfaatnya

```text
China-Imports-Port-Clustering-App/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
├── pages/
│   ├── 1_Beranda.py
│   ├── 2_Input_Data.py
│   └── 3_Dashboard.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── narrative.py
│   └── visualization.py
└── dataset_input/
    ├── port_imports.csv
    ├── provinces_data.csv
    ├── HS2022_to_SITC_Conversion.xlsx
    └── port_geolocation.csv
```

| Bagian | Manfaat |
|---|---|
| `README.md` | Dokumentasi utama mengenai tujuan, halaman, struktur, alur kode, dan cara menjalankan aplikasi. |
| `app.py` | Entry point aplikasi; mengatur konfigurasi halaman, session default, CSS global, dan registrasi tiga halaman Streamlit. |
| `pages/1_Beranda.py` | Memberikan konteks penelitian dan navigasi awal tanpa menjalankan komputasi. |
| `pages/2_Input_Data.py` | Menangani upload, validasi, konfigurasi K, eksekusi pipeline, dan ringkasan hasil. |
| `pages/3_Dashboard.py` | Menyajikan hasil analisis melalui lima tampilan dan mengatur interaksi visual. |
| `src/config.py` | Menyimpan konstanta bersama seperti rentang K, K default, rentang pencarian K optimal (2–6), ambang analisis naratif, dan deskripsi kode SITC. |
| `src/__init__.py` | Menandai `src` sebagai package Python sehingga modul dapat diimpor oleh halaman Streamlit. |
| `src/preprocessing.py` | Mengubah data mentah menjadi dataset siap model dengan validasi yang ketat. |
| `src/modeling.py` | Menjalankan K-Prototypes, evaluasi, surrogate Random Forest, dan SHAP. |
| `src/narrative.py` | Menghitung statistik per klaster dan membentuk narasi deskriptif numerik, SITC, dan geografis dari hasil modeling yang tersimpan. |
| `src/visualization.py` | Memisahkan pembuatan chart dari halaman Streamlit agar kode visualisasi dapat digunakan ulang, termasuk donut SITC, peta statis PNG, dan pemetaan warna kategori SITC. |
| `dataset_input/` | Menyimpan contoh data masukan untuk pengujian lokal. |
| `.streamlit/config.toml` | Mengatur tema, mode headless, dan menyembunyikan navigasi multipage bawaan Streamlit. |
| `requirements.txt` | Mencatat dependensi Python yang dibutuhkan. |

---

## 7. Alur Kerja Kode

### 7.1 Startup: `app.py`

Saat aplikasi dijalankan:

```bash
streamlit run app.py
```

Urutan yang terjadi:

1. `st.set_page_config()` mengatur judul, ikon, layout lebar, dan sidebar.
2. `_DEFAULTS` menginisialisasi session state:
   - data upload;
   - status validasi;
   - jumlah klaster;
   - hasil analisis;
   - error pemrosesan;
   - tampilan dashboard.
3. `BASE_CSS` diterapkan untuk tampilan global.
4. `st.Page()` mendaftarkan:
   - Beranda sebagai halaman default;
   - Input Data;
   - Dashboard.
5. `st.navigation(..., position="hidden")` menjalankan halaman yang dipilih.

Alur navigasi utama:

```text
Beranda
   ↓ tombol Mulai Analisis
Input Data
   ↓ upload + konfigurasi K + proses
Dashboard
   ├── Ringkasan
   ├── Profil Klaster
   ├── Evaluasi
   ├── Peta
   └── Data
```

### 7.2 Input Data: upload dan validasi

Fungsi utama di `pages/2_Input_Data.py`:

- `_store_upload()` — membaca berkas, menghitung hash isi, menyimpan DataFrame, dan menjalankan validasi kolom.
- `read_table()` — membaca CSV/XLSX menjadi DataFrame.
- `validate_columns()` — memeriksa kolom wajib secara case-insensitive.
- `align_columns()` — menyeragamkan nama kolom tanpa mengubah isi data.
- `_file_status_row()` — menampilkan status validasi berkas.
- `_current_signature()` — membuat signature data + K untuk mendeteksi hasil yang sudah kedaluwarsa.
- `_reset_input_data()` — membersihkan session data.
- `_render_summary()` — menampilkan ringkasan setelah pipeline selesai.

### 7.3 Menjalankan pipeline

Saat tombol **Proses Data dan Jalankan Analisis** ditekan:

```text
pages/2_Input_Data.py
   ↓
run_preprocessing()
   ↓
run_modeling()
   ↓
st.session_state["analysis_result"]
```

`run_preprocessing()` menerima empat DataFrame dan mengembalikan:

- `df_imports`;
- `df_sitc`;
- `df_aggregated`;
- `df_finished`;
- `df_final`;
- `report`.

`run_modeling()` menerima `df_final` dan `n_clusters`, lalu mengembalikan:

- `clustered` — dataset final dengan kolom `Cluster`;
- `model`;
- `X_model`;
- `elbow`;
- `silhouette`;
- `surrogate`;
- `shap`;
- `config`.

### 7.4 Dashboard: dispatch tampilan

`pages/3_Dashboard.py` menjalankan:

1. `_render_sidebar()` — memilih view melalui `st.radio`.
2. `render()` — menampilkan header, mengambil `analysis_result`, dan memilih renderer:

```text
Ringkasan        → _render_summary()
Profil Klaster   → _render_profile()
Evaluasi         → _render_evaluation()
Peta             → _render_map()
Data             → _render_data()
```

Jika belum ada hasil analisis, `_no_results()` menampilkan arahan kembali ke Input Data.

### 7.5 Profil Klaster

```text
_render_profile()
├── _render_profile_statistics()
│   ├── _render_cluster_overview()
│   ├── _render_numeric_stats_table()
│   ├── _render_categorical_modes()
│   ├── _render_sitc_distribution()
│   ├── _render_sitc_entropy() → sitc_entropy_bars()
│   ├── _render_mean_feature_section()
│   ├── _render_distribution_section() → cluster_boxplots / cluster_violinplots
│   └── _render_zscore_section()
├── _render_cluster_scatters() → cluster_scatter()
└── _render_shap_section()
    ├── global_shap_bar()
    ├── shap_cluster_heatmap()
    └── shap_beeswarm_figure()
```

### 7.6 Evaluasi

```text
_render_evaluation()
├── _render_kprototypes_evaluation()
│   ├── _render_elbow_section()
│   ├── _render_silhouette_section()
│   │   ├── _render_silhouette_plot()
│   │   ├── _render_silhouette_table()
│   │   └── _render_silhouette_downloads()
│   └── _render_kprototypes_config()
└── _render_surrogate_evaluation()
    ├── _render_surrogate_metrics()
    ├── _render_surrogate_report()
    └── _render_surrogate_confusion()
```

### 7.7 Peta

```text
_render_map()
├── _find_coordinate_columns()
├── filter data berdasarkan SITC
├── geo_scatter()
└── tampilkan collision table jika ada koordinat identik
```

### 7.8 Data

```text
_render_data()
├── _dataset_catalog()
├── ringkasan preprocessing
├── preview dataset
├── download CSV per tahap
└── preview + download subset klaster
```

### 7.9 Narasi Karakteristik Klaster

```text
generate_all_cluster_narratives(model_result)
├── analyze_shap_importance()
│   └── check_shap_validity() → fidelity, recall kelas, jumlah observasi uji
├── analyze_numeric_profile()      → mean, Z-score, rata-rata tertinggi, SHAP share
├── analyze_categorical_profile()  → modus, persentase, normalized entropy, SHAP share SITC
├── analyze_geographic_profile()   → pelabuhan unik per provinsi dan kawasan kepulauan
└── generate_cluster_narrative()
    ├── generate_numeric_narrative()
    ├── generate_categorical_narrative()
    └── generate_geographic_narrative()
```

Seluruh input dibaca dari `analysis_result` yang sudah tersimpan di session state.
Modul ini tidak memanggil `run_preprocessing()`, `run_modeling()`, `train_surrogate()`,
atau `compute_shap()` sehingga perpindahan tampilan Dashboard tidak memicu ulang
pipeline. Statistik numerik dan kategorikal memakai dataset final setelah IQR
capping dalam satuan asli, bukan `X_model` yang sudah distandardisasi. Analisis
SHAP memakai `shap_by_class` dan `y_test` yang sudah dihitung modeling: untuk
klaster *k* hanya baris uji dengan label aktual *k* yang dipakai, dengan
kontribusi dari kelas *k* dan seluruh kolom one-hot SITC dijumlahkan sebagai satu
kelompok fitur.

Integrasi visualisasi di Dashboard Ringkasan:

```text
_render_cluster_narratives()
├── _numeric_heading_html()                → judul fitur + badge Z-score
├── _render_sitc_donut_section()          → donut + panel legenda responsif
│   ├── sitc_distribution_donut()         → donut komposisi SITC per klaster
│   └── _sitc_legend_html()               → legenda kanan; turun ke bawah di layar sempit
└── _render_cluster_geographic_map()      → hanya saat kartu klaster dibuka
    ├── select_cluster_ports()            → seluruh pelabuhan unik klaster
    ├── compute_map_extent()              → extent adaptif minimum 350 km
    ├── select_map_rows()                 → filter Cluster × SITC, buang marker duplikat
    ├── st.radio(key="narrative_map_sitc_<k>") → default kategori modus klaster
    └── _map_png_cached() → static_geo_map_png() → st.image()
```

Kartu klaster memakai expander stateful (`on_change="rerun"`) sehingga isi hanya
diproses saat kartu dibuka. Donut memakai dua kolom Streamlit: chart di kiri dan
panel legenda HTML di kanan. Di bawah breakpoint layar sempit, Streamlit
menumpuk kolom sehingga legenda berpindah ke bawah chart; Plotly tidak lagi
memakai legenda internal dengan margin kanan tetap. PNG peta dibuat secara lazy:
tidak ada gambar yang dibuat saat Dashboard pertama kali dimuat, dan seluruh
kombinasi Cluster × SITC tidak pernah dibuat sekaligus. Cache `st.cache_data` memakai baris hasil filter
(signature dataset), ID klaster, kategori SITC, warna identitas klaster,
geographic extent, serta lebar/tinggi/scale gambar sebagai key. Mengubah radio
button hanya memicu rerun UI; preprocessing, K-Prototypes, surrogate, dan SHAP
tidak dijalankan ulang.

`compute_map_extent()` menghitung bounding box dari **seluruh pelabuhan unik
klaster**, bukan hanya SITC terpilih. Pusat peta adalah titik tengah bounding
box; setiap sumbu minimal 350 km (`MAP_MIN_EXTENT_KM` di `src/config.py`).
Persebaran yang lebih luas mendapat padding 12% (`MAP_EXTENT_PADDING`).
Konversi kilometer ke derajat longitude memakai kosinus latitude pusat, dan
kondisi koordinat identik tetap memperoleh rentang minimum yang valid. Figure
memakai `fitbounds=False` dengan `center`, `lonaxis_range`, dan `lataxis_range`
eksplisit, sehingga zoom PNG terbatas dan tidak berubah saat kategori SITC
diganti.

---

## 8. Alur Preprocessing

Modul: `src/preprocessing.py`

```text
run_preprocessing()
├── build_hs_sitc_mapping()
│   ├── validate_columns()
│   ├── align_columns()
│   ├── ekstraksi HS6
│   └── deteksi pemetaan ambigu
├── convert_hs_to_sitc()
│   ├── extract_hs6()
│   └── merge transaksi dengan mapping
├── aggregate_sitc_first_digit()
│   └── agregasi nilai impor per port × SITC
├── merge_provinces()
│   ├── prepare_geolocation()
│   ├── merge port dengan provinsi/koordinat
│   └── merge indikator makro provinsi
├── clean_and_impute()
│   ├── hapus duplikasi
│   ├── normalisasi teks
│   └── imputasi median
├── apply_iqr_capping()
│   ├── usd_value per SITC
│   └── fitur numerik lain secara global
└── validasi kolom final
```

Aturan penting:

- `usd_value` di-cap per kelompok SITC.
- `pdrb`, `china_investment_us_ribu`, dan `micro_small_industries` di-cap global.
- IQR multiplier = 3.
- Nama kolom asli tidak diubah oleh proses tampilan; label visual hanya diterapkan di dashboard.

---

## 9. Alur Modeling

Modul: `src/modeling.py`

```text
run_modeling()
├── prepare_model_matrix()
│   ├── ambil fitur kategorikal + numerik
│   ├── konversi numerik
│   ├── StandardScaler
│   └── bangun array object
├── compute_elbow()
│   └── K-Prototypes untuk K = 2–8
├── fit_final_kprototypes()
│   └── model final dengan K terpilih
├── kprototypes_silhouette()
│   └── silhouette campuran
├── train_surrogate()
│   ├── one-hot SITC
│   ├── stratified train/test split
│   └── Random Forest 500 trees
├── compute_shap()
│   ├── TreeExplainer
│   ├── SHAP seluruh data uji per kelas
│   └── grouping ke fitur asli
└── gabungkan label klaster ke dataset
```

### Pencarian K optimal

Fungsi terpisah, dipanggil dari tombol **Cari K Optimal (Silhouette)** di halaman Input Data:

```text
find_optimal_k_by_silhouette(df_final, k_values=OPTIMAL_K_RANGE)
├── prepare_model_matrix()
├── untuk setiap K = 2–6:
│   ├── fit_final_kprototypes()
│   └── kprototypes_silhouette()
├── lewati K yang gagal inisialisasi (silhouette = NaN)
└── pilih silhouette rata-rata tertinggi (seri → K terkecil)
```

Mengembalikan `best_k`, `best_silhouette`, `scores`, dan `skipped_k`. Fungsi ini hanya menjalankan K-Prototypes dan silhouette; surrogate Random Forest serta SHAP tidak dilatih.

### Parameter utama

| Parameter | Nilai |
|---|---|
| Rentang K | 2–8 |
| K default | 4 |
| Rentang pencarian K optimal | 2–6 (`OPTIMAL_K_RANGE`) |
| K-Prototypes init | `Huang` |
| `n_init` final | 5 |
| `n_init` Elbow | 3 |
| `random_state` | 42 |
| `n_jobs` | -1 |
| Silhouette sample | 2500 |
| Random Forest trees | 500 |
| Test size surrogate | 20% |
| Stratified split | Ya |

### Catatan interpretasi

- K-Prototypes tidak menghasilkan SHAP.
- SHAP dihitung dari model surrogate Random Forest yang meniru label klaster.
- Fidelity surrogate harus diperiksa sebelum menafsirkan SHAP.
- Nilai SHAP bukan hubungan kausal.
- Silhouette campuran menggunakan squared Euclidean pada fitur numerik terstandardisasi ditambah penalti ketidakcocokan kategorikal yang dikali gamma.

---

## 10. Visualisasi dan Label Tampilan

Modul: `src/visualization.py`

### Chart yang tersedia

- `cluster_size_bar()` — ukuran klaster.
- `cluster_pie()` — proporsi klaster.
- `numerical_mean_bars()` — rata-rata fitur numerik.
- `cluster_boxplots()` — boxplot kuartil.
- `cluster_violinplots()` — violinplot kepadatan.
- `sitc_stacked_bar()` — komposisi SITC.
- `sitc_entropy_bars()` — normalized entropy kategori SITC per klaster.
- `sitc_distribution_donut()` — donut komposisi kategori SITC per klaster; legenda dirender Dashboard sebagai panel responsif di kanan chart.
- `cluster_z_heatmap()` — Z-score profil klaster.
- `elbow_line()` — Elbow curve.
- `global_shap_bar()` — global SHAP importance.
- `shap_cluster_heatmap()` — SHAP per kelas klaster.
- `shap_beeswarm_figure()` — beeswarm SHAP.
- `confusion_heatmap()` — confusion matrix surrogate.
- `geo_scatter()` — peta sebaran pelabuhan.
- `select_map_rows()` — filter baris peta Cluster × SITC tanpa marker duplikat.
- `select_cluster_ports()` — seluruh pelabuhan unik satu klaster untuk dasar extent.
- `compute_map_extent()` — extent adaptif minimum 350 km dengan padding 12%.
- `static_geo_map_png()` — ekspor peta pelabuhan satu klaster menjadi PNG statis.
- `cluster_scatter()` — scatter nilai aktual.
- `silhouette_plot()` — silhouette per observasi.
- `kprototypes_silhouette_samples()` — nilai silhouette per observasi.

### Label nilai aktual

Untuk chart yang menampilkan nilai aktual:

- `Nilai Impor (USD)`
- `SITC`
- `PDRB (Miliar rupiah)`
- `Nilai Investasi Tiongkok (Ribuan USD)`
- `Jumlah Industri Mikro Kecil`

### Label non-nilai-aktual

Untuk SHAP dan Z-score:

- `Nilai Impor`
- `SITC`
- `PDRB`
- `Nilai Investasi Tiongkok`
- `Jumlah Industri Mikro Kecil`

Label ini hanya mengubah teks tampilan. Nama kolom asli pada DataFrame dan proses modeling/preprocessing tetap sama.

### Peta

Peta menggunakan Plotly `scatter_geo` dengan cakupan Asia. Kategori SITC dipilih satu per satu agar marker tidak saling menumpuk. Caption peta menyertakan daftar arti kode SITC 0–9.

### Aturan analisis naratif dan threshold

Seluruh ambang berikut tersimpan di `src/config.py` agar narasi dan pengujian memakai sumber yang sama.

**Perbandingan numerik (Z-score rata-rata klaster terhadap dataset final):**

| Z-score | Interpretasi narasi |
|---|---|
| `Z ≥ 1,0` | Jauh di atas rata-rata keseluruhan |
| `0,5 ≤ Z < 1,0` | Di atas rata-rata keseluruhan |
| `-0,5 < Z < 0,5` | Mendekati rata-rata keseluruhan |
| `-1,0 < Z ≤ -0,5` | Di bawah rata-rata keseluruhan |
| `Z ≤ -1,0` | Jauh di bawah rata-rata keseluruhan |

Rata-rata tertinggi antar klaster diperiksa terpisah dari Z-score dengan toleransi floating-point. Jika standar deviasi global nol, interpretasi perbandingan dilewati.

**Kategorikal SITC (normalized Shannon entropy):**

| Entropy | Interpretasi |
|---|---|
| `0 ≤ H ≤ 0,50` | Terkonsentrasi pada sedikit kategori |
| `0,50 < H < 0,80` | Moderat; tidak ada kalimat entropy khusus |
| `0,80 ≤ H ≤ 1,00` | Relatif merata |

Narasi memakai `m` yang sama dengan chart normalized entropy pada Dashboard, yaitu jumlah kategori SITC unik di **seluruh dataset**. Dengan begitu nilai entropy pada narasi dan chart selalu identik, dan klaster dengan dua kategori tidak otomatis bernilai 1,0 apabila dataset memiliki lebih banyak kategori. Jika hanya ada satu kategori (`m = 1`), entropy ditetapkan 0.

**Interpretasi SHAP (model surrogate):**

| Ambang | Nilai | Fungsi |
|---|---|---|
| `SURROGATE_FIDELITY_MIN` | `0,80` | Fidelity holdout minimum; di bawahnya semua klaim SHAP dibatalkan |
| `SURROGATE_CLASS_RECALL_MIN` | `0,70` | Recall kelas minimum per klaster |
| `SHAP_MIN_TEST_SAMPLES` | `10` | Jumlah observasi uji minimum per klaster |
| `SHAP_IMPORTANCE_THRESHOLD` | `25%` | Share mean absolute SHAP minimum agar fitur disebut pembeda kuat |

Share dihitung dari lima kelompok fitur: empat fitur numerik dan satu kelompok SITC hasil penjumlahan kontribusi absolut seluruh kolom one-hot. Mean absolute SHAP mengukur kekuatan kontribusi, bukan arah pengaruh.

**Geografis:**

| Ambang | Nilai | Fungsi |
|---|---|---|
| `GEO_DOMINANCE_THRESHOLD` | `80%` | Satu provinsi/kawasan dianggap dominan |
| `GEO_COVERAGE_MIN` | `80%` | Cakupan provinsi minimum sebelum klaim persebaran diperkuat |
| `NARRATIVE_TOP_PROVINCES` | `3` | Maksimum provinsi yang disebut |
| `NARRATIVE_TOP_REGIONS` | `3` | Maksimum kawasan kepulauan yang disebut |

Jumlah pelabuhan dan persebaran geografis dihitung dari pelabuhan unik per klaster; satu pelabuhan tidak dihitung berulang meskipun memiliki beberapa kategori SITC. Pemetaan 38 provinsi ke tujuh kawasan kepulauan (Sumatera, Jawa, Kalimantan, Sulawesi, Bali dan Nusa Tenggara, Maluku, Papua) berada di `src/narrative.py`.

---

## 11. Output dan Unduhan

### Output analisis

- Dataset final dengan kolom `Cluster`.
- Model K-Prototypes.
- Matriks fitur terstandardisasi.
- Elbow table.
- Silhouette score dan jumlah sampel.
- Model surrogate dan metrik.
- Classification report.
- Confusion matrix.
- Hasil SHAP per kelas.
- Narasi deskriptif karakteristik klaster (numerik, SITC, geografis, dan identitas utama).
- Dataset per tahap preprocessing.

### Unduhan CSV

- Nilai silhouette seluruh data.
- Daftar observasi dengan silhouette negatif.
- Dataset input mentah.
- Hasil konversi HS–SITC.
- Hasil agregasi pelabuhan × SITC.
- Data setelah penggabungan provinsi.
- Dataset final preprocessing.
- Dataset berlabel klaster.
- Subset satu klaster.

Semua CSV menggunakan encoding `utf-8-sig` agar kompatibel dengan Excel.

---

## 12. Menjalankan Aplikasi

### Prasyarat

- Python 3.10 atau lebih baru.
- Koneksi internet tidak diperlukan untuk proses modeling, tetapi halaman menggunakan font eksternal.
- File input tersedia.
- Untuk peta statis pada analisis deskriptif: `kaleido` (tercantum di `requirements.txt`) dan Chrome/Chromium.

### Instalasi

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
plotly_get_chrome -y             # sekali saja: unduh Chrome untuk ekspor PNG Kaleido
```

`plotly_get_chrome` hanya diperlukan bila sistem belum memiliki Chrome/Chromium.
Jika browser tersedia, Kaleido memakainya langsung. Tanpa browser, ekspor PNG
gagal dan bagian peta menampilkan pesan; narasi dan visualisasi lain tetap berjalan.

### Menjalankan

```bash
streamlit run app.py
```

Aplikasi akan terbuka pada browser dan dimulai dari halaman Beranda.

### Penggunaan

1. Buka **Input Data**.
2. Unggah empat berkas wajib.
3. Pastikan seluruh status berkas valid.
4. Pilih K:
   - manual pada rentang 2–8;
   - atau klik **Cari K Optimal (Silhouette)** untuk menguji K = 2–6 dan memakai K dengan silhouette tertinggi.
5. Klik **Proses Data dan Jalankan Analisis**.
6. Tunggu preprocessing, K-Prototypes, evaluasi, surrogate, dan SHAP selesai.
7. Klik **Lihat Hasil Analisis**.
8. Jelajahi lima tampilan Dashboard.
9. Unduh dataset atau hasil silhouette dari tampilan Data/Evaluasi.

---

## 13. Konfigurasi Saat Ini

| Konfigurasi | Nilai |
|---|---|
| K minimum | 2 |
| K maksimum | 8 |
| K default | 4 |
| Rentang pencarian K optimal | 2–6 |
| Random state | 42 |
| Silhouette sample | 2500 |
| IQR multiplier | 3 |
| Surrogate test size | 20% |
| Random Forest estimators | 500 |
| Minimum map extent | 350 km (`MAP_MIN_EXTENT_KM`) |
| Map extent padding | 12% (`MAP_EXTENT_PADDING`) |

Konfigurasi K berada di `src/config.py`. Konstanta fitur dan skema berada di `src/preprocessing.py`. Konstanta extent peta statis juga berada di `src/config.py` agar dapat disesuaikan setelah pemeriksaan visual.

---

## 14. Batasan dan Catatan Implementasi

- Aplikasi dirancang untuk **single-user** dan penyimpanan state di session Streamlit.
- Hasil analisis berada di memori sesi; refresh atau sesi baru memerlukan upload dan proses ulang.
- Tidak ada database, autentikasi, atau penyimpanan permanen.
- Model surrogate adalah model pendekatan, bukan model klaster utama.
- SHAP menjelaskan perilaku surrogate, bukan hubungan kausal.
- Narasi karakteristik klaster dibentuk dari dataset final setelah IQR capping dalam satuan asli, bukan dari `X_model` yang sudah distandardisasi.
- Klaim SHAP pada narasi hanya muncul jika fidelity holdout ≥ 0,80, recall kelas ≥ 0,70, dan tersedia minimal 10 observasi uji untuk klaster tersebut. Jika salah satu syarat gagal, statistik numerik, SITC, dan geografis tetap ditampilkan tanpa klaim SHAP.
- Mean absolute SHAP mengukur kekuatan kontribusi fitur, bukan arah pengaruh; narasi tidak menyimpulkan hubungan kausal atau menyatakan bahwa suatu nilai fitur meningkatkan peluang masuk klaster.
- Persebaran geografis dihitung dari pelabuhan unik per klaster. Jika cakupan provinsi di bawah 80%, narasi menyertakan keterangan cakupan dan tidak menghasilkan klaim dominasi yang kuat.
- Pemetaan provinsi ke kawasan kepulauan bersifat tetap untuk 38 provinsi; provinsi baru atau nama di luar daftar tidak dipetakan dan tidak ditebak.
- Peta pada analisis deskriptif (Ringkasan) adalah **PNG statis**: tidak dapat di-zoom, di-pan, di-hover, dan tanpa toolbar Plotly. Halaman **Peta** Dashboard tetap memakai Plotly interaktif seperti sebelumnya.
- Ekspor PNG memerlukan Kaleido dan Chrome/Chromium. Jika tidak tersedia, bagian peta menampilkan pesan dan narasi tetap berjalan.
- PNG peta dibuat lazy saat kartu klaster dibuka dan disimpan di cache berdasarkan data hasil filter, ID klaster, kategori SITC, warna identitas klaster, geographic extent, serta resolusi gambar.
- Geographic extent peta dihitung dari seluruh pelabuhan unik klaster dengan lebar/tinggi minimum 350 km dan padding 12% untuk persebaran yang lebih luas; pusat peta adalah titik tengah bounding box. Karena extent tidak bergantung pada SITC terpilih, mengganti kategori tidak mengubah zoom maupun posisi peta.
- Donut SITC memakai palet kualitatif terpisah dari `cluster_color_map`; warna klaster tetap dipakai untuk aksen kartu dan marker peta.
- Badge Z-score pada bagian A memakai warna kategori sendiri (hijau, abu, kuning, merah lembut), terpisah dari `cluster_color_map`, dan hanya meringkas klasifikasi Z-score yang sama dengan kalimat narasi.
- Kontrol skala log pada tab Profil Klaster saat ini dinonaktifkan dan kodenya dipertahankan sebagai komentar.
- Peta menggunakan `scatter_geo`, bukan tile Carto.
- Tampilan Data menampilkan nama kolom asli karena berfungsi sebagai ekspor dataset, bukan label presentasi.
- Jumlah fitur numerik modeling saat ini tetap empat; validasi tambahan jumlah fitur tidak diaktifkan.

---

## 15. Troubleshooting

### Berkas tidak valid

Periksa:

- nama kolom;
- pemisah CSV;
- encoding;
- baris kosong;
- apakah file benar-benar memiliki header.

### HS6 dipetakan ke beberapa SITC

Pipeline berhenti untuk mencegah duplikasi nilai. Periksa tabel konversi dan pastikan satu HS6 hanya memiliki satu SITC.

### Pelabuhan tidak ditemukan pada geolokasi

Tambahkan nama pelabuhan yang sama persis ke `port_geolocation.csv`. Nama pelabuhan dibandingkan secara case-sensitive setelah trim.

### Provinsi tidak ditemukan

Pastikan nama provinsi pada geolokasi dan `provinces_data.csv` konsisten. Nama provinsi dinormalisasi dengan `clean_province()`.

### Silhouette tidak tersedia

Silhouette membutuhkan minimal dua klaster dan data yang valid. Periksa jumlah observasi dan variasi fitur.

### SHAP tidak tersedia

SHAP dapat gagal jika paket tidak tersedia, model surrogate tidak dapat dilatih, atau data terlalu kecil. Pesan alasan ditampilkan pada tab Analisis SHAP.

### Peta tidak tampil

Pastikan dataset memiliki kolom latitude dan longitude yang valid. Peta juga membutuhkan kolom `sitc`.

---

## Ringkasan Alur Singkat

```text
app.py
  → Beranda
  → Input Data
      → validasi empat berkas
      → run_preprocessing()
      → run_modeling()
      → analysis_result
  → Dashboard
      → Ringkasan
      → Profil Klaster
      → Evaluasi
      → Peta
      → Data
```

Dengan struktur ini, `app.py` berfungsi sebagai pengatur navigasi dan konfigurasi global, `pages/` menangani interaksi Streamlit, sedangkan `src/` menangani preprocessing, modeling, evaluasi, dan pembuatan visualisasi secara terpisah.
