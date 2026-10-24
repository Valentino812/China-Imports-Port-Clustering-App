"""Dashboard page with five SPA-like internal views.

The five views (Ringkasan, Profil Klaster, Evaluasi, Peta, Data) are switched via
``st.session_state`` so this file remains a single Streamlit page. The header and
sidebar render once and stay consistent across view changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

from src.modeling import SILHOUETTE_SAMPLE_SIZE
from src.visualization import (
    cluster_boxplots,
    cluster_numeric_stats,
    cluster_pie,
    cluster_scatter,
    cluster_size_bar,
    cluster_violinplots,
    cluster_z_heatmap,
    confusion_heatmap,
    elbow_line,
    global_shap_bar,
    kprototypes_silhouette_samples,
    numerical_mean_bars,
    shap_beeswarm_figure,
    shap_cluster_heatmap,
    silhouette_plot,
    sitc_stacked_bar,
)

VIEWS = ["Ringkasan", "Profil Klaster", "Evaluasi", "Peta", "Data"]

VIEW_TITLES = {
    "Ringkasan": (
        "Ringkasan Hasil Klasterisasi",
        "Tinjauan menyeluruh atas hasil pengelompokan kombinasi pelabuhan dan kategori komoditas SITC, "
        "mencakup jumlah observasi, komposisi klaster, serta proporsi distribusi tiap kelompok.",
    ),
    "Profil Klaster": (
        "Karakteristik & Profil Klaster",
        "Analisis mendalam karakteristik tiap klaster melalui statistik deskriptif fitur numerik, "
        "modus fitur kategorikal, distribusi kelompok komoditas SITC, sebaran wilayah provinsi, "
        "dan interpretasi berbasis nilai SHAP dari model surrogate Random Forest.",
    ),
    "Evaluasi": (
        "Evaluasi Kualitas Model",
        "Penilaian kualitas klasterisasi menggunakan metode Elbow (nilai cost K-Prototypes), "
        "skor Silhouette campuran untuk mengukur kohesi dan separasi klaster, serta evaluasi "
        "performa model surrogate Random Forest sebagai landasan validitas analisis SHAP.",
    ),
    "Peta": (
        "Peta Distribusi Spasial",
        "Visualisasi interaktif persebaran geografis klaster pelabuhan di seluruh wilayah Indonesia, "
        "difilter berdasarkan kategori komoditas SITC untuk menghindari tumpang tindih titik lokasi.",
    ),
    "Data": (
        "Eksplorasi & Ekspor Data",
        "Pratinjau dan ekspor dataset pada setiap tahapan pipeline analisis — mulai dari data impor "
        "mentah, hasil konversi HS–SITC, agregasi, penggabungan provinsi, hingga dataset final berlabel klaster.",
    ),
}

# Nama kelompok SITC Rev. 4 berdasarkan digit pertama kode SITC.
SITC_LABELS = {
    "0": "Bahan makanan dan binatang hidup",
    "1": "Minuman dan tembakau",
    "2": "Bahan mentah, tidak dapat dimakan, kecuali bahan bakar",
    "3": "Bahan bakar mineral, pelumas, dan bahan terkait",
    "4": "Minyak, lemak, dan lilin hewani serta nabati",
    "5": "Bahan kimia dan produk terkait",
    "6": "Barang manufaktur berbasis bahan baku",
    "7": "Mesin dan peralatan transportasi",
    "8": "Barang manufaktur jadi",
    "9": "Komoditas dan transaksi tidak diklasifikasikan di tempat lain",
}


def _sitc_label(value: str) -> str:
    label = SITC_LABELS.get(str(value))
    return f"SITC {value}: {label}" if label else f"SITC {value}"


DASHBOARD_CSS = r"""
<style>
/* ─── Dashboard: unhide sidebar only while this page is rendered ────── */
body:has(.dashboard-page-marker) [data-testid='stSidebar'] {
    display: block !important;
    background: rgba(255,255,255,0.95) !important;
    border-right: 1px solid rgba(226,229,241,0.8) !important;
    min-width: 265px;
    box-shadow: 2px 0 12px rgba(79,107,237,0.06);
}
body:has(.dashboard-page-marker) [data-testid='stSidebar'] > div:first-child {
    padding-top: 1.25rem;
}

/* Streamlit renders the expand button inside the header only while the
   sidebar is collapsed. Keep that button reachable as a floating control. */
body:has(.dashboard-page-marker) header[data-testid='stHeader'] {
    display: block !important;
    position: absolute !important;
    height: 0 !important;
    min-height: 0 !important;
    background: transparent !important;
    pointer-events: none !important;
}
body:has(.dashboard-page-marker) [data-testid='stToolbar'] {
    display: flex !important;
    position: fixed !important;
    left: 1rem;
    bottom: 1rem;
    width: auto !important;
    height: auto !important;
    padding: 0 !important;
    background: transparent !important;
    pointer-events: auto !important;
    z-index: 1000;
}
body:has(.dashboard-page-marker) [data-testid='stToolbar'] > div > div:not(:first-child) {
    display: none !important;
}
body:has(.dashboard-page-marker) [data-testid='stExpandSidebarButton'] {
    background: #FFFFFF !important;
    border: 1px solid rgba(226,229,241,0.9) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    box-shadow: 0 2px 8px rgba(79,107,237,0.12) !important;
}

/* Sidebar text */
[data-testid='stSidebar'] [data-testid='stMarkdownContainer'] p,
[data-testid='stSidebar'] label,
[data-testid='stSidebar'] label span,
[data-testid='stSidebar'] [role='radiogroup'] p,
[data-testid='stSidebar'] [data-testid='stWidgetLabel'] {
    color: #1A1C2E !important;
    font-size: 14px !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    line-height: 1.5 !important;
}
[data-testid='stSidebar'] [role='radiogroup'] label {
    padding: 7px 10px;
    border-radius: 8px;
    font-weight: 500 !important;
}

/* Sidebar brand block */
.side-brand {
    padding: 6px 4px 18px 4px;
    border-bottom: 1px solid rgba(226,229,241,0.9);
    margin-bottom: 16px;
}
.side-brand-title {
    font-size: 14px;
    font-weight: 700;
    color: #1A1C2E;
    line-height: 1.5;
    font-family: 'Inter', system-ui, sans-serif;
}

/* Sidebar section label */
.side-section-label {
    font-size: 10.5px;
    letter-spacing: 0.12em;
    font-weight: 700;
    color: #6B7280;
    text-transform: uppercase;
    margin: 16px 0 8px 2px;
    font-family: 'Inter', system-ui, sans-serif;
}

/* Sidebar footer */
.sidebar-footer-caption {
    margin-top: 12px;
    text-align: center;
    font-size: 11.5px;
    color: #9CA3AF;
    font-family: 'Inter', system-ui, sans-serif;
    line-height: 1.5;
}
</style>
"""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _cluster_structure(clustered: pd.DataFrame) -> pd.DataFrame:
    struct = (
        clustered.groupby("Cluster", sort=True)
        .agg(
            jumlah_kombinasi_Port_SITC=("Cluster", "size"),
            jumlah_pelabuhan_unik=("port", "nunique"),
            jumlah_provinsi=("Provinsi", "nunique"),
        )
    )
    struct["persentase_dataset"] = (
        struct["jumlah_kombinasi_Port_SITC"] / len(clustered) * 100
    ).round(2)
    return struct.reset_index()


def _get_mode_text(series: pd.Series) -> str:
    modes = series.mode(dropna=True).astype(str).tolist()
    return ", ".join(modes) if modes else "Tidak tersedia"


@st.cache_data(show_spinner=False)
def _silhouette_samples_cached(
    X_model: pd.DataFrame,
    labels: np.ndarray,
    gamma: float,
    numerical_cols: tuple[str, ...],
    categorical_cols: tuple[str, ...],
    sample_size: int,
    random_state: int,
) -> pd.DataFrame:
    """Cache the per-observation silhouette while the analysis result is stable."""
    return kprototypes_silhouette_samples(
        X_model,
        labels,
        gamma,
        list(numerical_cols),
        list(categorical_cols),
        sample_size=sample_size,
        random_state=random_state,
    )


def _goto_input() -> None:
    st.switch_page("pages/2_Input_Data.py")


def _no_results(view: str) -> None:
    st.info(
        "Belum ada hasil analisis pada sesi ini. Silakan buka halaman **Input Data**, unggah keempat "
        "berkas wajib, lalu jalankan proses pemodelan."
    )
    st.button(
        "Ke Halaman Input Data →",
        type="primary",
        on_click=_goto_input,
    )


# ---------------------------------------------------------------------------
# View: Ringkasan
# ---------------------------------------------------------------------------

def _render_summary(result: dict) -> None:
    clustered = result["modeling"]["clustered"]
    struct = _cluster_structure(clustered)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Observasi", f"{len(clustered):,}")
    c2.metric("Jumlah Pelabuhan", f"{clustered['port'].nunique():,}")
    c3.metric("Cakupan Provinsi", f"{clustered['Provinsi'].nunique():,}")
    c4.metric("Jumlah Klaster (K)", str(result["modeling"]["config"]["n_clusters"]))

    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(
            cluster_size_bar(struct, "jumlah_kombinasi_Port_SITC", "Ukuran Klaster"),
            width="stretch",
        )
    with right:
        st.plotly_chart(cluster_pie(struct), width="stretch")

    st.subheader("Tabel Ukuran dan Proporsi Tiap Klaster")
    display = struct.rename(
        columns={
            "Cluster": "Klaster",
            "jumlah_kombinasi_Port_SITC": "Jumlah Observasi",
            "jumlah_pelabuhan_unik": "Pelabuhan Unik",
            "jumlah_provinsi": "Provinsi",
            "persentase_dataset": "Proporsi (%)",
        }
    )
    st.dataframe(display, width="stretch", hide_index=True)



# ---------------------------------------------------------------------------
# View: Profil Klaster
# ---------------------------------------------------------------------------

def _render_profile(result: dict) -> None:
    clustered = result["modeling"]["clustered"]
    numerical_cols = result["modeling"]["numerical_cols"]
    categorical_cols = result["modeling"]["categorical_cols"]
    struct = _cluster_structure(clustered)
    stats = cluster_numeric_stats(clustered, numerical_cols)

    st.subheader("Distribusi Observasi per Klaster")
    overview = struct[["Cluster", "jumlah_kombinasi_Port_SITC", "persentase_dataset"]].rename(
        columns={
            "Cluster": "Klaster",
            "jumlah_kombinasi_Port_SITC": "Jumlah Observasi",
            "persentase_dataset": "Proporsi (%)",
        }
    )
    st.dataframe(overview, width="stretch", hide_index=True)
    st.caption(
        "Tabel ini merangkum jumlah observasi (kombinasi pelabuhan–SITC) dan proporsinya "
        "terhadap total dataset untuk setiap klaster yang terbentuk."
    )

    st.subheader("Statistik Deskriptif Fitur Numerik per Klaster")
    if stats.empty:
        st.warning("Statistik fitur numerik tidak tersedia untuk dataset ini.")
    else:
        st.dataframe(
            stats[["Klaster", "Fitur", "Mean", "Median", "Q1", "Q3", "IQR"]].round(3),
            width="stretch",
            hide_index=True,
        )
    st.caption(
        "Statistik deskriptif (rata-rata, median, kuartil bawah Q1, kuartil atas Q3, dan rentang "
        "antarkuartil IQR) dihitung untuk setiap fitur numerik pada masing-masing klaster, "
        "menggunakan nilai dalam satuan asli tanpa standardisasi."
    )

    st.subheader("Nilai Modus Fitur Kategorikal per Klaster")
    mode_rows = []
    for cluster_id, group in clustered.groupby("Cluster", sort=True):
        row = {"Klaster": cluster_id}
        for col in categorical_cols:
            row[f"Modus {col.upper()}"] = _get_mode_text(group[col])
        mode_rows.append(row)
    st.dataframe(pd.DataFrame(mode_rows), width="stretch", hide_index=True)
    st.caption(
        "Tabel ini menampilkan nilai yang paling sering muncul (modus) pada setiap fitur kategorikal "
        "untuk masing-masing klaster. Apabila terdapat lebih dari satu nilai dengan frekuensi tertinggi, "
        "seluruh nilai tersebut ditampilkan."
    )

    st.subheader("Distribusi Proporsi Kategori SITC per Klaster")
    sitc_pct = pd.crosstab(clustered["Cluster"], clustered["sitc"], normalize="index").mul(100)
    sitc_pct = sitc_pct.reindex(sorted(sitc_pct.columns), axis=1).round(2)
    sitc_display = sitc_pct.reset_index().rename(columns={"Cluster": "Klaster"})
    st.dataframe(sitc_display, width="stretch", hide_index=True)
    st.caption(
        "Tabel lintas-tabulasi (crosstab) ini menunjukkan proporsi (%) setiap kategori SITC "
        "terhadap total observasi dalam klaster yang bersangkutan."
    )
    st.plotly_chart(sitc_stacked_bar(sitc_pct), width="stretch")
    st.caption(
        "Diagram batang bertumpuk ini memvisualisasikan komposisi kategori SITC di setiap klaster "
        "dalam bentuk proporsi relatif, sehingga perbedaan pola komoditas antara klaster dapat "
        "dibandingkan secara langsung."
    )

    feature_specs = [
        {
            "feature": "usd_value",
            "label": "Nilai Impor (USD)",
            "number_format": ",.2f",
        },
        {
            "feature": "pdrb",
            "label": "PDRB",
            "number_format": ",.2f",
        },
        {
            "feature": "china_investment_us_ribu",
            "label": "Investasi Tiongkok",
            "number_format": ",.2f",
        },
        {
            "feature": "micro_small_industries",
            "label": "Jumlah Industri Mikro dan Kecil",
            "number_format": ",.0f",
        },
    ]

    st.subheader("Perbandingan Rata-Rata Fitur Numerik per Klaster")
    st.plotly_chart(numerical_mean_bars(clustered, numerical_cols), width="stretch")
    st.caption(
        "Diagram batang ini menampilkan nilai rata-rata setiap fitur numerik untuk masing-masing klaster "
        "dalam satuan aslinya. Perbandingan ini memungkinkan identifikasi perbedaan besaran absolut "
        "antarklaster pada setiap fitur."
    )

    st.subheader("Ringkasan Statistik Kuartil Atribut Numerik per Klaster")
    log_usd_box = st.checkbox(
        "Gunakan skala logaritmik untuk Nilai Impor (USD)",
        key="profile_box_log_usd",
        value=False,
    )

    feature_labels = {spec["feature"]: spec["label"] for spec in feature_specs}
    number_formats = {spec["feature"]: spec["number_format"] for spec in feature_specs}
    box_figure = cluster_boxplots(
        clustered,
        stats,
        numerical_cols,
        feature_labels,
        title="Ringkasan Statistik Kuartil Atribut Numerik per Klaster",
        log_usd=log_usd_box,
        number_formats=number_formats,
    )
    st.plotly_chart(box_figure, width="stretch")
    box_caption = (
        "Boxplot menampilkan median, kuartil pertama (Q1), kuartil ketiga (Q3), IQR, dan pencilan "
        "untuk setiap fitur numerik per klaster. Whisker menggunakan aturan 1,5 × IQR; pencilan "
        "di luar whisker tetap ditampilkan."
    )
    box_meta = dict(box_figure.layout.meta or {})
    if box_meta.get("excluded_nonpositive"):
        box_caption += (
            f" Sebanyak {box_meta['excluded_nonpositive']:,} nilai nol atau negatif tidak dapat "
            "ditampilkan pada skala log."
        )
    st.caption(box_caption)

    st.subheader("Pola Kepadatan Nilai Atribut Numerik Antar-Klaster")
    log_usd_violin = st.checkbox(
        "Gunakan skala logaritmik untuk Nilai Impor (USD)",
        key="profile_violin_log_usd",
        value=False,
    )
    violin_figure = cluster_violinplots(
        clustered,
        stats,
        numerical_cols,
        feature_labels,
        title="Kepadatan Nilai Atribut Numerik per Klaster (satuan asli)",
        log_usd=log_usd_violin,
        number_formats=number_formats,
    )
    st.plotly_chart(violin_figure, width="stretch")
    violin_caption = (
        "Violin plot menunjukkan kepadatan distribusi nilai setiap atribut numerik per klaster. "
        "Lebar violin mencerminkan kepadatan observasi, sedangkan garis mean menunjukkan rata-rata "
        "klaster. Standar deviasi (SD) mengukur rata-rata jarak nilai observasi dari mean: semakin "
        "besar SD, semakin lebar dan bervariasi sebaran nilai dalam klaster. Nilai mean dan SD "
        "ditampilkan pada tooltip. Pencilan tetap ditampilkan."
    )
    violin_meta = dict(violin_figure.layout.meta or {})
    if violin_meta.get("excluded_nonpositive"):
        violin_caption += (
            f" Sebanyak {violin_meta['excluded_nonpositive']:,} nilai nol atau negatif tidak dapat "
            "ditampilkan pada skala log."
        )
    st.caption(violin_caption)

    st.subheader("Heatmap Profil Klaster Berdasarkan Z-Score")
    cluster_means = clustered.groupby("Cluster", sort=True)[numerical_cols].mean()
    global_means = clustered[numerical_cols].mean()
    global_stds = clustered[numerical_cols].std(ddof=0).replace(0, np.nan)
    z_profile = (
        cluster_means.subtract(global_means, axis="columns")
        .divide(global_stds, axis="columns")
        .fillna(0)
    )
    st.plotly_chart(cluster_z_heatmap(z_profile), width="stretch")
    st.caption(
        "Heatmap ini menampilkan Z-score rata-rata setiap fitur numerik per klaster relatif terhadap "
        "rata-rata dan standar deviasi global dataset. Nilai positif mengindikasikan fitur berada "
        "di atas rata-rata global, sedangkan nilai negatif mengindikasikan di bawah rata-rata. "
        "Pola ini digunakan untuk mengidentifikasi karakteristik pembeda setiap klaster."
    )

    _render_cluster_scatters(result)
    _render_shap_section(result)


def _render_cluster_scatters(result: dict) -> None:
    clustered = result["modeling"]["clustered"]
    st.subheader("Sebaran Klaster Nilai Impor terhadap Indikator Ekonomi Provinsi")
    st.caption(
        "Setiap titik merepresentasikan satu kombinasi pelabuhan–SITC, dengan warna menunjukkan "
        "keanggotaan klaster. Sumbu menggunakan nilai asli tanpa transformasi. Aktifkan skala "
        "logaritmik apabila distribusi nilai sangat menceng untuk memudahkan pembacaan pola."
    )

    specs = [
        {
            "y": "china_investment_us_ribu",
            "y_label": "Investasi Tiongkok",
            "title": "Nilai Impor vs Investasi Tiongkok",
            "key": "profile_scatter_log_investment",
            "caption": (
                "Scatter plot ini menampilkan distribusi kombinasi pelabuhan–SITC berdasarkan nilai "
                "impor dan realisasi investasi Tiongkok di provinsi terkait. Pola konsentrasi atau "
                "persebaran dapat menggambarkan karakteristik klaster, namun tidak mengindikasikan "
                "hubungan kausalitas antara kedua variabel."
            ),
        },
        {
            "y": "micro_small_industries",
            "y_label": "Jumlah Industri Mikro dan Kecil",
            "title": "Nilai Impor vs Jumlah Industri Mikro dan Kecil",
            "key": "profile_scatter_log_industries",
            "caption": (
                "Scatter plot ini menampilkan distribusi kombinasi pelabuhan–SITC berdasarkan nilai "
                "impor dan jumlah unit usaha industri mikro dan kecil di provinsi terkait. "
                "Interpretasi terbatas pada karakteristik sebaran; jumlah industri mikro dan kecil "
                "bukan proksi langsung kapasitas atau kebutuhan impor."
            ),
        },
        {
            "y": "pdrb",
            "y_label": "PDRB",
            "title": "Nilai Impor vs PDRB",
            "key": "profile_scatter_log_pdrb",
            "caption": (
                "Scatter plot ini menampilkan distribusi kombinasi pelabuhan–SITC berdasarkan nilai "
                "impor dan Produk Domestik Regional Bruto (PDRB) provinsi terkait. PDRB "
                "mencerminkan skala aktivitas ekonomi wilayah dan tidak menunjukkan hubungan "
                "kausalitas langsung dengan nilai impor."
            ),
        },
    ]

    for spec in specs:
        log_scale = st.checkbox(
            "Gunakan skala logaritmik",
            key=spec["key"],
            value=False,
            help=(
                "Skala logaritmik hanya mengubah tampilan sumbu. Nilai sumber dan label klaster "
                "tidak berubah; observasi bernilai 0 tidak dapat ditampilkan."
            ),
        )
        fig = cluster_scatter(
            clustered,
            "usd_value",
            spec["y"],
            "Nilai Impor (USD)",
            spec["y_label"],
            spec["title"],
            log_scale=log_scale,
        )
        st.plotly_chart(fig, width="stretch")

        caption = spec["caption"]
        meta = dict(fig.layout.meta or {})
        if meta.get("excluded_nonpositive"):
            caption += (
                f" Sebanyak {meta['excluded_nonpositive']:,} observasi bernilai 0 atau negatif "
                "tidak dapat ditampilkan pada skala log."
            )
        if meta.get("sampled"):
            caption += (
                f" Visualisasi menggunakan sampel deterministik {meta['n_plotted']:,} dari "
                f"{meta['n_total']:,} observasi (random_state=42)."
            )
        st.caption(caption)


def _render_shap_section(result: dict) -> None:
    st.subheader("Analisis SHAP (Interpretasi Model Surrogate)")
    st.caption(
        "SHAP mengukur seberapa besar kontribusi setiap fitur terhadap prediksi model surrogate. "
        "Karena K-Prototypes tidak menghasilkan nilai kepentingan fitur, model surrogate dilatih "
        "untuk meniru label klasternya. Seberapa andal nilai SHAP bergantung pada seberapa baik "
        "surrogate meniru K-Prototypes — yang dapat diperiksa melalui matriks konfusi di tab Evaluasi."
    )
    shap_result = result["modeling"]["shap"]
    if not shap_result.get("available"):
        st.warning(
            "Visualisasi SHAP tidak dapat ditampilkan. Alasan: "
            + str(shap_result.get("reason", "prasyarat data tidak terpenuhi"))
        )
        return

    st.markdown("**Kepentingan Fitur Global (Global Feature Importance)**")
    st.plotly_chart(global_shap_bar(shap_result["global_importance"]), width="stretch")
    st.caption(
        "Diagram batang ini menampilkan rata-rata nilai absolut SHAP setiap fitur di seluruh observasi "
        "dan klaster, sebagai ukuran kepentingan fitur secara global terhadap prediksi model surrogate."
    )

    st.markdown("**Kepentingan Fitur per Klaster (Heatmap Kontribusi)**")
    st.plotly_chart(
        shap_cluster_heatmap(shap_result["importance_per_cluster"]), width="stretch"
    )
    st.caption(
        "Heatmap ini menampilkan rata-rata nilai absolut SHAP per fitur untuk setiap klaster, "
        "sehingga perbedaan tingkat kepentingan fitur antarklaster dapat diidentifikasi."
    )

    st.markdown("**Distribusi Nilai SHAP per Fitur (Box Plot per Klaster)**")
    classes = shap_result["classes"]
    features = result["modeling"]["X_model"].loc[
        shap_result["X_eval"].index, result["modeling"]["feature_cols"]
    ]
    tabs = st.tabs([f"Cluster {c}" for c in classes])
    for tab, cluster_id in zip(tabs, classes):
        with tab:
            grouped_values = shap_result["grouped_by_cluster"][int(cluster_id)]
            try:
                fig = shap_beeswarm_figure(
                    grouped_values.to_numpy(), features, f"SHAP Box Plot — Cluster {cluster_id}"
                )
                st.pyplot(fig, width="stretch")
                st.caption(
                    "Box plot ini menampilkan distribusi nilai SHAP (Q1, median, Q3, dan whisker) "
                    "untuk setiap fitur pada klaster yang dipilih. Fitur diurutkan berdasarkan "
                    "rata-rata nilai SHAP absolut; nilai positif mengindikasikan kontribusi ke arah "
                    "prediksi klaster yang lebih tinggi, nilai negatif ke arah sebaliknya."
                )
                import matplotlib.pyplot as plt

                plt.close(fig)
            except Exception as exc:
                st.warning(f"Box plot SHAP untuk Cluster {cluster_id} tidak dapat dibuat: {exc}")



# ---------------------------------------------------------------------------
# View: Evaluasi
# ---------------------------------------------------------------------------

def _render_evaluation(result: dict) -> None:
    modeling = result["modeling"]
    elbow = modeling["elbow"]
    k = modeling["config"]["n_clusters"]
    silhouette = modeling["silhouette"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Klaster Terpilih (K)", str(k))
    c2.metric("Skor Silhouette (Campuran)", _fmt(silhouette))
    c3.metric("Ukuran Sampel Evaluasi", f"{modeling['silhouette_n']:,}")
    c4.metric("Koefisien Penyeimbang (Gamma)", _fmt(modeling["gamma"]))

    st.subheader("Penentuan Jumlah Klaster Optimal — Metode Elbow")
    st.plotly_chart(elbow_line(elbow, k), width="stretch")
    st.caption(
        "Grafik nilai cost K-Prototypes terhadap jumlah klaster (K). Nilai K optimal dipilih "
        "pada titik 'siku' (elbow), yaitu saat penurunan cost mulai tidak signifikan. "
        f"Nilai K yang digunakan pada model ini adalah K = {k}."
    )
    st.dataframe(
        elbow.rename(columns={"K": "Jumlah Klaster (K)", "Cost": "Total Cost"}).round(4),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Tabel ini mencatat nilai total cost K-Prototypes untuk setiap nilai K yang diuji."
    )

    st.subheader("Evaluasi Kohesi dan Separasi Klaster (Skor Silhouette)")
    st.caption(
        "Skor Silhouette dihitung menggunakan metrik ketidaksamaan campuran K-Prototypes: "
        "jumlah kuadrat selisih fitur numerik terstandardisasi ditambah penalti ketidakcocokan "
        "fitur kategorikal yang dibobot oleh parameter gamma."
    )
    if pd.isna(silhouette):
        st.warning("Skor Silhouette tidak dapat dihitung untuk konfigurasi data ini.")
    else:
        st.info(
            f"Skor Silhouette = {silhouette:.4f}, dihitung berdasarkan {modeling['silhouette_n']:,} sampel observasi "
            + ("(penarikan sampel deterministik, random_state=42)." if modeling["silhouette_n"] < len(modeling["clustered"]) else "(seluruh data).")
        )

        X_model = modeling.get("X_model")
        labels = modeling.get("labels")
        if X_model is None or labels is None:
            st.warning(
                "Nilai silhouette per observasi tidak tersedia pada hasil modeling ini, "
                "sehingga Silhouette Plot tidak dapat ditampilkan."
            )
        else:
            silhouette_df = _silhouette_samples_cached(
                X_model,
                np.asarray(labels),
                float(modeling["gamma"]),
                tuple(modeling["numerical_cols"]),
                tuple(modeling["categorical_cols"]),
                int(SILHOUETTE_SAMPLE_SIZE),
                42,
            )
            if silhouette_df.empty:
                st.warning(
                    "Silhouette Plot tidak dapat dibuat karena nilai per observasi tidak "
                    "tersedia atau jumlah klaster tidak valid."
                )
            else:
                sampled = len(silhouette_df) < len(modeling["clustered"])
                st.markdown("**Silhouette Plot per Klaster**")
                st.plotly_chart(
                    silhouette_plot(
                        silhouette_df,
                        sampled=sampled,
                        n_total=len(modeling["clustered"]),
                    ),
                    width="stretch",
                )
                st.caption(
                    "Setiap batang merepresentasikan nilai silhouette satu observasi, "
                    "dikelompokkan dan diurutkan per klaster. Garis vertikal menunjukkan "
                    "rata-rata nilai silhouette keseluruhan sebagai acuan kualitas pemisahan klaster."
                )

                mean_plot = float(silhouette_df["Silhouette"].mean())
                negative_share = float(
                    (silhouette_df["Silhouette"] < 0).mean() * 100
                )
                if abs(mean_plot - float(silhouette)) > 1e-6:
                    st.warning(
                        "Rata-rata silhouette dari plot "
                        f"({mean_plot:.6f}) berbeda dari skor evaluasi "
                        f"({float(silhouette):.6f}). Periksa konsistensi sampel evaluasi."
                    )

                sample_note = (
                    f"Plot memakai sampel deterministik {len(silhouette_df):,} dari "
                    f"{len(modeling['clustered']):,} observasi (random_state=42); rata-rata "
                    "yang ditampilkan adalah rata-rata sampel tersebut."
                    if sampled
                    else f"Plot memakai seluruh {len(silhouette_df):,} observasi; rata-rata "
                    "sama dengan skor evaluasi di atas."
                )
                st.caption(
                    "Nilai silhouette mendekati 1 menunjukkan observasi relatif cocok dengan "
                    "klasternya dan terpisah dari klaster terdekat. Nilai mendekati 0 menunjukkan "
                    "observasi berada di sekitar batas antarklaster, sedangkan nilai negatif "
                    "menunjukkan observasi mungkin lebih dekat ke klaster lain. "
                    f"Rata-rata yang diplot = {mean_plot:.4f}; proporsi observasi bernilai "
                    f"negatif = {negative_share:.2f}%. {sample_note} Tidak ada ambang kualitas "
                    "tunggal: nilai rata-rata, sebaran per klaster, dan proporsi negatif perlu "
                    "dibaca bersama-sama."
                )

                summary = (
                    silhouette_df.groupby("Cluster")["Silhouette"]
                    .agg(["size", "mean", "min", "max"])
                    .reset_index()
                )
                summary.columns = [
                    "Cluster",
                    "Jumlah",
                    "Rata-rata",
                    "Minimum",
                    "Maksimum",
                ]
                negative_by_cluster = (
                    silhouette_df.groupby("Cluster")["Silhouette"]
                    .apply(lambda values: float((values < 0).mean() * 100))
                    .reset_index(name="% Negatif")
                )
                summary = summary.merge(negative_by_cluster, on="Cluster")
                summary["Klaster"] = summary["Cluster"].apply(
                    lambda value: f"Cluster {int(value)}"
                )
                st.dataframe(
                    summary[
                        [
                            "Klaster",
                            "Jumlah",
                            "Rata-rata",
                            "Minimum",
                            "Maksimum",
                            "% Negatif",
                        ]
                    ].round(
                        {
                            "Rata-rata": 3,
                            "Minimum": 3,
                            "Maksimum": 3,
                            "% Negatif": 2,
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )

                if "Row" not in silhouette_df.columns:
                    st.warning(
                        "Kolom indeks observasi tidak tersedia sehingga unduhan silhouette "
                        "tidak dapat disiapkan. Jalankan ulang analisis."
                    )
                else:
                    metadata = (
                        modeling["clustered"][["port", "sitc"]]
                        .reset_index(drop=True)
                    )
                    metadata["Row"] = metadata.index
                    silhouette_export = silhouette_df.merge(
                        metadata, on="Row", how="left"
                    )

                    silhouette_all = (
                        silhouette_export[["port", "sitc", "Silhouette"]]
                        .rename(columns={"Silhouette": "silhouette"})
                        .sort_values(["port", "sitc"])
                        .reset_index(drop=True)
                    )
                    silhouette_negative = (
                        silhouette_export.loc[
                            silhouette_export["Silhouette"] < 0, ["port", "sitc"]
                        ]
                        .sort_values(["port", "sitc"])
                        .reset_index(drop=True)
                    )

                    download_col1, download_col2 = st.columns(2)
                    with download_col1:
                        st.download_button(
                            "Unduh Nilai Silhouette Seluruh Data (CSV)",
                            data=silhouette_all.to_csv(index=False).encode(
                                "utf-8-sig"
                            ),
                            file_name="silhouette_seluruh_data.csv",
                            mime="text/csv",
                            key="download_silhouette_all",
                            width="stretch",
                        )
                    with download_col2:
                        st.download_button(
                            "Unduh Daftar Silhouette Negatif (CSV)",
                            data=silhouette_negative.to_csv(index=False).encode(
                                "utf-8-sig"
                            ),
                            file_name="silhouette_negatif.csv",
                            mime="text/csv",
                            key="download_silhouette_negative",
                            width="stretch",
                            disabled=silhouette_negative.empty,
                        )
                    if sampled:
                        st.caption(
                            "Unduhan mengikuti observasi yang diplot pada Silhouette Plot. "
                            f"Karena data melebihi 2.500 observasi, berkas berisi sampel "
                            f"deterministik {len(silhouette_df):,} observasi "
                            "(random_state=42)."
                        )

                st.caption(
                    "Ringkasan statistik nilai silhouette per klaster, mencakup jumlah observasi, "
                    "rata-rata, nilai minimum, maksimum, dan proporsi observasi dengan nilai silhouette "
                    "negatif — yang mengindikasikan observasi berpotensi lebih dekat ke klaster lain."
                )

    st.subheader("Konfigurasi Parameter Algoritma K-Prototypes")
    config = modeling["config"]
    st.dataframe(
        pd.DataFrame(
            {
                "Parameter": ["init", "n_init", "random_state", "n_jobs", "n_clusters", "gamma"],
                "Nilai": [
                    str(config["init"]),
                    str(config["n_init"]),
                    str(config["random_state"]),
                    str(config["n_jobs"]),
                    str(config["n_clusters"]),
                    f"{modeling['gamma']:.4f}",
                ],
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Tabel ini mencatat nilai parameter yang digunakan pada eksekusi algoritma K-Prototypes, "
        "termasuk nilai gamma yang menentukan bobot relatif fitur kategorikal terhadap fitur numerik "
        "dalam metrik ketidaksamaan campuran."
    )

    st.subheader("Evaluasi Performa Model Surrogate (Random Forest)")
    surrogate = modeling["surrogate"]
    st.dataframe(
        surrogate["metrics"].round(4), width="stretch", hide_index=True
    )
    st.caption(
        "Metrik ini mengukur tingkat kesesuaian (fidelity) antara prediksi model surrogate Random Forest "
        "dan label klaster K-Prototypes pada data uji. Nilai akurasi dan F1-score yang tinggi "
        "mengindikasikan bahwa surrogate mereplikasi label klaster dengan baik, sehingga nilai SHAP "
        "yang diturunkan dari surrogate dapat dianggap valid sebagai interpretasi pengelompokan."
    )

    report = surrogate["classification_report"]
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df[report_df.index.map(lambda x: x not in {"accuracy", "macro avg", "weighted avg"})]
    report_df.index = [f"Cluster {i}" for i in report_df.index]
    st.markdown("**Laporan Klasifikasi per Klaster (Classification Report)**")
    st.dataframe(report_df.round(4), width="stretch")
    st.caption(
        "Tabel klasifikasi menampilkan precision, recall, dan F1-score model surrogate untuk "
        "setiap klaster pada data uji. Metrik ini dihitung berdasarkan perbandingan prediksi "
        "surrogate dengan label klaster K-Prototypes."
    )

    st.markdown("**Matriks Konfusi (Confusion Matrix)**")
    st.plotly_chart(
        confusion_heatmap(surrogate["confusion_matrix"], surrogate["class_labels"]),
        width="stretch",
    )
    st.caption(
        "Matriks konfusi mengukur seberapa baik surrogate meniru label klaster K-Prototypes pada data uji "
        "— bukan akurasi terhadap data asli. Semakin dominan elemen diagonal, semakin tinggi fidelity "
        "surrogate, dan semakin dapat dipercaya nilai SHAP sebagai cerminan logika pengelompokan K-Prototypes."
    )


# ---------------------------------------------------------------------------
# View: Peta
# ---------------------------------------------------------------------------

def _find_coordinate_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    lat = next((lookup[k] for k in ["latitude", "lat", "lintang"] if k in lookup), None)
    lon = next((lookup[k] for k in ["longitude", "lon", "lng", "bujur"] if k in lookup), None)
    return lat, lon


def _render_map(result: dict) -> None:
    clustered = result["modeling"]["clustered"]
    lat_col, lon_col = _find_coordinate_columns(clustered)

    if not lat_col or not lon_col:
        st.info(
            "Dataset input saat ini belum memuat kolom koordinat latitude dan longitude pelabuhan. "
            "Visualisasi peta dinonaktifkan untuk menjaga keakuratan informasi geografis."
        )
        st.markdown(
            "**Persyaratan kolom untuk mengaktifkan visualisasi peta:**\n"
            "- `latitude` (atau `lat` / `lintang`)\n"
            "- `longitude` (atau `lon` / `lng` / `bujur`)\n\n"
            "Sertakan data koordinat resmi pelabuhan pada berkas input, kemudian jalankan ulang "
            "proses analisis untuk menampilkan peta interaktif."
        )
        with st.expander("Contoh sintaks kode peta (untuk pengembangan fitur mendatang):"):
            st.code(
                "fig = px.scatter_geo(\n"
                "    clustered, lat='latitude', lon='longitude',\n"
                "    color='Cluster', hover_name='port',\n"
                "    scope='asia', projection='natural earth'\n"
                ")\n"
                "st.plotly_chart(fig, width='stretch')",
                language="python",
            )
        st.dataframe(
            pd.DataFrame({"Daftar Kolom pada Dataset": list(clustered.columns)}),
            width="stretch",
            hide_index=True,
        )
        return

    map_data = clustered.copy()
    map_data[lat_col] = pd.to_numeric(map_data[lat_col], errors="coerce")
    map_data[lon_col] = pd.to_numeric(map_data[lon_col], errors="coerce")
    map_data = map_data.dropna(subset=[lat_col, lon_col])
    map_data = map_data[
        map_data[lat_col].between(-90, 90) & map_data[lon_col].between(-180, 180)
    ]
    if map_data.empty:
        st.warning("Tidak ditemukan data koordinat yang valid pada dataset.")
        return

    if "sitc" not in map_data.columns:
        st.warning("Kolom `sitc` tidak ditemukan sehingga filter kategori SITC tidak tersedia.")
        return

    sitc_options = sorted(
        map_data["sitc"].astype(str).unique(),
        key=lambda value: int(value) if value.isdigit() else 99,
    )
    if st.session_state.get("map_sitc_choice") not in sitc_options:
        st.session_state["map_sitc_choice"] = sitc_options[0]

    selected_sitc = st.selectbox(
        "Kategori SITC (pilih satu)",
        sitc_options,
        key="map_sitc_choice",
        format_func=_sitc_label,
        help=(
            "Peta hanya menampilkan satu kategori SITC agar beberapa kategori pada "
            "pelabuhan yang sama tidak saling menumpuk."
        ),
    )

    subset = map_data[map_data["sitc"].astype(str) == selected_sitc].copy()
    subset = subset.drop_duplicates(subset=["port"]).reset_index(drop=True)
    subset["Klaster"] = subset["Cluster"].apply(
        lambda value: f"Cluster {int(value)}" if pd.notna(value) else "Cluster N/A"
    )

    from src.visualization import geo_scatter

    st.plotly_chart(
        geo_scatter(
            subset,
            lat_col,
            lon_col,
            "port",
            f"Sebaran Klaster Pelabuhan — SITC {selected_sitc}",
            color_col="Klaster",
        ),
        width="stretch",
    )

    collisions = subset[
        subset.duplicated(subset=[lat_col, lon_col], keep=False)
    ].sort_values([lat_col, lon_col, "port"])
    caption = (
        f"Peta menampilkan {len(subset):,} pelabuhan unik untuk kategori {_sitc_label(selected_sitc)}. "
        "Setiap titik merepresentasikan satu pelabuhan; warna menunjukkan keanggotaan klaster hasil pemodelan. "
        "Filter per kategori SITC diterapkan untuk mencegah penumpukan marker pada lokasi yang sama."
    )
    if not collisions.empty:
        caption += (
            f" Sebanyak {len(collisions):,} pelabuhan berbagi koordinat geografis dengan pelabuhan lain "
            "sehingga marker-nya saling bertumpang tindih — daftar selengkapnya tersedia pada panel di bawah."
        )
    st.caption(caption)

    if not collisions.empty:
        with st.expander("Pelabuhan dengan koordinat identik"):
            st.write(
                "Nama pelabuhan berikut berada pada koordinat yang sama sehingga marker-nya "
                "saling menumpuk. Posisi tidak digeser agar tetap akurat secara geografis."
            )
            st.dataframe(
                collisions[["port", "Provinsi", lat_col, lon_col, "Klaster"]],
                width="stretch",
                hide_index=True,
            )



# ---------------------------------------------------------------------------
# View: Data
# ---------------------------------------------------------------------------

def _dataset_catalog(result: dict) -> dict[str, tuple[str, pd.DataFrame]]:

    pre = result["preprocessing"]
    clustered = result["modeling"]["clustered"]
    return {
        "1. Dataset input mentah (port_imports)": (
            "dataset_input_port_imports.csv",
            pre["df_imports"],
        ),
        "2. Hasil konversi HS ke SITC": ("hasil_konversi_sitc.csv", pre["df_sitc"]),
        "3. Hasil agregasi Pelabuhan × SITC": (
            "hasil_agregasi_port_sitc.csv",
            pre["df_aggregated"],
        ),
        "4. Data setelah penggabungan provinsi": (
            "data_gabungan_provinsi.csv",
            pre["df_finished"],
        ),
        "5. Dataset final prapemrosesan": ("dataset_final_preprocessing.csv", pre["df_final"]),
        "6. Dataset dengan label klaster": ("dataset_berlabel_klaster.csv", clustered),
    }


def _render_data(result: dict) -> None:
    catalog = _dataset_catalog(result)

    st.subheader("Ringkasan Statistik Tahapan Prapemrosesan")
    report = result["preprocessing"]["report"]
    conversion = report["conversion"]
    summary = {
        "Baris data impor awal": f"{report['rows_imports']:,}",
        "Kode HS-6 tidak terbaca/valid": f"{conversion['rows_invalid_hs6']:,}",
        "Transaksi tanpa pemetaan SITC": f"{conversion['rows_unmapped']:,}",
        "Nilai transaksi tanpa pemetaan (USD)": f"US$ {conversion['usd_value_unmapped']:,.2f}",
        "Baris setelah agregasi Pelabuhan × SITC": f"{report['rows_aggregated']:,}",
        "Pelabuhan tanpa pemetaan data provinsi": f"{report['province']['rows_unmapped_port']:,}",
        "Baris duplikat yang dibersihkan": f"{report['cleaning']['duplicates_removed']:,}",
        "Baris dataset final siap model": f"{report['rows_final']:,}",
    }
    st.dataframe(
        pd.DataFrame({"Tahapan": list(summary), "Nilai": list(summary.values())}),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Tabel ini merangkum jumlah baris data pada setiap tahapan prapemrosesan, "
        "mulai dari data impor mentah hingga dataset final yang siap digunakan sebagai input model."
    )
    with st.expander("Detail penanganan outlier (IQR capping) dan imputasi median:"):
        st.json({"capping": report["capping"], "median_imputed": report["cleaning"]["median_imputed"]})

    st.subheader("Pratinjau dan Ekspor Dataset per Tahapan")
    choice = st.selectbox("Pilih tahapan dataset", list(catalog.keys()), key="data_dataset_choice")
    filename, df = catalog[choice]

    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah Baris", f"{len(df):,}")
    c2.metric("Jumlah Kolom", f"{len(df.columns):,}")
    c3.metric("Data Kosong (Missing)", f"{int(df.isna().sum().sum()):,}")

    max_rows = st.selectbox(
        "Jumlah baris ditampilkan", [25, 50, 100, 250, 500], index=1, key="data_max_rows"
    )
    st.dataframe(df.head(max_rows), width="stretch", hide_index=True)
    st.caption(
        f"Pratinjau menampilkan {min(max_rows, len(df)):,} baris pertama dari total {len(df):,} baris "
        f"pada tahapan '{choice}'. Gunakan tombol unduh untuk mengekspor seluruh dataset dalam format CSV."
    )

    st.download_button(
        f"Unduh berkas: {choice}",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
        key="download_dataset_choice",
    )

    st.subheader("Pratinjau Dataset Berlabel Klaster")
    clustered = result["modeling"]["clustered"]
    cluster_choice = st.selectbox(
        "Pilih nomor klaster", sorted(clustered["Cluster"].unique()), key="data_cluster_choice"
    )
    subset = clustered[clustered["Cluster"] == cluster_choice]
    st.dataframe(subset.head(max_rows), width="stretch", hide_index=True)
    st.caption(
        f"Pratinjau menampilkan {min(max_rows, len(subset)):,} baris pertama dari {len(subset):,} "
        f"observasi pada Klaster {cluster_choice}. Gunakan tombol unduh untuk mengekspor seluruh "
        "subset klaster dalam format CSV."
    )
    st.download_button(
        f"Unduh observasi Klaster {cluster_choice} (CSV)",
        data=subset.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"cluster_{cluster_choice}.csv",
        mime="text/csv",
        key="download_cluster_subset",
    )


# ---------------------------------------------------------------------------
# Sidebar + router
# ---------------------------------------------------------------------------

def _render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            "<div class='side-brand'><div class='side-brand-title'>DASHBOARD KLASTERISASI<br> KOMBINASI PELABUHAN DAN <br> KELOMPOK KOMODITAS BERDASARKAN AKTIVITAS <br> IMPOR TIONGKOK</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='side-section-label'>TAMPILAN DASHBOARD</div>", unsafe_allow_html=True)
        view = st.radio(
            "Tampilan dashboard",
            options=VIEWS,
            key="dashboard_view",
            label_visibility="collapsed",
        )
        if st.button("← Kembali ke Beranda", width="stretch"):
            st.switch_page("pages/1_Beranda.py")
        st.markdown(
            "<div class='sidebar-footer-caption'>Sistem Analitik Klasterisasi Kombinasi Pelabuhan dan Kelompok Komoditas</div>",
            unsafe_allow_html=True,
        )
    return view


def render() -> None:
    st.markdown(
        '<div class="dashboard-page-marker"></div>', unsafe_allow_html=True
    )
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

    view = _render_sidebar()

    st.markdown(
        """
        <div class="app-topbar">
          <div class="app-topbar-title">Dashboard Klasterisasi Kombinasi Pelabuhan dan Kelompok Komoditas di Indonesia Berdasarkan Aktivitas Impor Tiongkok</div>
          <div class="app-topbar-subtitle">Analitik · K-Prototypes · Pelabuhan Indonesia</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    title, description = VIEW_TITLES[view]
    st.title(title)
    st.caption(description)

    result = st.session_state.get("analysis_result")
    if not result:
        _no_results(view)
        return

    renderers = {
        "Ringkasan": _render_summary,
        "Profil Klaster": _render_profile,
        "Evaluasi": _render_evaluation,
        "Peta": _render_map,
        "Data": _render_data,
    }
    renderers[view](result)


render()

