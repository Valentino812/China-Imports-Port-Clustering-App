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

from src.visualization import (
    cluster_pie,
    cluster_size_bar,
    cluster_z_heatmap,
    confusion_heatmap,
    elbow_line,
    global_shap_bar,
    numerical_mean_bars,
    shap_beeswarm_figure,
    shap_cluster_heatmap,
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
/* ─── Dashboard: unhide sidebar ─────────────────────────────────────── */
[data-testid='stSidebar'] {
    display: block !important;
    background: rgba(255,255,255,0.95) !important;
    border-right: 1px solid rgba(226,229,241,0.8) !important;
    min-width: 265px;
    box-shadow: 2px 0 12px rgba(79,107,237,0.06);
}
[data-testid='stSidebarCollapsedControl'] { display: block !important; }
[data-testid='stSidebar'] > div:first-child { padding-top: 1.25rem; }

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

    st.subheader("Distribusi Jumlah dan Proporsi Observasi per Klaster")
    overview = struct[["Cluster", "jumlah_kombinasi_Port_SITC", "persentase_dataset"]].rename(
        columns={
            "Cluster": "Klaster",
            "jumlah_kombinasi_Port_SITC": "Jumlah Observasi",
            "persentase_dataset": "Proporsi (%)",
        }
    )
    st.dataframe(overview, width="stretch", hide_index=True)

    st.subheader("Statistik Fitur Numerik per Klaster")
    rows = []
    for cluster_id, group in clustered.groupby("Cluster", sort=True):
        for feature in numerical_cols:
            q1 = group[feature].quantile(0.25)
            q3 = group[feature].quantile(0.75)
            rows.append(
                {
                    "Klaster": cluster_id,
                    "Fitur": feature,
                    "Mean": round(group[feature].mean(), 3),
                    "Median": round(group[feature].median(), 3),
                    "Q1": round(q1, 3),
                    "Q3": round(q3, 3),
                    "IQR": round(q3 - q1, 3),
                }
            )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.subheader("Modus Fitur Kategorikal per Klaster")
    mode_rows = []
    for cluster_id, group in clustered.groupby("Cluster", sort=True):
        row = {"Klaster": cluster_id}
        for col in categorical_cols:
            row[f"Modus {col.upper()}"] = _get_mode_text(group[col])
        mode_rows.append(row)
    st.dataframe(pd.DataFrame(mode_rows), width="stretch", hide_index=True)

    st.subheader("Distribusi Kategori SITC per Klaster")
    sitc_pct = pd.crosstab(clustered["Cluster"], clustered["sitc"], normalize="index").mul(100)
    sitc_pct = sitc_pct.reindex(sorted(sitc_pct.columns), axis=1).round(2)
    sitc_display = sitc_pct.reset_index().rename(columns={"Cluster": "Klaster"})
    st.dataframe(sitc_display, width="stretch", hide_index=True)
    st.plotly_chart(sitc_stacked_bar(sitc_pct), width="stretch")

    st.subheader("Distribusi Wilayah Provinsi (5 Teratas per Klaster)")
    province_counts = (
        clustered.groupby(["Cluster", "Provinsi"])
        .size()
        .rename("jumlah_kombinasi_Port_SITC")
        .reset_index()
        .sort_values(["Cluster", "jumlah_kombinasi_Port_SITC"], ascending=[True, False])
    )
    top_prov = province_counts.groupby("Cluster", sort=True).head(5).reset_index(drop=True)
    top_prov = top_prov.rename(
        columns={"Cluster": "Klaster", "jumlah_kombinasi_Port_SITC": "Jumlah Observasi"}
    )
    st.dataframe(top_prov, width="stretch", hide_index=True)

    st.subheader("Perbandingan Nilai Rata-Rata Fitur Numerik")
    st.plotly_chart(numerical_mean_bars(clustered, numerical_cols), width="stretch")
    st.caption(
        "Diagram batang ini menampilkan nilai rata-rata setiap fitur numerik (nilai impor USD, PDRB, "
        "investasi Tiongkok, dan jumlah industri mikro & kecil) untuk masing-masing klaster dalam "
        "skala satuan aslinya, sehingga perbedaan besaran absolut antar kelompok dapat dibandingkan secara langsung."
    )

    from src.visualization import numeric_boxplots

    st.subheader("Distribusi Sebaran Fitur Numerik (Boxplot)")
    st.plotly_chart(numeric_boxplots(clustered, numerical_cols), width="stretch")
    st.caption(
        "Diagram kotak (boxplot) memperlihatkan distribusi nilai tiap fitur numerik per klaster, "
        "meliputi median (Q2), batas kuartil bawah (Q1) dan atas (Q3), rentang antarquartile (IQR), "
        "serta nilai pencilan di luar batas whisker. Visualisasi ini berguna untuk menilai homogenitas "
        "dan variabilitas internal masing-masing klaster."
    )

    st.subheader("Heatmap Profil Klaster Terstandardisasi")
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
        "Heatmap ini menampilkan nilai Z-score rata-rata setiap fitur numerik per klaster, "
        "dihitung relatif terhadap rata-rata dan standar deviasi keseluruhan dataset. "
        "Nilai positif (lebih terang) menandakan fitur tersebut di atas rata-rata global, "
        "sedangkan nilai negatif (lebih gelap) menandakan di bawah rata-rata — "
        "memudahkan identifikasi pola pembeda yang menjadi ciri khas tiap klaster."
    )

    _render_shap_section(result)


def _render_shap_section(result: dict) -> None:
    st.subheader("Analisis SHAP (Interpretasi Model Surrogate)")
    st.caption(
        "Analisis SHAP (SHapley Additive exPlanations) mengukur dan memvisualisasikan kontribusi "
        "masing-masing fitur terhadap keputusan prediksi model surrogate Random Forest. "
        "Karena model K-Prototypes bersifat black-box dalam hal interpretasi fitur, model surrogate "
        "dilatih untuk mereplikasi label klasternya, kemudian nilai SHAP diekstrak dari surrogate "
        "tersebut sebagai pendekatan kuantitatif dalam menjelaskan faktor penentu pengelompokan."
    )
    shap_result = result["modeling"]["shap"]
    if not shap_result.get("available"):
        st.warning(
            "Visualisasi SHAP tidak dapat ditampilkan. Alasan: "
            + str(shap_result.get("reason", "prasyarat data tidak terpenuhi"))
        )
        return

    st.markdown("**Tingkat Kepentingan Fitur Global (Global Feature Importance)**")
    st.plotly_chart(global_shap_bar(shap_result["global_importance"]), width="stretch")

    st.markdown("**Pola Kontribusi Fitur per Klaster**")
    st.plotly_chart(
        shap_cluster_heatmap(shap_result["importance_per_cluster"]), width="stretch"
    )

    st.markdown("**Distribusi Pengaruh Fitur (Beeswarm Plot per Klaster)**")
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
                    grouped_values.to_numpy(), features, f"SHAP Beeswarm — Cluster {cluster_id}"
                )
                st.pyplot(fig, width="stretch")
                import matplotlib.pyplot as plt

                plt.close(fig)
            except Exception as exc:  
                st.warning(f"Visualisasi Beeswarm untuk Cluster {cluster_id} tidak dapat dibuat: {exc}")



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

    st.subheader("Metode Elbow (Nilai Cost K-Prototypes)")
    st.plotly_chart(elbow_line(elbow, k), width="stretch")
    st.dataframe(
        elbow.rename(columns={"K": "Jumlah Klaster (K)", "Cost": "Total Cost"}).round(4),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Evaluasi Kohesi & Separasi (Silhouette Campuran)")
    st.write(
        "Skor Silhouette dihitung menggunakan metrik ketidaksamaan (dissimilarity) campuran K-Prototypes: "
        "penjumlahan kuadrat selisih fitur numerik terstandardisasi dengan penalti ketidakcocokan kategori "
        "yang dibobotkan oleh parameter gamma."
    )
    if pd.isna(silhouette):
        st.warning("Skor Silhouette tidak dapat dihitung untuk konfigurasi data ini.")
    else:
        st.info(
            f"Skor Silhouette = {silhouette:.4f}, dihitung berdasarkan {modeling['silhouette_n']:,} sampel observasi "
            + ("(penarikan sampel deterministik, random_state=42)." if modeling["silhouette_n"] < len(modeling["clustered"]) else "(seluruh data).")
        )

    st.subheader("Parameter Konfigurasi Algoritma K-Prototypes")
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

    st.subheader("Evaluasi Performa Model Surrogate (Random Forest)")
    surrogate = modeling["surrogate"]
    st.dataframe(
        surrogate["metrics"].round(4), width="stretch", hide_index=True
    )
    st.caption(
        "Metrik di atas mengukur tingkat kesesuaian (fidelity) antara prediksi model surrogate Random Forest "
        "dan label klaster yang dihasilkan oleh K-Prototypes pada data uji. Semakin tinggi nilai akurasi dan "
        "F1-score surrogate, semakin valid kesimpulan analisis SHAP sebagai penjelasan atas keputusan "
        "pengelompokan — karena SHAP diderivasi dari surrogate, bukan langsung dari K-Prototypes."
    )

    report = surrogate["classification_report"]
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df[report_df.index.map(lambda x: x not in {"accuracy", "macro avg", "weighted avg"})]
    report_df.index = [f"Cluster {i}" for i in report_df.index]
    st.markdown("**Laporan Klasifikasi per Klaster (Classification Report)**")
    st.dataframe(report_df.round(4), width="stretch")

    st.markdown("**Matriks Konfusi (Confusion Matrix)**")
    st.plotly_chart(
        confusion_heatmap(surrogate["confusion_matrix"], surrogate["class_labels"]),
        width="stretch",
    )

    st.download_button(
        "Unduh Metrik Evaluasi Surrogate (CSV)",
        data=surrogate["metrics"].to_csv(index=False).encode("utf-8-sig"),
        file_name="evaluasi_surrogate.csv",
        mime="text/csv",
        key="download_eval_metrics",
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
        "Setiap titik merepresentasikan satu pelabuhan; warna menunjukkan keanggotaan klaster hasil pemodelan."
    )
    if not collisions.empty:
        caption += (
            f" Terdapat {len(collisions):,} pelabuhan yang berbagi koordinat geografis dengan pelabuhan lain "
            "sehingga marker-nya saling menumpuk pada peta — daftar lengkapnya tersedia di panel di bawah."
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

    st.subheader("Ringkasan Tahapan Prapemrosesan")
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
    with st.expander("Detail penanganan outlier (IQR capping) dan imputasi median:"):
        st.json({"capping": report["capping"], "median_imputed": report["cleaning"]["median_imputed"]})

    st.subheader("Pratinjau dan Ekspor Dataset")
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
        f"Pratinjau menampilkan {min(max_rows, len(df)):,} baris pertama dari total {len(df):,} baris data "
        f"pada tahapan '{choice}'. Gunakan tombol unduh di bawah untuk mengekspor keseluruhan dataset dalam format CSV."
    )

    st.download_button(
        f"Unduh berkas: {choice}",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
        key="download_dataset_choice",
    )

    st.subheader("Pratinjau Data per Klaster")
    clustered = result["modeling"]["clustered"]
    cluster_choice = st.selectbox(
        "Pilih nomor klaster", sorted(clustered["Cluster"].unique()), key="data_cluster_choice"
    )
    subset = clustered[clustered["Cluster"] == cluster_choice]
    st.dataframe(subset.head(max_rows), width="stretch", hide_index=True)
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

