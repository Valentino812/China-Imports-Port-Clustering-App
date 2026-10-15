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
    "Ringkasan": ("Ringkasan", "Tinjauan tingkat tinggi hasil klasterisasi."),
    "Profil Klaster": ("Profil Klaster", "Karakteristik dan distribusi setiap klaster."),
    "Evaluasi": ("Evaluasi Model", "Elbow, Silhouette campuran, dan evaluasi surrogate."),
    "Peta": ("Peta Distribusi Spasial", "Sebaran klaster bila koordinat tersedia."),
    "Data": ("Data", "Pratinjau dan ekspor dataset pada setiap tahap."),
}

DASHBOARD_CSS = r"""
<style>
/* Dashboard satu-satunya halaman yang menampilkan sidebar. */
[data-testid='stSidebar'] {
    display: block !important;
    background: #F4F3F3;
    border-right: 1px solid #D8D8D8;
    min-width: 260px;
}
[data-testid='stSidebarCollapsedControl'] { display: block !important; }
[data-testid='stSidebar'] > div:first-child { padding-top: 1rem; }
[data-testid='stSidebar'] [data-testid='stMarkdownContainer'] p,
[data-testid='stSidebar'] label,
[data-testid='stSidebar'] label span,
[data-testid='stSidebar'] [role='radiogroup'] p,
[data-testid='stSidebar'] [data-testid='stWidgetLabel'] {
    color: #292929 !important;
    font-size: 14px !important;
    line-height: 1.5 !important;
}
[data-testid='stSidebar'] [role='radiogroup'] label { padding-top: 5px; padding-bottom: 5px; font-weight: 500 !important; }
.side-brand { padding: 8px 4px 16px 4px; border-bottom: 1px solid #D9D9D9; margin-bottom: 15px; }
.side-brand-title { font-size: 16px; font-weight: 700; color: #1E1E1E; line-height: 1.45; }
.side-section-label { font-size: 11px; letter-spacing: .10em; font-weight: 700; color: #686868; margin: 14px 0 8px 2px; }
.sidebar-footer-caption { margin-top: 10px; text-align: center; font-size: 12px; color: #666666; }
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
        "Belum ada hasil analisis pada sesi ini. Buka halaman **Input Data**, unggah ketiga "
        "file wajib, lalu jalankan proses."
    )
    st.button(
        "Ke Halaman Input Data  â†’",
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
    c1.metric("Jumlah Observasi", f"{len(clustered):,}")
    c2.metric("Jumlah Pelabuhan", f"{clustered['port'].nunique():,}")
    c3.metric("Jumlah Provinsi", f"{clustered['Provinsi'].nunique():,}")
    c4.metric("Jumlah Klaster (K)", str(result["modeling"]["config"]["n_clusters"]))

    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(
            cluster_size_bar(struct, "jumlah_kombinasi_Port_SITC", "Ukuran Klaster"),
            width="stretch",
        )
    with right:
        st.plotly_chart(cluster_pie(struct), width="stretch")

    st.subheader("Tabel Ukuran dan Proporsi Klaster")
    display = struct.rename(
        columns={
            "Cluster": "Klaster",
            "jumlah_kombinasi_Port_SITC": "Jumlah Observasi",
            "jumlah_pelabuhan_unik": "Pelabuhan Unik",
            "jumlah_provinsi": "Provinsi",
            "persentase_dataset": "Persentase (%)",
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

    st.subheader("Jumlah dan Persentase Observasi per Klaster")
    overview = struct[["Cluster", "jumlah_kombinasi_Port_SITC", "persentase_dataset"]].rename(
        columns={
            "Cluster": "Klaster",
            "jumlah_kombinasi_Port_SITC": "Jumlah Observasi",
            "persentase_dataset": "Persentase (%)",
        }
    )
    st.dataframe(overview, width="stretch", hide_index=True)

    st.subheader("Statistik Fitur Numerik per Klaster (satuan asli)")
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

    st.subheader("Distribusi SITC per Klaster")
    sitc_pct = pd.crosstab(clustered["Cluster"], clustered["sitc"], normalize="index").mul(100)
    sitc_pct = sitc_pct.reindex(sorted(sitc_pct.columns), axis=1).round(2)
    sitc_display = sitc_pct.reset_index().rename(columns={"Cluster": "Klaster"})
    st.dataframe(sitc_display, width="stretch", hide_index=True)
    st.plotly_chart(sitc_stacked_bar(sitc_pct), width="stretch")

    st.subheader("Distribusi Provinsi (5 teratas per klaster)")
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

    st.subheader("Boxplot Fitur Numerik")
    st.plotly_chart(numerical_mean_bars(clustered, numerical_cols), width="stretch")
    st.caption("Grafik batang rata-rata fitur numerik dalam satuan asli.")

    from src.visualization import numeric_boxplots

    st.plotly_chart(numeric_boxplots(clustered, numerical_cols), width="stretch")

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
    st.caption("Nilai Z-score relatif terhadap keseluruhan data (digunakan hanya untuk perbandingan).")

    _render_shap_section(result)


def _render_shap_section(result: dict) -> None:
    st.subheader("Analisis SHAP (Penjelasan Surrogate)")
    st.caption(
        "SHAP menjelaskan prediksi model surrogate Random Forest, bukan mekanisme internal "
        "K-Prototypes."
    )
    shap_result = result["modeling"]["shap"]
    if not shap_result.get("available"):
        st.warning(
            "SHAP tidak dapat ditampilkan. Alasan: "
            + str(shap_result.get("reason", "prasyarat tidak terpenuhi"))
        )
        return

    st.markdown("**Global Feature Importance**")
    st.plotly_chart(global_shap_bar(shap_result["global_importance"]), width="stretch")

    st.markdown("**Kontribusi Fitur per Klaster**")
    st.plotly_chart(
        shap_cluster_heatmap(shap_result["importance_per_cluster"]), width="stretch"
    )

    st.markdown("**Beeswarm per Klaster**")
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
                    grouped_values.to_numpy(), features, f"SHAP Beeswarm â€” Cluster {cluster_id}"
                )
                st.pyplot(fig, width="stretch")
                import matplotlib.pyplot as plt

                plt.close(fig)
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Beeswarm untuk Cluster {cluster_id} tidak dapat dibuat: {exc}")


# ---------------------------------------------------------------------------
# View: Evaluasi
# ---------------------------------------------------------------------------

def _render_evaluation(result: dict) -> None:
    modeling = result["modeling"]
    elbow = modeling["elbow"]
    k = modeling["config"]["n_clusters"]
    silhouette = modeling["silhouette"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("K Terpilih", str(k))
    c2.metric("Silhouette (campuran)", _fmt(silhouette))
    c3.metric("Observasi Silhouette", f"{modeling['silhouette_n']:,}")
    c4.metric("Gamma K-Prototypes", _fmt(modeling["gamma"]))

    st.subheader("Elbow Method (Cost K-Prototypes)")
    st.plotly_chart(elbow_line(elbow, k), width="stretch")
    st.dataframe(
        elbow.rename(columns={"K": "K", "Cost": "Cost"}).round(4),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Silhouette Campuran")
    st.write(
        "Silhouette dihitung dengan dissimilarity campuran K-Prototypes: kuadrat selisih fitur "
        "numerik terstandardisasi ditambah gamma dikali jumlah ketidakcocokan kategorikal."
    )
    if pd.isna(silhouette):
        st.warning("Silhouette tidak dapat dihitung untuk konfigurasi ini.")
    else:
        st.info(
            f"Silhouette = {silhouette:.4f}, dihitung pada {modeling['silhouette_n']:,} observasi "
            + ("(sampel deterministik, random_state=42)." if modeling["silhouette_n"] < len(modeling["clustered"]) else "(seluruh data).")
        )

    st.subheader("Konfigurasi K-Prototypes")
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

    st.subheader("Evaluasi Surrogate Random Forest")
    surrogate = modeling["surrogate"]
    st.dataframe(
        surrogate["metrics"].round(4), width="stretch", hide_index=True
    )
    st.caption(
        "Metrik ini mengukur fidelity surrogate dalam mereproduksi label klaster, bukan "
        "validasi independen kualitas klaster."
    )

    report = surrogate["classification_report"]
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df[report_df.index.map(lambda x: x not in {"accuracy", "macro avg", "weighted avg"})]
    report_df.index = [f"Cluster {i}" for i in report_df.index]
    st.markdown("**Classification Report per Klaster**")
    st.dataframe(report_df.round(4), width="stretch")

    st.markdown("**Confusion Matrix**")
    st.plotly_chart(
        confusion_heatmap(surrogate["confusion_matrix"], surrogate["class_labels"]),
        width="stretch",
    )

    st.download_button(
        "Unduh metrik evaluasi (CSV)",
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
            "Dataset yang digunakan tidak menyediakan kolom koordinat latitude dan longitude. "
            "Peta tidak dibuat agar tidak menampilkan lokasi yang tidak terverifikasi."
        )
        st.markdown(
            "**Kolom yang dibutuhkan untuk mengaktifkan peta:**\n"
            "- `latitude` (atau `lat` / `lintang`)\n"
            "- `longitude` (atau `lon` / `lng` / `bujur`)\n\n"
            "Tambahkan koordinat yang berasal dari sumber valid (misalnya tabel koordinat pelabuhan "
            "resmi) ke dataset input, lalu jalankan ulang analisis."
        )
        with st.expander("Template kode peta (untuk pengembangan berikutnya)"):
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
            pd.DataFrame({"Kolom tersedia": list(clustered.columns)}),
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
        st.warning("Tidak ada koordinat valid pada dataset.")
        return

    from src.visualization import geo_scatter

    st.plotly_chart(
        geo_scatter(map_data, lat_col, lon_col, "port", "Sebaran Klaster Pelabuhan"),
        width="stretch",
    )
    st.caption(f"Menampilkan {len(map_data):,} observasi dengan koordinat valid.")


# ---------------------------------------------------------------------------
# View: Data
# ---------------------------------------------------------------------------

def _dataset_catalog(result: dict) -> dict[str, tuple[str, pd.DataFrame]]:
    pre = result["preprocessing"]
    clustered = result["modeling"]["clustered"]
    return {
        "1. Dataset input (combined_dataset)": (
            "dataset_input_combined.csv",
            pre["df_combined"],
        ),
        "2. Hasil konversi HSâ†’SITC": ("hasil_konversi_sitc.csv", pre["df_sitc"]),
        "3. Hasil agregasi Port Ã— SITC": (
            "hasil_agregasi_port_sitc.csv",
            pre["df_aggregated"],
        ),
        "4. Data setelah penggabungan provinsi": (
            "data_gabungan_provinsi.csv",
            pre["df_finished"],
        ),
        "5. Dataset final preprocessing": ("dataset_final_preprocessing.csv", pre["df_final"]),
        "6. Dataset dengan label klaster": ("dataset_berlabel_klaster.csv", clustered),
    }


def _render_data(result: dict) -> None:
    catalog = _dataset_catalog(result)

    st.subheader("Ringkasan Tahapan Proses")
    report = result["preprocessing"]["report"]
    conversion = report["conversion"]
    summary = {
        "Baris impor awal": f"{report['rows_combined']:,}",
        "HS6 tidak terbaca": f"{conversion['rows_invalid_hs6']:,}",
        "Baris tanpa pemetaan SITC": f"{conversion['rows_unmapped']:,}",
        "Nilai USD tanpa pemetaan": f"{conversion['usd_value_unmapped']:,.2f}",
        "Baris setelah agregasi Port Ã— SITC": f"{report['rows_aggregated']:,}",
        "Pelabuhan tanpa pemetaan provinsi": f"{report['province']['rows_unmapped_port']:,}",
        "Duplikat dihapus": f"{report['cleaning']['duplicates_removed']:,}",
        "Baris final": f"{report['rows_final']:,}",
    }
    st.dataframe(
        pd.DataFrame({"Tahapan": list(summary), "Nilai": list(summary.values())}),
        width="stretch",
        hide_index=True,
    )
    with st.expander("Detail IQR capping dan imputasi median"):
        st.json({"capping": report["capping"], "median_imputed": report["cleaning"]["median_imputed"]})

    st.subheader("Pratinjau dan Ekspor Dataset")
    choice = st.selectbox("Pilih dataset", list(catalog.keys()), key="data_dataset_choice")
    filename, df = catalog[choice]

    c1, c2, c3 = st.columns(3)
    c1.metric("Baris", f"{len(df):,}")
    c2.metric("Kolom", f"{len(df.columns):,}")
    c3.metric("Nilai kosong", f"{int(df.isna().sum().sum()):,}")

    max_rows = st.selectbox(
        "Jumlah baris ditampilkan", [25, 50, 100, 250, 500], index=1, key="data_max_rows"
    )
    st.dataframe(df.head(max_rows), width="stretch", hide_index=True)
    st.caption(f"Menampilkan maksimal {max_rows} baris dari {len(df):,} baris.")

    st.download_button(
        f"Unduh dataset: {choice}",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
        key="download_dataset_choice",
    )

    st.subheader("Pratinjau Profil Klaster")
    clustered = result["modeling"]["clustered"]
    cluster_choice = st.selectbox(
        "Pilih klaster", sorted(clustered["Cluster"].unique()), key="data_cluster_choice"
    )
    subset = clustered[clustered["Cluster"] == cluster_choice]
    st.dataframe(subset.head(max_rows), width="stretch", hide_index=True)
    st.download_button(
        f"Unduh observasi Cluster {cluster_choice}",
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
            "<div class='side-brand'><div class='side-brand-title'>DASHBOARD KLASTERISASI<br>IMPOR TIONGKOK</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='side-section-label'>TAMPILAN DASHBOARD</div>", unsafe_allow_html=True)
        view = st.radio(
            "Tampilan dashboard",
            options=VIEWS,
            key="dashboard_view",
            label_visibility="collapsed",
        )
        if st.button("â† Kembali ke Homepage", width="stretch"):
            st.switch_page("pages/1_Beranda.py")
        st.markdown(
            "<div class='sidebar-footer-caption'>Aplikasi analitik klasterisasi</div>",
            unsafe_allow_html=True,
        )
    return view


def render() -> None:
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

    view = _render_sidebar()

    st.markdown(
        """
        <div class="app-topbar">
          <div class="app-topbar-title">Dashboard Klasterisasi Impor Tiongkok</div>
          <div class="app-topbar-subtitle">Analitik Â· Klasterisasi Â· Indonesia</div>
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
