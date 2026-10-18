"""Plotly / Matplotlib figure builders for the dashboard views.

Keeping every figure constructor here keeps ``pages/3_Dashboard.py`` focused on
layout and state handling. Data is never fabricated here: figures reflect the
actual analysis output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

NEUTRAL_SEQUENCE = ["#4C6EF5", "#12B886", "#F59F00", "#FA5252", "#7950F2", "#15AABF", "#E64980"]

# Palet kualitatif kontras tinggi dan colorblind-safe (Okabe-Ito). Dipakai untuk
# kategori diskret agar setiap kelompok mendapat warna yang jelas berbeda, bukan
# gradasi dari skala warna kontinu.
QUALITATIVE_SEQUENCE = [
    "#0072B2",  # biru
    "#E69F00",  # kuning/oranye
    "#009E73",  # hijau
    "#D55E00",  # merah bata
    "#CC79A7",  # magenta
    "#56B4E9",  # biru muda
    "#F0E442",  # kuning
]


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color="#6B6B6B"),
    )
    fig.update_layout(
        template="plotly_white",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _style(fig: go.Figure, height: int = 360, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        font=dict(family="Inter, sans-serif", color="#1A1C1C"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=15)))
    return fig


def cluster_size_bar(cluster_structure: pd.DataFrame, value_col: str, title: str) -> go.Figure:
    if cluster_structure.empty:
        return _empty_figure("Tidak ada data klaster untuk ditampilkan.")
    df = cluster_structure.copy()
    df["Label"] = df["Cluster"].apply(lambda c: f"Cluster {c}")
    fig = px.bar(
        df,
        x="Label",
        y=value_col,
        text=value_col,
        color="Label",
        color_discrete_sequence=NEUTRAL_SEQUENCE,
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, xaxis_title="Klaster", yaxis_title=value_col)
    return _style(fig, title=title)


def cluster_pie(cluster_structure: pd.DataFrame) -> go.Figure:
    if cluster_structure.empty:
        return _empty_figure("Tidak ada data klaster untuk ditampilkan.")
    df = cluster_structure.copy()
    df["Label"] = df["Cluster"].apply(lambda c: f"Cluster {c}")
    fig = px.pie(
        df,
        names="Label",
        values="jumlah_kombinasi_Port_SITC",
        hole=0.45,
        color_discrete_sequence=NEUTRAL_SEQUENCE,
    )
    fig.update_traces(textinfo="percent+label")
    return _style(fig, title="Proporsi Klaster")


def numeric_boxplots(data: pd.DataFrame, numerical_cols: list[str]) -> go.Figure:
    if not numerical_cols:
        return _empty_figure("Tidak ada fitur numerik untuk boxplot.")
    long_df = data.melt(
        id_vars=["Cluster"], value_vars=numerical_cols, var_name="Fitur", value_name="Nilai"
    )
    long_df["Label"] = long_df["Cluster"].apply(lambda c: f"Cluster {c}")
    fig = px.box(
        long_df,
        x="Label",
        y="Nilai",
        color="Label",
        facet_col="Fitur",
        facet_col_wrap=2,
        color_discrete_sequence=NEUTRAL_SEQUENCE,
    )
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.update_layout(showlegend=False)
    return _style(fig, height=420, title="Distribusi Fitur Numerik per Klaster")


def sitc_stacked_bar(sitc_distribution: pd.DataFrame) -> go.Figure:
    if sitc_distribution.empty:
        return _empty_figure("Tidak ada distribusi SITC untuk ditampilkan.")
    df = sitc_distribution.copy()
    df.index = [f"Cluster {c}" for c in df.index]
    df = df.reset_index().rename(columns={"index": "Klaster"})
    long_df = df.melt(id_vars="Klaster", var_name="SITC", value_name="Persentase")
    long_df["SITC"] = long_df["SITC"].astype(str).apply(lambda s: f"SITC {s}")
    fig = px.bar(
        long_df,
        x="Klaster",
        y="Persentase",
        color="SITC",
        color_discrete_sequence=px.colors.qualitative.Set3,
        text="SITC",
    )
    fig.update_traces(textposition="inside", insidetextanchor="middle", textfont_size=10)
    fig.update_layout(barmode="stack", yaxis_title="Proporsi dalam klaster (%)", yaxis_range=[0, 100])
    return _style(fig, height=400, title="Komposisi Kelompok SITC per Klaster")


def cluster_z_heatmap(z_profile: pd.DataFrame) -> go.Figure:
    if z_profile.empty:
        return _empty_figure("Tidak ada profil terstandardisasi untuk ditampilkan.")
    fig = px.imshow(
        z_profile,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-2,
        zmax=2,
        text_auto=".2f",
        labels=dict(x="Fitur numerik", y="Klaster", color="Z-score"),
    )
    return _style(fig, height=380, title="Profil Rata-rata Fitur Numerik (Z-score)")


def elbow_line(elbow_df: pd.DataFrame, selected_k: int) -> go.Figure:
    if elbow_df.empty:
        return _empty_figure("Elbow belum tersedia.")
    fig = px.line(
        elbow_df,
        x="K",
        y="Cost",
        markers=True,
        labels={"Cost": "Cost K-Prototypes", "K": "Jumlah klaster (K)"},
    )
    fig.add_vline(x=selected_k, line_dash="dash", line_color="#555555")
    fig.add_annotation(x=selected_k, y=elbow_df["Cost"].max(), text=f"K aktif = {selected_k}")
    return _style(fig, height=380, title="Elbow Method untuk K-Prototypes")


def global_shap_bar(global_importance: pd.DataFrame) -> go.Figure:
    if global_importance is None or global_importance.empty:
        return _empty_figure("Global SHAP importance tidak tersedia.")
    df = global_importance.sort_values("Mean absolute SHAP")
    fig = px.bar(
        df,
        x="Mean absolute SHAP",
        y="Fitur",
        orientation="h",
        color_discrete_sequence=["#4C6EF5"],
        text="Mean absolute SHAP",
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    return _style(fig, height=360, title="Global Feature Importance (SHAP)")


def shap_cluster_heatmap(importance_per_cluster: pd.DataFrame) -> go.Figure:
    if importance_per_cluster is None or importance_per_cluster.empty:
        return _empty_figure("SHAP per klaster tidak tersedia.")
    fig = px.imshow(
        importance_per_cluster,
        aspect="auto",
        color_continuous_scale="YlOrRd",
        text_auto=".4f",
        labels=dict(x="Fitur clustering", y="Kelas target surrogate", color="Mean |SHAP|"),
    )
    return _style(fig, height=380, title="Kontribusi Fitur per Klaster (SHAP)")


def confusion_heatmap(cm: np.ndarray, labels: list) -> go.Figure:
    if cm is None or len(cm) == 0:
        return _empty_figure("Confusion matrix belum tersedia.")
    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        x=[f"Pred {c}" for c in labels],
        y=[f"Aktual {c}" for c in labels],
        labels=dict(x="Prediksi surrogate", y="Label asli K-Prototypes", color="Jumlah"),
    )
    return _style(fig, height=420, title="Confusion Matrix Surrogate")


def shap_beeswarm_figure(shap_values, features: pd.DataFrame, title: str):
    """Build a Matplotlib beeswarm figure for one cluster's SHAP values."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    plt.figure(figsize=(8, 5))
    shap.summary_plot(
        shap_values,
        features,
        plot_type="dot",
        max_display=min(15, features.shape[1]),
        show=False,
    )
    fig = plt.gcf()
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig


def geo_scatter(
    data: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    hover_name: str | None,
    title: str,
    color_col: str = "Cluster",
    legend_title: str = "Klaster",
) -> go.Figure:
    """Scatter map whose markers are coloured by a discrete category.

    ``color_col`` is converted to text before plotting. That forces Plotly to use
    a qualitative palette; passing a numeric column (for example the raw
    ``Cluster`` labels) would otherwise produce a continuous colour scale.
    """
    df = data.copy()
    group_col = "__kelompok_warna__"
    df[group_col] = df[color_col].astype(str)
    groups = sorted(df[group_col].unique())
    color_map = {
        group: QUALITATIVE_SEQUENCE[i % len(QUALITATIVE_SEQUENCE)]
        for i, group in enumerate(groups)
    }

    hover_data: dict[str, object] = {}
    for col in ("Provinsi", "Klaster", "sitc"):
        if col in df.columns and col != color_col:
            hover_data[col] = True
    if "usd_value" in df.columns:
        hover_data["usd_value"] = ":,.2f"

    fig = px.scatter_geo(
        df,
        lat=lat_col,
        lon=lon_col,
        color=group_col,
        hover_name=hover_name,
        hover_data=hover_data,
        category_orders={group_col: groups},
        color_discrete_map=color_map,
        labels={group_col: legend_title},
        scope="asia",
        projection="natural earth",
        title=title,
    )
    fig.update_traces(
        marker=dict(size=10, opacity=0.9, line=dict(width=1, color="white"))
    )
    fig.update_geos(fitbounds="locations", visible=True, showcountries=True, countrycolor="#D0D0D0")
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=50, b=0), legend_title_text=legend_title)
    return fig


def feature_separation_bar(feature_stats: pd.DataFrame) -> go.Figure:
    if feature_stats is None or feature_stats.empty:
        return _empty_figure("Tidak ada statistik fitur.")
    df = feature_stats.dropna(subset=["Rasio Pemisahan"]).sort_values("Rasio Pemisahan")
    if df.empty:
        return _empty_figure("Rasio pemisahan antar klaster tidak dapat dihitung.")
    fig = px.bar(
        df,
        x="Rasio Pemisahan",
        y="Fitur",
        orientation="h",
        color_discrete_sequence=["#12B886"],
        text="Rasio Pemisahan",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    return _style(fig, height=320, title="Rasio Pemisahan Rata-rata Antar Klaster")


def numerical_mean_bars(data: pd.DataFrame, numerical_cols: list[str]) -> go.Figure:
    if not numerical_cols:
        return _empty_figure("Tidak ada fitur numerik.")
    means = data.groupby("Cluster")[numerical_cols].mean().reset_index()
    long_df = means.melt(id_vars="Cluster", var_name="Fitur", value_name="Rata-rata")
    long_df["Label"] = long_df["Cluster"].apply(lambda c: f"Cluster {c}")
    fig = px.bar(
        long_df,
        x="Label",
        y="Rata-rata",
        color="Label",
        facet_col="Fitur",
        facet_col_wrap=2,
        color_discrete_sequence=NEUTRAL_SEQUENCE,
        text="Rata-rata",
    )
    fig.update_traces(texttemplate="%{text:.3g}", textposition="outside")
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(showlegend=False)
    return _style(fig, height=420, title="Rata-rata Fitur Numerik per Klaster (satuan asli)")
