"""Input Data page: upload the four mandatory files, configure K and run the pipeline."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src.modeling import run_modeling
from src.preprocessing import (
    CONVERSION_REQUIRED,
    GEOLOCATION_REQUIRED,
    IMPORTS_REQUIRED,
    PROVINCES_REQUIRED,
    align_columns,
    read_table,
    run_preprocessing,
    validate_columns,
)

CAPPING_LABELS = {
    "usd_value": ("Nilai impor (USD)", "Per SITC"),
    "pdrb": ("PDRB", "Global"),
    "china_investment_us_ribu": ("Investasi Tiongkok", "Global"),
    "micro_small_industries": ("Jumlah industri mikro dan kecil", "Global"),
}


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _store_upload(kind: str, upload) -> None:
    content = upload.getvalue()
    signature = _sha256(content)
    key = f"{kind}_signature"
    if st.session_state.get(key) == signature:
        return
    try:
        df = read_table(content, upload.name)
    except Exception as exc:
        st.session_state[f"{kind}_valid"] = False
        st.session_state[f"{kind}_message"] = f"Berkas tidak dapat dibaca: {exc}"
        st.session_state[f"{kind}_bytes"] = content
        st.session_state[f"{kind}_name"] = upload.name
        st.session_state[f"{kind}_df"] = None
        st.session_state[key] = signature
        st.session_state["analysis_result"] = None
        st.session_state["analysis_signature"] = None
        return

    required = {
        "imports": IMPORTS_REQUIRED,
        "provinces": PROVINCES_REQUIRED,
        "conversion": CONVERSION_REQUIRED,
        "geolocation": GEOLOCATION_REQUIRED,
    }[kind]
    valid, message = validate_columns(df, required, upload.name)
    st.session_state[f"{kind}_bytes"] = content
    st.session_state[f"{kind}_name"] = upload.name
    st.session_state[f"{kind}_df"] = df
    st.session_state[f"{kind}_valid"] = valid
    st.session_state[f"{kind}_message"] = message
    st.session_state[key] = signature
    st.session_state["analysis_result"] = None
    st.session_state["analysis_signature"] = None


def _reset_input_data() -> None:
    n_clusters = st.session_state.get("n_clusters", 4)
    st.session_state.clear()
    st.session_state["n_clusters"] = n_clusters


def _current_signature() -> tuple:
    return (
        st.session_state.get("imports_signature"),
        st.session_state.get("provinces_signature"),
        st.session_state.get("conversion_signature"),
        st.session_state.get("geolocation_signature"),
        int(st.session_state.get("n_clusters", 4)),
    )


def _file_status_row(label: str, kind: str, required_cols: list[str]) -> None:
    name = st.session_state.get(f"{kind}_name")
    valid = st.session_state.get(f"{kind}_valid")
    message = st.session_state.get(f"{kind}_message")
    df = st.session_state.get(f"{kind}_df")

    if df is None:
        st.markdown(f"**{label}**: <span class='file-missing'>belum diunggah</span>", unsafe_allow_html=True)
    elif valid:
        st.markdown(
            f"**{label}**: <span class='file-ok'>valid</span> · `{name}` · {len(df):,} baris × "
            f"{len(df.columns)} kolom",
            unsafe_allow_html=True,
        )
        st.caption("📋 Kolom yang diverifikasi: " + " · ".join(f"`{c}`" for c in required_cols))
    else:
        st.markdown(
            f"**{label}**: <span class='file-missing'>tidak valid</span> · `{name}`",
            unsafe_allow_html=True,
        )
    if message:
        (st.caption if valid else st.error)(message)


def _render_summary(pre_report: dict, model_result: dict) -> None:
    clustered = model_result["clustered"]
    st.subheader("Ringkasan Hasil Pemodelan")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observasi Final", f"{pre_report['rows_final']:,}")
    c2.metric("Jumlah Pelabuhan", f"{clustered['port'].nunique():,}")
    c3.metric("Cakupan Provinsi", f"{clustered['Provinsi'].nunique():,}")
    c4.metric("Jumlah Klaster (K)", str(model_result["config"]["n_clusters"]))

    sizes = (
        clustered["Cluster"].value_counts().sort_index()
        .rename_axis("Cluster").reset_index(name="Jumlah")
    )
    sizes["Persentase"] = (sizes["Jumlah"] / len(clustered) * 100).round(1).astype(str) + "%"
    sizes = sizes.rename(columns={"Cluster": "Klaster", "Jumlah": "Jumlah Observasi", "Persentase": "Proporsi"})
    st.dataframe(sizes, width="stretch", hide_index=True)

    sil = model_result["silhouette"]
    sil_txt = "N/A" if pd.isna(sil) else f"{sil:.4f}"
    st.caption(
        f"📊 Skor Silhouette campuran: **{sil_txt}**"
        f" — dihitung pada {model_result['silhouette_n']:,} observasi.\n\n"
        f"📈 Fidelity model surrogate (Akurasi Random Forest):"
        f" **{model_result['surrogate']['metrics'].iloc[0]['Nilai']:.4f}**\n\n"
        f"🔍 Analisis SHAP: **{'✅ Tersedia' if model_result['shap'].get('available') else '⚠️ Tidak tersedia'}**"
    )

    with st.expander("Detail Ringkasan Tahapan Prapemrosesan"):
        conversion = pre_report["conversion"]
        st.markdown(
            f"- Baris data impor awal: **{pre_report['rows_imports']:,}**\n"
            f"- Kode HS-6 tidak terbaca/valid: **{conversion['rows_invalid_hs6']:,}**\n"
            f"- Transaksi tanpa pemetaan SITC: **{conversion['rows_unmapped']:,}** "
            f"(nilai transaksi: US$ {conversion['usd_value_unmapped']:,.2f})\n"
            f"- Baris setelah agregasi Pelabuhan × SITC: **{pre_report['rows_aggregated']:,}**\n"
            f"- Pelabuhan tanpa pemetaan data provinsi: **{pre_report['province']['rows_unmapped_port']:,}**\n"
            f"- Baris duplikat yang dibersihkan: **{pre_report['cleaning']['duplicates_removed']:,}**\n"
            f"- Pembatasan outlier (IQR capping): diterapkan pada seluruh fitur numerik "
            "(rincian di bawah)"
        )

        capping = pre_report["capping"]
        total_rows = max(int(pre_report["rows_final"]), 1)
        capping_table = pd.DataFrame(
            [
                {
                    "Fitur": label,
                    "Cakupan": scope,
                    "Jumlah Baris": int(capping.get(column, 0)),
                    "Persentase": (
                        f"{capping.get(column, 0) / total_rows * 100:.2f}%"
                    ),
                }
                for column, (label, scope) in CAPPING_LABELS.items()
            ]
        )
        st.markdown("**Pembatasan Outlier (IQR Capping) per Fitur Numerik**")
        st.dataframe(capping_table, width="stretch", hide_index=True)
        st.caption(
            f"Persentase dihitung terhadap {total_rows:,} baris dataset final. "
            "Jumlah menunjukkan baris observasi yang nilainya diubah, bukan jumlah nilai unik."
        )
        if "rows_any_capped" in capping:
            st.caption(
                f"Sebanyak {capping['rows_any_capped']:,} baris terpengaruh capping "
                "pada minimal satu fitur numerik."
            )


def render() -> None:
    st.title("Input Data & Konfigurasi")
    st.write(
        "Unggah keempat berkas data persyaratan di bawah ini. Tahapan prapemrosesan dan pemodelan "
        "akan siap dijalankan setelah seluruh berkas terverifikasi valid."
    )

    # ------------------------------------------------------------------
    # Uploads
    # ------------------------------------------------------------------
    header_col, reset_col = st.columns([4, 1], vertical_alignment="bottom")
    with header_col:
        st.subheader("1 · Berkas Data Input Wajib")
    with reset_col:
        has_session_data = any(
            st.session_state.get(key) is not None
            for key in (
                "imports_df",
                "provinces_df",
                "conversion_df",
                "geolocation_df",
                "analysis_result",
            )
        )
        st.button(
            "Reset Input Data",
            icon=":material/delete:",
            type="secondary",
            on_click=_reset_input_data,
            width="stretch",
            disabled=not has_session_data,
            help=(
                "Menghapus seluruh berkas input dan hasil analisis pada sesi ini. "
                "Nilai K tetap dipertahankan. Tindakan ini tidak dapat dibatalkan."
            ),
        )
    up1, up2 = st.columns(2)
    with up1:
        imports_upload = st.file_uploader(
            "port_imports.csv",
            type=["csv", "xlsx"],
            key="upload_imports",
            help="Data impor pelabuhan: port, hs_code, usd_value.",
        )
    with up2:
        provinces_upload = st.file_uploader(
            "provinces_data.csv",
            type=["csv", "xlsx"],
            key="upload_provinces",
            help="Indikator makro provinsi: Provinsi, pdrb, china_investment_us_ribu, micro_small_industries.",
        )

    up3, up4 = st.columns(2)
    with up3:
        conversion_upload = st.file_uploader(
            "HS2022_to_SITC_Conversion.xlsx",
            type=["xlsx", "csv"],
            key="upload_conversion",
            help="Tabel korespondensi konversi HS 2022 ke SITC Rev. 4 (From HS 2022, To SITC Rev. 4).",
        )
    with up4:
        geolocation_upload = st.file_uploader(
            "port_geolocation.csv",
            type=["csv", "xlsx"],
            key="upload_geolocation",
            help="Lokasi dan provinsi pelabuhan: port, Provinsi, latitude, longitude.",
        )

    if imports_upload is not None:
        _store_upload("imports", imports_upload)
    if provinces_upload is not None:
        _store_upload("provinces", provinces_upload)
    if conversion_upload is not None:
        _store_upload("conversion", conversion_upload)
    if geolocation_upload is not None:
        _store_upload("geolocation", geolocation_upload)

    st.markdown("##### Status Verifikasi Berkas")
    status_left, status_right = st.columns(2)
    with status_left:
        _file_status_row("Data Impor (port_imports)", "imports", IMPORTS_REQUIRED)
        _file_status_row("Data Provinsi (provinces_data)", "provinces", PROVINCES_REQUIRED)
    with status_right:
        _file_status_row("Tabel Konversi HS–SITC", "conversion", CONVERSION_REQUIRED)
        _file_status_row(
            "Data Geolokasi Pelabuhan (port_geolocation)",
            "geolocation",
            GEOLOCATION_REQUIRED,
        )

    all_valid = bool(
        st.session_state.get("imports_valid")
        and st.session_state.get("provinces_valid")
        and st.session_state.get("conversion_valid")
        and st.session_state.get("geolocation_valid")
    )

    # ------------------------------------------------------------------
    # Configuration + process
    # ------------------------------------------------------------------
    st.subheader("2 · Parameter & Konfigurasi Klasterisasi")
    st.session_state.setdefault("n_clusters", 4)
    st.slider(
        "Jumlah klaster (K)",
        min_value=2,
        max_value=12,
        step=1,
        key="n_clusters",
        help="Nilai awal K = 4 sesuai acuan eksperimen. Rentang pengujian klaster 2–12.",
    )
    st.caption(
        "⚙️ Fitur yang digunakan dalam pemodelan:\n\n"
        "**Kategorikal:** `sitc` (kelompok komoditas SITC Rev. 4).\n\n"
        "**Numerik:** `usd_value` (nilai impor, USD) · `pdrb` (PDRB provinsi) · "
        "`china_investment_us_ribu` (investasi Tiongkok, ribuan USD) · "
        "`micro_small_industries` (jumlah industri mikro & kecil).\n\n"
        "Kolom `port` dan `Provinsi` berfungsi sebagai identitas observasi "
        "dan tidak diikutsertakan dalam proses modelling."
    )

    missing = [
        label
        for label, kind in [
            ("port_imports", "imports"),
            ("provinces_data", "provinces"),
            ("tabel konversi", "conversion"),
            ("port_geolocation", "geolocation"),
        ]
        if not st.session_state.get(f"{kind}_valid")
    ]
    if missing:
        st.warning("Berkas berikut belum tersedia atau belum valid: " + ", ".join(missing) + ". Silakan periksa kembali berkas yang diunggah.")

    process_clicked = st.button(
        "Proses Data dan Jalankan Analisis",
        type="primary",
        width="stretch",
        disabled=not all_valid,
    )

    current_signature = _current_signature()

    if process_clicked and all_valid:
        st.session_state["processing_error"] = None
        st.session_state["analysis_result"] = None
        progress = st.progress(0, text="Menginisialisasi pipeline analisis...")
        try:
            with st.status("Menjalankan prapemrosesan dan pemodelan klaster...", expanded=True) as status:
                st.write("1/2 · Prapemrosesan: konversi HS–SITC, agregasi transaksi, pemetaan provinsi & geolokasi pelabuhan, dan IQR capping.")
                pre_result = run_preprocessing(
                    align_columns(st.session_state["imports_df"], IMPORTS_REQUIRED),
                    align_columns(st.session_state["provinces_df"], PROVINCES_REQUIRED),
                    align_columns(st.session_state["conversion_df"], CONVERSION_REQUIRED),
                    align_columns(st.session_state["geolocation_df"], GEOLOCATION_REQUIRED),
                )
                progress.progress(0.35, text="Tahap prapemrosesan data selesai.")

                st.write("2/2 · Pemodelan: K-Prototypes, evaluasi Elbow & Silhouette, model surrogate, dan analisis SHAP.")

                def _cb(fraction: float, text: str) -> None:
                    progress.progress(min(0.35 + 0.65 * fraction, 1.0), text=text)

                model_result = run_modeling(
                    pre_result["df_final"],
                    n_clusters=int(st.session_state["n_clusters"]),
                    progress_callback=_cb,
                )
                status.update(label="Seluruh proses pipeline berhasil diselesaikan.", state="complete")

            st.session_state["analysis_result"] = {
                "preprocessing": pre_result,
                "modeling": model_result,
            }
            st.session_state["analysis_signature"] = current_signature
            st.session_state["pipeline_summary"] = pre_result["report"]
            progress.progress(1.0, text="Pipeline selesai.")
        except Exception as exc:  # noqa: BLE001 - surface any pipeline failure
            st.session_state["processing_error"] = str(exc)
            st.session_state["analysis_result"] = None
            st.session_state["analysis_signature"] = None

    if st.session_state.get("processing_error"):
        st.error(f"Proses belum berhasil dijalankan: {st.session_state['processing_error']}")

    # ------------------------------------------------------------------
    # Results availability
    # ------------------------------------------------------------------
    result = st.session_state.get("analysis_result")
    result_signature = st.session_state.get("analysis_signature")
    is_current = result is not None and result_signature == _current_signature()

    if result is not None and not is_current:
        st.warning(
            "Terdeteksi perubahan data input atau nilai parameter K. Hasil analisis sebelumnya tidak lagi "
            "sesuai dan perlu diproses ulang."
        )

    if is_current:
        st.success("Prapemrosesan dan pemodelan berhasil diselesaikan! Hasil analisis siap untuk ditinjau.")
        _render_summary(result["preprocessing"]["report"], result["modeling"])
        if st.button("Lihat Hasil Analisis", type="primary", width="stretch"):
            st.switch_page("pages/3_Dashboard.py")

    st.divider()
    if st.button("← Kembali ke Beranda"):
        st.switch_page("pages/1_Beranda.py")


render()

