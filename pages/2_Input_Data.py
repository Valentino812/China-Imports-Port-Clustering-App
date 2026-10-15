"""Input Data page: upload the three mandatory files, configure K and run the pipeline."""

from __future__ import annotations

import hashlib
import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src.modeling import run_modeling
from src.preprocessing import (
    COMBINED_REQUIRED,
    CONVERSION_REQUIRED,
    PROVINCES_REQUIRED,
    align_columns,
    read_table,
    run_preprocessing,
    validate_columns,
)

PROJECT_DIR = Path(__file__).resolve().parents[2]
REFERENCE_CONVERSION = (
    PROJECT_DIR
    / "China-Imports-Port-Clustering-Ipynb"
    / "dataset"
    / "HS2022toSITC4ConversionAndCorrelationTables.xlsx"
)


@st.cache_data(show_spinner=False)
def _load_reference_conversion(path_str: str):
    path = Path(path_str)
    if not path.exists():
        return None, None
    content = path.read_bytes()
    df = pd.read_excel(BytesIO(content), dtype=str)
    return content, df


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
        st.session_state[f"{kind}_message"] = f"File tidak dapat dibaca: {exc}"
        st.session_state[f"{kind}_bytes"] = content
        st.session_state[f"{kind}_name"] = upload.name
        st.session_state[f"{kind}_df"] = None
        st.session_state[key] = signature
        st.session_state["analysis_result"] = None
        st.session_state["analysis_signature"] = None
        return

    required = {
        "combined": COMBINED_REQUIRED,
        "provinces": PROVINCES_REQUIRED,
        "conversion": CONVERSION_REQUIRED,
    }[kind]
    valid, message = validate_columns(df, required, upload.name)
    st.session_state[f"{kind}_bytes"] = content
    st.session_state[f"{kind}_name"] = upload.name
    st.session_state[f"{kind}_df"] = df
    st.session_state[f"{kind}_valid"] = valid
    st.session_state[f"{kind}_message"] = message
    st.session_state[key] = signature
    if kind == "conversion":
        st.session_state["conversion_source"] = "Unggahan pengguna"
    # Data berubah: hasil lama tidak lagi relevan.
    st.session_state["analysis_result"] = None
    st.session_state["analysis_signature"] = None


def _ensure_reference_conversion() -> None:
    if st.session_state.get("conversion_df") is not None:
        return
    content, df = _load_reference_conversion(str(REFERENCE_CONVERSION))
    if content is None or df is None:
        return
    valid, message = validate_columns(df, CONVERSION_REQUIRED, REFERENCE_CONVERSION.name)
    st.session_state["conversion_bytes"] = content
    st.session_state["conversion_name"] = REFERENCE_CONVERSION.name
    st.session_state["conversion_df"] = df
    st.session_state["conversion_valid"] = valid
    st.session_state["conversion_message"] = (
        message + " (dimuat otomatis dari direktori referensi project)"
    )
    st.session_state["conversion_source"] = "Direktori referensi project"
    st.session_state["conversion_signature"] = _sha256(content)


def _current_signature() -> tuple:
    return (
        st.session_state.get("combined_signature"),
        st.session_state.get("provinces_signature"),
        st.session_state.get("conversion_signature"),
        int(st.session_state.get("n_clusters", 4)),
    )


def _file_status_row(label: str, kind: str, required_cols: list[str]) -> None:
    name = st.session_state.get(f"{kind}_name")
    valid = st.session_state.get(f"{kind}_valid")
    message = st.session_state.get(f"{kind}_message")
    df = st.session_state.get(f"{kind}_df")

    if df is None:
        st.markdown(f"**{label}** â€” <span class='file-missing'>belum tersedia</span>", unsafe_allow_html=True)
    elif valid:
        st.markdown(
            f"**{label}** â€” <span class='file-ok'>valid</span> Â· `{name}` Â· {len(df):,} baris Ã— "
            f"{len(df.columns)} kolom",
            unsafe_allow_html=True,
        )
        st.caption("Kolom wajib: " + ", ".join(required_cols))
    else:
        st.markdown(
            f"**{label}** â€” <span class='file-missing'>tidak valid</span> Â· `{name}`",
            unsafe_allow_html=True,
        )
    if message:
        (st.caption if valid else st.error)(message)


def _render_summary(pre_report: dict, model_result: dict) -> None:
    clustered = model_result["clustered"]
    st.subheader("Ringkasan Hasil")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baris final", f"{pre_report['rows_final']:,}")
    c2.metric("Pelabuhan unik", f"{clustered['port'].nunique():,}")
    c3.metric("Provinsi", f"{clustered['Provinsi'].nunique():,}")
    c4.metric("Jumlah klaster", str(model_result["config"]["n_clusters"]))

    sizes = (
        clustered["Cluster"].value_counts().sort_index()
        .rename_axis("Cluster").reset_index(name="Jumlah")
    )
    sizes["Persentase"] = (sizes["Jumlah"] / len(clustered) * 100).round(1).astype(str) + "%"
    st.dataframe(sizes, width="stretch", hide_index=True)

    sil = model_result["silhouette"]
    sil_txt = "N/A" if pd.isna(sil) else f"{sil:.4f}"
    st.caption(
        f"Silhouette campuran: {sil_txt} pada {model_result['silhouette_n']:,} observasi Â· "
        f"Surrogate fidelity: "
        f"{model_result['surrogate']['metrics'].iloc[0]['Nilai']:.4f} Â· "
        f"SHAP: {'tersedia' if model_result['shap'].get('available') else 'tidak tersedia'}"
    )

    with st.expander("Ringkasan tahapan preprocessing"):
        conversion = pre_report["conversion"]
        st.markdown(
            f"- Baris impor awal: **{pre_report['rows_combined']:,}**\n"
            f"- HS6 tidak terbaca: **{conversion['rows_invalid_hs6']:,}**\n"
            f"- Baris tanpa pemetaan SITC: **{conversion['rows_unmapped']:,}** "
            f"(nilai USD {conversion['usd_value_unmapped']:,.2f})\n"
            f"- Baris setelah agregasi Port Ã— SITC: **{pre_report['rows_aggregated']:,}**\n"
            f"- Pelabuhan tanpa pemetaan provinsi: **{pre_report['province']['rows_unmapped_port']:,}**\n"
            f"- Duplikat dihapus: **{pre_report['cleaning']['duplicates_removed']:,}**\n"
            f"- IQR capping `usd_value` (per SITC): **{pre_report['capping']['usd_value']:,}** baris"
        )


def render() -> None:
    st.title("Input Data")
    st.write(
        "Unggah tiga file wajib di bawah ini. Preprocessing tidak akan dijalankan sebelum "
        "ketiganya tersedia dan tervalidasi."
    )

    _ensure_reference_conversion()

    # ------------------------------------------------------------------
    # Uploads
    # ------------------------------------------------------------------
    st.subheader("1 Â· File Input Wajib")
    up1, up2, up3 = st.columns(3)
    with up1:
        combined_upload = st.file_uploader(
            "combined_dataset.csv",
            type=["csv", "xlsx"],
            key="upload_combined",
            help="Data impor pelabuhan: port, hs_code, usd_value.",
        )
    with up2:
        provinces_upload = st.file_uploader(
            "provinces_data.csv",
            type=["csv", "xlsx"],
            key="upload_provinces",
            help="Indikator provinsi: Provinsi, pdrb, china_investment_us_ribu, micro_small_industries.",
        )
    with up3:
        conversion_upload = st.file_uploader(
            "HS2022toSITC4ConversionAndCorrelationTables.xlsx",
            type=["xlsx", "csv"],
            key="upload_conversion",
            help="Tabel konversi HS 2022 ke SITC Rev. 4 (From HS 2022, To SITC Rev. 4).",
        )

    if combined_upload is not None:
        _store_upload("combined", combined_upload)
    if provinces_upload is not None:
        _store_upload("provinces", provinces_upload)
    if conversion_upload is not None:
        _store_upload("conversion", conversion_upload)

    st.markdown("##### Status file")
    status_left, status_right = st.columns(2)
    with status_left:
        _file_status_row("Data impor (combined_dataset)", "combined", COMBINED_REQUIRED)
        _file_status_row("Data provinsi (provinces_data)", "provinces", PROVINCES_REQUIRED)
    with status_right:
        _file_status_row("Tabel konversi HSâ€“SITC", "conversion", CONVERSION_REQUIRED)
        source = st.session_state.get("conversion_source")
        if source:
            st.caption(f"Sumber tabel konversi: {source}.")

    all_valid = bool(
        st.session_state.get("combined_valid")
        and st.session_state.get("provinces_valid")
        and st.session_state.get("conversion_valid")
    )

    # ------------------------------------------------------------------
    # Configuration + process
    # ------------------------------------------------------------------
    st.subheader("2 Â· Konfigurasi Klasterisasi")
    st.slider(
        "Jumlah klaster K",
        min_value=2,
        max_value=7,
        value=int(st.session_state.get("n_clusters", 4)),
        step=1,
        key="n_clusters",
        help="Nilai awal K = 4 sesuai notebook. Rentang valid 2â€“7.",
    )
    st.caption(
        "Fitur model: `sitc` (kategorikal) dan `usd_value`, `pdrb`, `china_investment_us_ribu`, "
        "`micro_small_industries` (numerik). `port` dan `Provinsi` hanya sebagai metadata."
    )

    missing = [
        label
        for label, kind in [
            ("combined_dataset", "combined"),
            ("provinces_data", "provinces"),
            ("tabel konversi", "conversion"),
        ]
        if not st.session_state.get(f"{kind}_valid")
    ]
    if missing:
        st.warning("File berikut belum tersedia/valid: " + ", ".join(missing) + ".")

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
        progress = st.progress(0, text="Memulai pipeline...")
        try:
            with st.status("Menjalankan preprocessing dan pemodelan...", expanded=True) as status:
                st.write("1/2 Â· Preprocessing: konversi HSâ€“SITC, agregasi, pemetaan provinsi, IQR capping.")
                pre_result = run_preprocessing(
                    align_columns(st.session_state["combined_df"], COMBINED_REQUIRED),
                    align_columns(st.session_state["provinces_df"], PROVINCES_REQUIRED),
                    align_columns(st.session_state["conversion_df"], CONVERSION_REQUIRED),
                )
                progress.progress(0.35, text="Preprocessing selesai.")

                st.write("2/2 Â· Pemodelan: K-Prototypes, Elbow, Silhouette, surrogate, SHAP.")

                def _cb(fraction: float, text: str) -> None:
                    progress.progress(min(0.35 + 0.65 * fraction, 1.0), text=text)

                model_result = run_modeling(
                    pre_result["df_final"],
                    n_clusters=int(st.session_state["n_clusters"]),
                    progress_callback=_cb,
                )
                status.update(label="Pipeline selesai.", state="complete")

            st.session_state["analysis_result"] = {
                "preprocessing": pre_result,
                "modeling": model_result,
            }
            st.session_state["analysis_signature"] = current_signature
            st.session_state["pipeline_summary"] = pre_result["report"]
            progress.progress(1.0, text="Selesai.")
        except Exception as exc:  # noqa: BLE001 - surface any pipeline failure
            st.session_state["processing_error"] = str(exc)
            st.session_state["analysis_result"] = None
            st.session_state["analysis_signature"] = None

    if st.session_state.get("processing_error"):
        st.error(f"Proses belum berhasil: {st.session_state['processing_error']}")

    # ------------------------------------------------------------------
    # Results availability
    # ------------------------------------------------------------------
    result = st.session_state.get("analysis_result")
    result_signature = st.session_state.get("analysis_signature")
    is_current = result is not None and result_signature == _current_signature()

    if result is not None and not is_current:
        st.warning(
            "Data atau konfigurasi K berubah. Hasil analisis sebelumnya dibatalkan dan perlu "
            "diproses ulang."
        )

    if is_current:
        st.success("Preprocessing dan pemodelan berhasil. Hasil siap ditinjau.")
        _render_summary(result["preprocessing"]["report"], result["modeling"])
        if st.button("Lihat Hasil Analisis  â†’", type="primary", width="stretch"):
            st.switch_page("pages/3_Dashboard.py")

    st.divider()
    if st.button("â† Kembali ke Beranda"):
        st.switch_page("pages/1_Beranda.py")


render()
