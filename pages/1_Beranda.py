"""Beranda (homepage) page.

Displays the research title and application description. No preprocessing,
modelling or evaluation runs on this page.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

HOME_CSS = """
<style>
.st-key-manual_book_download {
    margin-top: 1.25rem;
}
.st-key-download_manual_book {
    width: 100% !important;
}
.st-key-manual_book_download [data-testid='stDownloadButton'] {
    display: flex;
    justify-content: center;
    width: 100%;
}
.st-key-manual_book_download [data-testid='stDownloadButton'] button {
    width: auto !important;
    max-width: 100%;
    min-height: 50px;
    padding: 13px 30px !important;
    font-size: 15.5px !important;
    gap: 10px;
}
.st-key-manual_book_download [data-testid='stDownloadButton'] button svg,
.st-key-manual_book_download [data-testid='stDownloadButton'] button [data-testid='stIconMaterial'] {
    width: 20px !important;
    height: 20px !important;
    font-size: 20px !important;
}
.manual-book-caption {
    margin-top: 6px;
    text-align: center;
    color: #73798B;
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 12.5px;
    line-height: 1.4;
}
</style>
"""


@st.cache_data(show_spinner=False)
def _manual_book_bytes(path: str, modified_at: float) -> bytes:
    return Path(path).read_bytes()


def render() -> None:
    st.markdown(
        """
        <section class="hero-panel">
          <div class="hero-title">KLASTERISASI KOMBINASI PELABUHAN DAN KELOMPOK KOMODITAS DI INDONESIA BERDASARKAN KARAKTERISTIK IMPOR TIONGKOK DENGAN MENGGUNAKAN METODE K-PROTOTYPES</div>
          <div class="hero-subtitle">Analitik · K-Prototypes · Pelabuhan Indonesia</div>
          <div class="hero-copy">
            Aplikasi analitik pemetaan dan pengelompokan (klasterisasi) kombinasi pelabuhan dan kelompok komoditas di 
            Indonesia berdasarkan karakteristik aktivitas impor dari Tiongkok serta kondisi makroekonomi dari masing-masing 
            wilayah pelabuhan. Sistem yang dikembangkan adalah dashboard analitik berbasis web yang dapat memberikan 
            gambaran terhadap karakteristik, pola, dan perbedaan antarwilayah melalui visualisasi hasil klasterisasi 
            interaktif. Selain itu, sistem juga mendukung proses identifikasi profil setiap kelompok secara objektif dan 
            terukur berdasarkan sejumlah indikator ekonomi dan logistik yang digunakan dalam analisis.
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="guide-banner">
          <span class="guide-banner-icon">📖</span> Ringkasan Panduan Pengguna
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="section-card">
              <h4>01 · Input Data</h4>
              <p>Buka halaman <strong>Input Data</strong>, lalu unggah empat berkas wajib secara berurutan:
              data transaksi impor pelabuhan, data makroekonomi provinsi, tabel konversi HS 2022 ke SITC Rev.&nbsp;4,
              dan data geolokasi pelabuhan. Setelah keempat berkas terverifikasi valid, atur jumlah klaster (K)
              menggunakan slider yang tersedia atau gunakan fitur pencarian K optimal berbasis Silhouette, kemudian
              tekan tombol <strong>Proses Data dan Jalankan Analisis</strong> untuk memulai pipeline.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="section-card">
              <h4>02 · Proses Klasterisasi</h4>
              <p>Seluruh tahapan klasterisasi dilakukan secara otomatis oleh aplikasi. Pipeline mencakup
              prapemrosesan data (konversi kode HS–SITC, agregasi transaksi, penggabungan data provinsi,
              dan pembatasan outlier), pemodelan klaster menggunakan algoritma K-Prototypes, evaluasi kualitas
              klaster melalui metode Elbow dan Silhouette, hingga pelatihan model surrogate Random Forest
              beserta analisis SHAP untuk interpretasi kontribusi fitur.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="section-card">
              <h4>03 · Eksplorasi Hasil</h4>
              <p>Setelah pipeline selesai, buka halaman <strong>Dashboard</strong> untuk mengeksplorasi hasil melalui
              lima menu: <strong>Ringkasan</strong> (proporsi dan analisis deskriptif klaster),
              <strong>Profil Klaster</strong> (statistik numerik, distribusi SITC, dan nilai SHAP per klaster),
              <strong>Evaluasi</strong> (kurva Elbow, skor Silhouette, dan fidelity model surrogate),
              <strong>Peta</strong> (distribusi spasial pelabuhan), serta
              <strong>Data</strong> (pratinjau dan ekspor dataset setiap tahap pipeline).</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(HOME_CSS, unsafe_allow_html=True)
    manual_book_path = Path(__file__).resolve().parents[1] / "Manual Book.pdf"
    with st.container(key="manual_book_download"):
        if manual_book_path.exists():
            st.download_button(
                "Unduh Panduan Pengguna Lengkap",
                data=_manual_book_bytes(
                    str(manual_book_path),
                    manual_book_path.stat().st_mtime,
                ),
                file_name="Manual_Book_China_Imports_Port_Clustering.pdf",
                mime="application/pdf",
                icon=":material/download:",
                icon_position="left",
                type="primary",
                width="content",
                key="download_manual_book",
            )
            st.markdown(
                '<div class="manual-book-caption">PDF Manual Book</div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning(
                "File Manual Book.pdf tidak ditemukan di folder root project. "
                "Letakkan berkas tersebut di samping app.py lalu muat ulang halaman."
            )

    st.divider()
    _spacer_left, center, _spacer_right = st.columns([1, 1.4, 1])
    with center:
        if st.button("Mulai Analisis", type="primary", width="stretch"):
            st.switch_page("pages/2_Input_Data.py")
    st.caption(
        "🔒 Seluruh data yang ada pada aplikasi diproses secara lokal sehingga  tidak ada informasi yang "
        "dikirim ke server eksternal. Unggah keempat berkas wajib, atur jumlah klaster, lalu jalankan pipeline "
        "otomatis yang mencakup prapemrosesan, pemodelan K-Prototypes, evaluasi Silhouette, model surrogate "
        "Random Forest, dan analisis SHAP yang dapat dijalankan dengan tombol Mulai Analisis.  "
    )


render()

