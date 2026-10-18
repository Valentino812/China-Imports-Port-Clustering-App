"""Beranda (homepage) page.

Displays the research title and application description. No preprocessing,
modelling or evaluation runs on this page.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st


def render() -> None:
    st.markdown(
        """
        <section class="hero-panel">
          <div class="hero-eyebrow">ANALISIS KLASTERING PELABUHAN DAN KELOMPOK KOMODITAS IMPOR TIONGKOK</div>
          <div class="hero-title">PENGELOMPOKAN KOMBINASI PELABUHAN DAN KELOMPOK KOMODITAS DI INDONESIA BERDASARKAN KARAKTERISTIK IMPOR TIONGKOK SERTA INDIKATOR MAKROEKONOMI</div>
          <div class="hero-copy">
            Aplikasi analitik pemetaan dan pengelompokan (klasterisasi) kombinasi pelabuhan dan kelompok komoditas di 
            Indonesia berdasarkan karakteristik aktivitas impor dari Tiongkok serta kondisi makroekonomi dari masing-masing 
            wilayah. Sistem yang dikembangkan adalah dashboard analitik berbasis web yang memungkinkan pengguna memperoleh 
            gambaran terhadap karakteristik, pola, dan perbedaan antarwilayah melalui visualisasi hasil klasterisasi yang 
            interaktif. Selain menghasilkan pemetaan wilayah, sistem juga mendukung proses identifikasi profil setiap kelompok 
            secara objektif dan terukur berdasarkan sejumlah indikator ekonomi dan logistik yang digunakan dalam analisis.
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="section-card">
              <h4>01 · Input Data</h4>
              <p>Unggah empat berkas data wajib: data transaksi impor, data makro provinsi, tabel konversi HS–SITC,
              dan data geolokasi pelabuhan, lalu tentukan jumlah klaster yang diinginkan.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="section-card">
              <h4>02 · Proses Klasterisasi</h4>
              <p>Prapemrosesan otomatis, pemodelan K-Prototypes, evaluasi Elbow &amp; Silhouette campuran, hingga
              pelatihan model surrogate Random Forest dan analisis SHAP.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="section-card">
              <h4>03 · Eksplorasi Hasil</h4>
              <p>Telaah wawasan mendalam melalui menu Ringkasan, Profil Klaster, Evaluasi Model, Distribusi Spasial,
              dan Unduhan Data pada dashboard interaktif.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    _spacer_left, center, _spacer_right = st.columns([1, 1.4, 1])
    with center:
        if st.button("Mulai Analisis →", type="primary", width="stretch"):
            st.switch_page("pages/2_Input_Data.py")
    st.caption(
        "Aplikasi dirancang modular: pemrosesan data, validasi, dan komputasi pemodelan dilakukan "
        "secara komprehensif pada halaman Input Data."
    )


render()

