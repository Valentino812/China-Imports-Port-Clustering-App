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
          <div class="hero-eyebrow">PORT &amp; IMPORT CLUSTERING ANALYTICS</div>
          <div class="hero-title">Analisis Klasterisasi Pelabuhan Indonesia Berdasarkan Impor dari Tiongkok</div>
          <div class="hero-copy">
            Aplikasi ini mengelompokkan pelabuhan Indonesia berdasarkan karakteristik impor dari
            Tiongkok dan indikator makroekonomi provinsi. Proses mencakup konversi HS 2022 ke
            SITC Rev. 4, pemodelan K-Prototypes, evaluasi campuran, serta penjelasan SHAP atas
            model surrogate.
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
              <h4>01 Â· Input Data</h4>
              <p>Unggah tiga file wajib: data impor, data provinsi, dan tabel konversi HSâ€“SITC,
              lalu tentukan jumlah klaster.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="section-card">
              <h4>02 Â· Proses Klasterisasi</h4>
              <p>Preprocessing, K-Prototypes, Elbow Method, Silhouette campuran, surrogate
              Random Forest, dan SHAP dijalankan secara berurutan.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="section-card">
              <h4>03 Â· Analisis Hasil</h4>
              <p>Jelajahi Ringkasan, Profil Klaster, Evaluasi, Peta, dan Data melalui dashboard
              interaktif dengan sidebar konsisten.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    _spacer_left, center, _spacer_right = st.columns([1, 1.4, 1])
    with center:
        if st.button("Mulai Analisis  â†’", type="primary", width="stretch"):
            st.switch_page("pages/2_Input_Data.py")
    st.caption(
        "Aplikasi ini tidak menjalankan preprocessing, pemodelan, atau evaluasi ketika halaman "
        "ini dibuka. Hasil analisis hanya dibuat pada halaman Input Data."
    )


render()
