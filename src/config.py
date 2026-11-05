"""Shared configuration constants for the application.

Keep the meaning of the cluster-count constants explicit:

* ``DEFAULT_K`` is the initial slider value used when the user starts an
  experiment.
* ``OPTIMAL_K`` is the evaluated optimum selected by the automatic option.

The two values are intentionally different.

The narrative thresholds are centralised here so the Dashboard narrative and
its tests share one source of truth.
"""

from __future__ import annotations

K_MIN = 2
K_MAX = 8
K_RANGE = range(K_MIN, K_MAX + 1)

DEFAULT_K = 4
OPTIMAL_K = 5

# ---------------------------------------------------------------------------
# Narrative analysis thresholds
# ---------------------------------------------------------------------------
# Numeric comparison (Z-score of the cluster mean against the dataset mean).
Z_SCORE_HIGH = 1.0
Z_SCORE_MODERATE = 0.5
Z_SCORE_LOW = -0.5
Z_SCORE_VERY_LOW = -1.0

# Categorical distribution (normalised Shannon entropy, cluster-local m).
ENTROPY_CONCENTRATED_MAX = 0.50
ENTROPY_BALANCED_MIN = 0.80

# Surrogate/SHAP safeguards.
SHAP_IMPORTANCE_THRESHOLD = 0.25
SURROGATE_FIDELITY_MIN = 0.80
SURROGATE_CLASS_RECALL_MIN = 0.70
SHAP_MIN_TEST_SAMPLES = 10

# Geographic analysis.
GEO_COVERAGE_MIN = 0.80
GEO_DOMINANCE_THRESHOLD = 0.80

# Floating-point tolerance when detecting tied cluster maxima and 100% shares.
CLUSTER_MAX_RTOL = 1e-9
SHARE_FULL_TOLERANCE = 1e-9

# Static port-map extent. The map keeps its auto-fit spirit but never zooms in
# closer than this minimum geographical window (kilometres). Adjust after a
# visual check without touching the plotting code.
MAP_MIN_EXTENT_KM = 350.0
MAP_EXTENT_PADDING = 0.12

# Display limits for the narrative (not analytical thresholds).
NARRATIVE_TOP_PROVINCES = 3
NARRATIVE_TOP_REGIONS = 3

# ---------------------------------------------------------------------------
# SITC Rev. 4 first-digit descriptions
# ---------------------------------------------------------------------------
# Single source of truth used by the Dashboard legends and the narrative module.
SITC_LABELS = {
    "0": "Bahan makanan dan hewan hidup",
    "1": "Minuman dan tembakau",
    "2": "Bahan mentah, tidak dapat dimakan, kecuali bahan bakar",
    "3": "Bahan bakar mineral, pelumas, dan bahan terkait",
    "4": "Minyak, lemak, dan lilin hewani maupun nabati",
    "5": "Bahan kimia dan produk terkait",
    "6": "Barang manufaktur yang diklasifikasikan terutama berdasarkan bahan bakunya",
    "7": "Mesin dan alat transportasi",
    "8": "Barang manufaktur lainnya",
    "9": "Komoditas dan transaksi yang tidak diklasifikasikan di tempat lain",
}

