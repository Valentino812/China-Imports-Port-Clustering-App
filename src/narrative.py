"""Narrative description of cluster characteristics for the Dashboard.

The module turns the results already stored by :func:`src.modeling.run_modeling`
into descriptive Indonesian text. It never re-runs preprocessing,
K-Prototypes, the surrogate Random Forest, or SHAP: every value is read from
the ``analysis_result`` kept in ``st.session_state``.

Responsibilities are split in two layers:

* ``analyze_*`` functions compute statistics and classifications and return
  plain dictionaries;
* ``generate_*`` functions turn those dictionaries into sentences.

Thresholds live in :mod:`src.config` so the narrative and its tests share one
source of truth.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    CLUSTER_MAX_RTOL,
    ENTROPY_BALANCED_MIN,
    ENTROPY_CONCENTRATED_MAX,
    GEO_COVERAGE_MIN,
    GEO_DOMINANCE_THRESHOLD,
    NARRATIVE_TOP_PROVINCES,
    NARRATIVE_TOP_REGIONS,
    SHAP_IMPORTANCE_THRESHOLD,
    SHAP_MIN_TEST_SAMPLES,
    SHARE_FULL_TOLERANCE,
    SITC_LABELS,
    SURROGATE_CLASS_RECALL_MIN,
    SURROGATE_FIDELITY_MIN,
    Z_SCORE_HIGH,
    Z_SCORE_LOW,
    Z_SCORE_MODERATE,
    Z_SCORE_VERY_LOW,
)
from src.preprocessing import clean_province

# ---------------------------------------------------------------------------
# Feature vocabulary
# ---------------------------------------------------------------------------
# ``label`` is used as a heading; ``sentence_label`` keeps the sentence natural.
NUMERIC_FEATURES = [
    {
        "feature": "usd_value",
        "label": "Nilai Impor",
        "sentence_label": "nilai impor",
        "unit": "USD",
        "number_format": ",.2f",
    },
    {
        "feature": "pdrb",
        "label": "PDRB",
        "sentence_label": "PDRB",
        "unit": "Miliar Rupiah",
        "number_format": ",.2f",
    },
    {
        "feature": "china_investment_us_ribu",
        "label": "Investasi Tiongkok",
        "sentence_label": "investasi Tiongkok",
        "unit": "Ribuan USD",
        "number_format": ",.2f",
    },
    {
        "feature": "micro_small_industries",
        "label": "Industri Mikro dan Kecil",
        "sentence_label": "jumlah industri mikro dan kecil",
        "unit": "unit usaha",
        "number_format": ",.0f",
    },
]
NUMERIC_FEATURE_NAMES = [item["feature"] for item in NUMERIC_FEATURES]

# ---------------------------------------------------------------------------
# Province -> main island region
# ---------------------------------------------------------------------------
# All 38 provinces used by ``dataset_input/provinces_data.csv``. Names follow
# ``src.preprocessing.clean_province`` (title case, "Dki Jakarta",
# "Di Yogyakarta", "Kepulauan ...").
PROVINCE_TO_REGION = {
    # Sumatera
    "Aceh": "Sumatera",
    "Sumatera Utara": "Sumatera",
    "Sumatera Barat": "Sumatera",
    "Riau": "Sumatera",
    "Jambi": "Sumatera",
    "Sumatera Selatan": "Sumatera",
    "Bengkulu": "Sumatera",
    "Lampung": "Sumatera",
    "Kepulauan Bangka Belitung": "Sumatera",
    "Kepulauan Riau": "Sumatera",
    # Jawa
    "Dki Jakarta": "Jawa",
    "Jawa Barat": "Jawa",
    "Jawa Tengah": "Jawa",
    "Di Yogyakarta": "Jawa",
    "Jawa Timur": "Jawa",
    "Banten": "Jawa",
    # Kalimantan
    "Kalimantan Barat": "Kalimantan",
    "Kalimantan Tengah": "Kalimantan",
    "Kalimantan Selatan": "Kalimantan",
    "Kalimantan Timur": "Kalimantan",
    "Kalimantan Utara": "Kalimantan",
    # Sulawesi
    "Sulawesi Utara": "Sulawesi",
    "Sulawesi Tengah": "Sulawesi",
    "Sulawesi Selatan": "Sulawesi",
    "Sulawesi Tenggara": "Sulawesi",
    "Gorontalo": "Sulawesi",
    "Sulawesi Barat": "Sulawesi",
    # Bali dan Nusa Tenggara
    "Bali": "Bali dan Nusa Tenggara",
    "Nusa Tenggara Barat": "Bali dan Nusa Tenggara",
    "Nusa Tenggara Timur": "Bali dan Nusa Tenggara",
    # Maluku
    "Maluku": "Maluku",
    "Maluku Utara": "Maluku",
    # Papua
    "Papua": "Papua",
    "Papua Barat": "Papua",
    "Papua Barat Daya": "Papua",
    "Papua Selatan": "Papua",
    "Papua Tengah": "Papua",
    "Papua Pegunungan": "Papua",
}

COMPARISON_SENTENCES = {
    "far_above": "Nilai ini jauh di atas rata-rata keseluruhan dataset.",
    "above": "Nilai ini berada di atas rata-rata keseluruhan dataset.",
    "near": "Nilai ini mendekati rata-rata keseluruhan dataset.",
    "below": "Nilai ini berada di bawah rata-rata keseluruhan dataset.",
    "far_below": "Nilai ini jauh di bawah rata-rata keseluruhan dataset.",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _is_close(value: float, reference: float, tolerance: float = CLUSTER_MAX_RTOL) -> bool:
    """Compare two floats allowing for floating-point noise."""
    if value is None or reference is None:
        return False
    if not (np.isfinite(value) and np.isfinite(reference)):
        return False
    return abs(value - reference) <= tolerance * max(1.0, abs(reference))


def _is_full(share: float) -> bool:
    return bool(np.isfinite(share) and share >= 1.0 - SHARE_FULL_TOLERANCE)


def _sitc_sort_key(code: object) -> tuple[int, str]:
    text = str(code)
    return (0, text.zfill(2)) if text.isdigit() else (1, text)


def _sitc_description(code: object) -> str:
    text = str(code)
    return SITC_LABELS.get(text, f"kelompok komoditas SITC {text}")


def _lower_first(text: str) -> str:
    """Lower-case the first character when a description is embedded in a sentence."""
    if not text:
        return text
    return text[:1].lower() + text[1:]


def _sitc_list(codes: list[str]) -> str:
    formatted = [f"SITC {code}" for code in codes]
    if not formatted:
        return ""
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]} dan {formatted[1]}"
    return f"{', '.join(formatted[:-1])}, dan {formatted[-1]}"


def _indonesian_count(count: int) -> str:
    words = {2: "dua", 3: "tiga", 4: "empat", 5: "lima"}
    return words.get(count, str(count))


def _format_number(value: float | None, number_format: str) -> str:
    if value is None or not np.isfinite(value):
        return "tidak tersedia"
    return format(float(value), number_format)


def _format_percentage(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "tidak tersedia"
    if _is_close(value, round(value), tolerance=1e-6):
        return f"{round(value):.0f}%"
    return f"{value:.1f}%"


def _canonical_province(value: object) -> str | None:
    """Return the title-cased province name, or ``None`` when unusable."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = clean_province(value)
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None
    text = " ".join(str(text).split())
    if not text or text.lower() in {"nan", "none"}:
        return None
    return text


def _classify_z_score(z_score: float | None) -> str:
    """Map a Z-score to the documented comparison buckets."""
    if z_score is None or not np.isfinite(z_score):
        return "unavailable"
    if z_score >= Z_SCORE_HIGH:
        return "far_above"
    if z_score >= Z_SCORE_MODERATE:
        return "above"
    if z_score > Z_SCORE_LOW:
        return "near"
    if z_score > Z_SCORE_VERY_LOW:
        return "below"
    return "far_below"


def _extract_fidelity(surrogate: dict[str, Any]) -> float | None:
    metrics = surrogate.get("metrics")
    if isinstance(metrics, pd.DataFrame) and {"Metrik", "Nilai"}.issubset(metrics.columns):
        rows = metrics.loc[metrics["Metrik"] == "Accuracy / fidelity", "Nilai"]
        if len(rows):
            try:
                return float(rows.iloc[0])
            except (TypeError, ValueError):
                pass
    report = surrogate.get("classification_report") or {}
    accuracy = report.get("accuracy")
    return float(accuracy) if accuracy is not None else None


def _extract_recall(surrogate: dict[str, Any], cluster: int) -> float | None:
    report = surrogate.get("classification_report") or {}
    for key in (cluster, str(cluster)):
        entry = report.get(key)
        if isinstance(entry, dict) and "recall" in entry:
            try:
                return float(entry["recall"])
            except (TypeError, ValueError):
                return None
    return None


# ---------------------------------------------------------------------------
# SHAP validity and importance
# ---------------------------------------------------------------------------

def check_shap_validity(model_result: dict[str, Any]) -> dict[str, Any]:
    """Check fidelity, recall and test-sample gates for SHAP claims."""
    result: dict[str, Any] = {
        "available": False,
        "fidelity": None,
        "reason": None,
        "classes": [],
        "clusters": {},
    }

    shap = model_result.get("shap") or {}
    surrogate = model_result.get("surrogate") or {}

    if not shap.get("available"):
        result["reason"] = shap.get("reason", "Hasil SHAP tidak tersedia.")
        return result
    for key in ("shap_by_class", "X_eval"):
        if key not in shap:
            result["reason"] = f"Struktur hasil SHAP tidak lengkap: {key} tidak ditemukan."
            return result
    if not isinstance(shap.get("X_eval"), pd.DataFrame):
        result["reason"] = "Data evaluasi SHAP tidak dapat dipetakan."
        return result

    fidelity = _extract_fidelity(surrogate)
    result["fidelity"] = fidelity
    if fidelity is None:
        result["reason"] = "Fidelity surrogate tidak dapat dibaca dari hasil modeling."
        return result
    if fidelity < SURROGATE_FIDELITY_MIN:
        result["reason"] = (
            f"Fidelity surrogate ({fidelity:.4f}) berada di bawah ambang "
            f"{SURROGATE_FIDELITY_MIN:.2f}; interpretasi SHAP tidak digunakan."
        )
        return result

    classes = [int(c) for c in shap.get("classes", [])]
    model_classes = [int(c) for c in surrogate.get("class_labels", classes)]
    if classes and model_classes and sorted(classes) != sorted(model_classes):
        result["reason"] = "Pemetaan kelas SHAP tidak sesuai dengan kelas surrogate."
        return result
    result["classes"] = classes

    y_test = surrogate.get("y_test")
    shap_by_class = shap.get("shap_by_class") or {}
    if not isinstance(y_test, pd.Series):
        result["reason"] = "Label aktual data uji tidak tersedia."
        return result

    per_cluster: dict[int, dict[str, Any]] = {}
    for cluster in sorted(int(c) for c in shap_by_class):
        reasons: list[str] = []
        n_test = int((y_test == cluster).sum())
        recall = _extract_recall(surrogate, cluster)
        if cluster not in classes:
            reasons.append("Kelas klaster tidak ditemukan pada model surrogate.")
        if n_test < SHAP_MIN_TEST_SAMPLES:
            reasons.append(
                f"Hanya {n_test} observasi uji untuk klaster ini "
                f"(minimal {SHAP_MIN_TEST_SAMPLES})."
            )
        if recall is None:
            reasons.append("Recall surrogate kelas ini tidak dapat dibaca.")
        elif recall < SURROGATE_CLASS_RECALL_MIN:
            reasons.append(
                f"Recall surrogate {recall:.4f} di bawah ambang "
                f"{SURROGATE_CLASS_RECALL_MIN:.2f}."
            )
        per_cluster[cluster] = {
            "valid": not reasons,
            "n_test": n_test,
            "recall": recall,
            "reason": " ".join(reasons) if reasons else None,
        }

    result["available"] = True
    result["clusters"] = per_cluster
    return result


def analyze_shap_importance(model_result: dict[str, Any]) -> dict[str, Any]:
    """Mean absolute SHAP per cluster and normalised feature shares.

    SITC one-hot columns are grouped into the single ``sitc`` feature by
    summing their absolute contributions per observation. Only test rows whose
    actual K-Prototypes label equals the analysed cluster are used, and the
    SHAP values always come from that cluster's own class output.
    """
    validity = check_shap_validity(model_result)
    result: dict[str, Any] = {
        "available": False,
        "fidelity": validity.get("fidelity"),
        "validity": validity,
        "reason": validity.get("reason"),
        "shares": {},
        "mean_absolute": {},
        "is_important": {},
    }
    if not validity["available"]:
        return result

    shap = model_result["shap"]
    surrogate = model_result["surrogate"]
    X_eval: pd.DataFrame = shap["X_eval"]
    shap_by_class = shap["shap_by_class"] or {}
    y_test: pd.Series = surrogate["y_test"]

    y_aligned = y_test.reindex(X_eval.index)
    if y_aligned.isna().any():
        result["reason"] = "Label aktual data uji tidak sejajar dengan baris matriks SHAP."
        return result

    sitc_positions = [
        X_eval.columns.get_loc(column)
        for column in X_eval.columns
        if str(column).startswith("SITC_")
    ]
    if not sitc_positions:
        result["reason"] = "Kolom one-hot SITC tidak ditemukan pada matriks SHAP."
        return result

    numeric_positions = {
        feature: X_eval.columns.get_loc(feature)
        for feature in NUMERIC_FEATURE_NAMES
        if feature in X_eval.columns
    }
    if len(numeric_positions) != len(NUMERIC_FEATURE_NAMES):
        result["reason"] = "Sebagian fitur numerik tidak ditemukan pada matriks SHAP."
        return result

    labels = y_aligned.to_numpy()
    for cluster, contributions in sorted(shap_by_class.items()):
        cluster = int(cluster)
        cluster_validity = validity["clusters"].get(cluster) or {}
        if not cluster_validity.get("valid"):
            continue
        array = np.asarray(contributions, dtype=float)
        if (
            array.ndim != 2
            or array.shape[0] != len(X_eval)
            or array.shape[1] != X_eval.shape[1]
        ):
            continue

        mask = labels == cluster
        if int(mask.sum()) < SHAP_MIN_TEST_SAMPLES:
            continue

        absolute = np.abs(array[mask])
        values = {"sitc": float(absolute[:, sitc_positions].sum(axis=1).mean())}
        for feature, position in numeric_positions.items():
            values[feature] = float(absolute[:, position].mean())

        total = float(sum(values.values()))
        if not np.isfinite(total) or total <= 0.0:
            continue

        shares = {feature: value / total for feature, value in values.items()}
        result["mean_absolute"][cluster] = values
        result["shares"][cluster] = shares
        result["is_important"][cluster] = {
            feature: share >= SHAP_IMPORTANCE_THRESHOLD
            for feature, share in shares.items()
        }

    result["available"] = bool(result["shares"])
    if not result["available"] and not result["reason"]:
        result["reason"] = "Tidak ada klaster yang memenuhi syarat interpretasi SHAP."
    return result


# ---------------------------------------------------------------------------
# Numeric profile
# ---------------------------------------------------------------------------

def analyze_numeric_profile(
    clustered: pd.DataFrame,
    shap_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Per-cluster means, Z-score buckets, maxima and SHAP shares."""
    analysis: dict[str, Any] = {
        "clusters": [],
        "global_mean": {},
        "global_std": {},
        "per_cluster": {},
    }
    if clustered is None or clustered.empty or "Cluster" not in clustered.columns:
        return analysis

    available_features = [f for f in NUMERIC_FEATURE_NAMES if f in clustered.columns]
    numeric = clustered[["Cluster", *available_features]].copy()
    for feature in available_features:
        numeric[feature] = pd.to_numeric(numeric[feature], errors="coerce")

    clusters = sorted(int(c) for c in numeric["Cluster"].dropna().unique())
    analysis["clusters"] = clusters

    maxima: dict[str, tuple[float, int] | None] = {}
    for feature in available_features:
        series = numeric[feature]
        analysis["global_mean"][feature] = (
            float(series.mean()) if series.notna().any() else None
        )
        analysis["global_std"][feature] = (
            float(series.std(ddof=0)) if series.notna().sum() > 1 else None
        )
        feature_means = numeric.groupby("Cluster")[feature].mean().dropna()
        if feature_means.empty:
            maxima[feature] = None
            continue
        max_value = float(feature_means.max())
        max_count = int(
            sum(_is_close(float(value), max_value) for value in feature_means)
        )
        maxima[feature] = (max_value, max_count)

    for cluster in clusters:
        group = numeric.loc[numeric["Cluster"] == cluster]
        items: list[dict[str, Any]] = []
        for spec in NUMERIC_FEATURES:
            feature = spec["feature"]
            item: dict[str, Any] = {
                "feature": feature,
                "label": spec["label"],
                "sentence_label": spec["sentence_label"],
                "unit": spec["unit"],
                "number_format": spec["number_format"],
                "mean": None,
                "z_score": None,
                "comparison": "unavailable",
                "is_cluster_max": False,
                "is_unique_max": False,
                "is_tied_max": False,
                "max_cluster_count": 0,
                "shap_share": None,
                "is_shap_important": False,
            }
            if feature not in numeric.columns:
                items.append(item)
                continue

            mean_value = group[feature].mean()
            if pd.isna(mean_value):
                items.append(item)
                continue
            mean_value = float(mean_value)
            item["mean"] = mean_value

            global_mean = analysis["global_mean"].get(feature)
            global_std = analysis["global_std"].get(feature)
            if (
                global_mean is not None
                and global_std is not None
                and np.isfinite(global_std)
                and global_std > 0
            ):
                z_score = (mean_value - global_mean) / global_std
                item["z_score"] = float(z_score)
                item["comparison"] = _classify_z_score(z_score)

            max_info = maxima.get(feature)
            if max_info is not None:
                max_value, max_count = max_info
                is_max = _is_close(mean_value, max_value)
                item["is_cluster_max"] = is_max
                item["max_cluster_count"] = max_count
                item["is_unique_max"] = is_max and max_count == 1
                item["is_tied_max"] = is_max and max_count > 1

            if shap_info and shap_info.get("available"):
                share = shap_info.get("shares", {}).get(cluster, {}).get(feature)
                if share is not None:
                    item["shap_share"] = float(share)
                    item["is_shap_important"] = share >= SHAP_IMPORTANCE_THRESHOLD
            items.append(item)
        analysis["per_cluster"][cluster] = items

    return analysis


# ---------------------------------------------------------------------------
# Categorical profile
# ---------------------------------------------------------------------------

def analyze_categorical_profile(
    clustered: pd.DataFrame,
    shap_info: dict[str, Any] | None = None,
    categorical_col: str = "sitc",
) -> dict[int, dict[str, Any]]:
    """Mode(s), percentages, normalized entropy and SITC SHAP importance.

    The entropy denominator uses the number of unique SITC categories in the
    **whole dataset**, exactly like ``src.visualization.sitc_entropy_bars``.
    Both outputs therefore always report the same normalized value. Using a
    cluster-local ``m`` would make a two-category cluster score 1.0 even when
    the dataset contains many more categories.
    """
    result: dict[int, dict[str, Any]] = {}
    if (
        clustered is None
        or clustered.empty
        or "Cluster" not in clustered.columns
        or categorical_col not in clustered.columns
    ):
        return result

    global_codes = clustered[categorical_col].dropna().astype(str)
    m_global = int(global_codes.nunique())
    denominator = float(np.log(m_global)) if m_global > 1 else 0.0

    for cluster, group in clustered.groupby("Cluster", sort=True):
        codes = group[categorical_col].dropna().astype(str)
        if codes.empty:
            continue
        counts = codes.value_counts()
        total = int(counts.sum())
        percentages = counts / total * 100.0

        highest = int(counts.max())
        modes = sorted(counts[counts == highest].index.tolist(), key=_sitc_sort_key)

        proportions = counts.to_numpy(dtype=float) / total
        raw_entropy = float(-(proportions * np.log(proportions)).sum())
        if raw_entropy <= 0.0:
            raw_entropy = 0.0
        entropy = raw_entropy / denominator if denominator > 0 else 0.0

        if entropy <= ENTROPY_CONCENTRATED_MAX:
            concentration = "concentrated"
        elif entropy >= ENTROPY_BALANCED_MIN:
            concentration = "balanced"
        else:
            concentration = "moderate"

        others = [
            {
                "code": str(code),
                "description": _sitc_description(code),
                "percentage": float(percentages[code]),
                "percentage_label": _format_percentage(float(percentages[code])),
            }
            for code in counts.index
            if str(code) not in modes
        ]
        others.sort(key=lambda row: (-row["percentage"], _sitc_sort_key(row["code"])))

        cluster_id = int(cluster)
        share = None
        if shap_info and shap_info.get("available"):
            share = shap_info.get("shares", {}).get(cluster_id, {}).get("sitc")

        result[cluster_id] = {
            "cluster": cluster_id,
            "counts": {str(code): int(count) for code, count in counts.items()},
            "percentages": {
                str(code): float(value) for code, value in percentages.items()
            },
            "modes": modes,
            "mode_percentage": float(percentages[modes[0]]) if modes else 0.0,
            "unique_categories": int(len(counts)),
            "m_global": m_global,
            "entropy": float(entropy),
            "entropy_raw": float(raw_entropy),
            "concentration": concentration,
            "others": others,
            "shap_share": float(share) if share is not None else None,
            "is_shap_important": bool(
                share is not None and share >= SHAP_IMPORTANCE_THRESHOLD
            ),
        }

    return result


# ---------------------------------------------------------------------------
# Geographic profile
# ---------------------------------------------------------------------------

def analyze_geographic_profile(clustered: pd.DataFrame) -> dict[int, dict[str, Any]]:
    """Unique-port distribution per province and main island region."""
    result: dict[int, dict[str, Any]] = {}
    if (
        clustered is None
        or clustered.empty
        or "Cluster" not in clustered.columns
        or "port" not in clustered.columns
    ):
        return result

    has_province = "Provinsi" in clustered.columns
    for cluster, group in clustered.groupby("Cluster", sort=True):
        ports = group.dropna(subset=["port"]).drop_duplicates(subset=["port"])
        total_ports = int(len(ports))
        cluster_id = int(cluster)

        if has_province:
            provinces = [_canonical_province(value) for value in ports["Provinsi"]]
        else:
            provinces = [None] * total_ports

        known = [name for name in provinces if name in PROVINCE_TO_REGION]
        mapped_count = len(known)
        coverage = (mapped_count / total_ports) if total_ports else 0.0

        province_counts = pd.Series(known, dtype=object).value_counts().to_dict()
        province_shares = [
            {
                "province": province,
                "count": int(count),
                "percentage": (count / mapped_count * 100.0) if mapped_count else 0.0,
            }
            for province, count in province_counts.items()
        ]
        province_shares.sort(key=lambda row: (-row["percentage"], str(row["province"])))

        region_counts: dict[str, int] = {}
        for province in known:
            region = PROVINCE_TO_REGION[province]
            region_counts[region] = region_counts.get(region, 0) + 1
        region_shares = [
            {
                "region": region,
                "count": int(count),
                "percentage": (count / mapped_count * 100.0) if mapped_count else 0.0,
            }
            for region, count in region_counts.items()
        ]
        region_shares.sort(key=lambda row: (-row["percentage"], str(row["region"])))

        dominant_province = None
        province_level = None
        if province_shares:
            top = province_shares[0]
            share = top["percentage"] / 100.0
            if _is_full(share):
                dominant_province = top["province"]
                province_level = "all"
            elif share >= GEO_DOMINANCE_THRESHOLD:
                dominant_province = top["province"]
                province_level = "strong"

        dominant_region = None
        region_share = 0.0
        if region_shares:
            region_share = region_shares[0]["percentage"] / 100.0
            if region_share >= GEO_DOMINANCE_THRESHOLD:
                dominant_region = region_shares[0]["region"]

        result[cluster_id] = {
            "cluster": cluster_id,
            "observation_count": int(len(group)),
            "port_count": total_ports,
            "mapped_port_count": mapped_count,
            "coverage": float(coverage),
            "province_shares": province_shares,
            "region_shares": region_shares,
            "dominant_province": dominant_province,
            "province_level": province_level,
            "dominant_region": dominant_region,
            "region_share": float(region_share),
        }

    return result


# ---------------------------------------------------------------------------
# Narrative generators
# ---------------------------------------------------------------------------

def generate_numeric_narrative(
    numeric_analysis: dict[str, Any],
    cluster: int,
) -> list[dict[str, Any]]:
    """Build one paragraph per numeric feature for ``cluster``."""
    items = numeric_analysis.get("per_cluster", {}).get(int(cluster), [])
    paragraphs: list[dict[str, Any]] = []

    for item in items:
        sentences: list[str] = []
        mean = item.get("mean")

        if mean is None:
            sentences.append(
                f"Rata-rata {item['sentence_label']} pada klaster ini tidak dapat "
                "dihitung karena data tidak tersedia."
            )
        else:
            sentences.append(
                f"Klaster {cluster} memiliki rata-rata {item['sentence_label']} "
                f"sebesar {_format_number(mean, item['number_format'])} {item['unit']}."
            )

        comparison_sentence = COMPARISON_SENTENCES.get(item.get("comparison"))
        max_sentence = None
        if item.get("is_unique_max"):
            max_sentence = (
                "Nilai tersebut juga merupakan rata-rata tertinggi dibandingkan "
                "klaster lainnya."
            )
        elif item.get("is_tied_max"):
            max_sentence = (
                "Nilai tersebut termasuk rata-rata tertinggi dibandingkan klaster lainnya."
            )

        if comparison_sentence and max_sentence:
            sentences.append(f"{comparison_sentence} {max_sentence}")
        elif comparison_sentence:
            sentences.append(comparison_sentence)
        elif max_sentence:
            sentences.append(max_sentence)
        elif mean is not None:
            sentences.append(
                "Perbandingan dengan rata-rata keseluruhan tidak dapat dihitung "
                "karena seluruh nilai fitur ini identik."
            )

        if item.get("is_shap_important"):
            sentences.append(
                f"Fitur {item['label']} ini juga menjadi salah satu karakteristik "
                "yang kuat dalam membedakan klaster ini dibandingkan kelompok lainnya."
            )

        paragraphs.append(
            {
                "feature": item["feature"],
                "label": item["label"],
                "paragraph": " ".join(sentences),
                "is_shap_important": bool(item.get("is_shap_important")),
            }
        )

    return paragraphs


def generate_categorical_narrative(
    categorical_analysis: dict[int, dict[str, Any]],
    cluster: int,
) -> dict[str, Any]:
    """Build mode, entropy and SITC-differentiator sentences for ``cluster``."""
    analysis = categorical_analysis.get(int(cluster))
    if not analysis:
        return {
            "paragraphs": [
                "Distribusi kategori SITC tidak tersedia untuk klaster ini."
            ],
            "other_distribution": [],
            "shap_important": False,
        }

    modes: list[str] = analysis["modes"]
    mode_codes = _sitc_list(modes)
    mode_pct = _format_percentage(analysis["mode_percentage"])
    paragraphs: list[str] = []

    if len(modes) == 1:
        paragraphs.append(
            f"Klaster {cluster} memiliki kategori SITC terbanyak pada {mode_codes}, "
            f"yaitu {_lower_first(_sitc_description(modes[0]))}, "
            f"dengan proporsi mencapai {mode_pct}."
        )
    elif len(modes) == 2:
        paragraphs.append(
            f"Klaster {cluster} memiliki dua kategori SITC terbanyak, yaitu "
            f"SITC {modes[0]} yang mencakup "
            f"{_lower_first(_sitc_description(modes[0]))} dan "
            f"SITC {modes[1]} yang mencakup "
            f"{_lower_first(_sitc_description(modes[1]))}. "
            f"Kedua kategori tersebut masing-masing memiliki proporsi sebesar {mode_pct}."
        )
    else:
        descriptions = [
            f"SITC {code} yang mencakup {_lower_first(_sitc_description(code))}"
            for code in modes
        ]
        paragraphs.append(
            f"Klaster {cluster} memiliki {_indonesian_count(len(modes))} kategori SITC "
            f"terbanyak, yaitu {', '.join(descriptions[:-1])}, dan {descriptions[-1]}. "
            f"Seluruh kategori tersebut masing-masing memiliki proporsi sebesar {mode_pct}."
        )

    if analysis["concentration"] == "concentrated":
        if len(modes) == 1:
            paragraphs.append(
                "Distribusi kategori SITC pada klaster ini menunjukkan konsentrasi "
                f"pada kategori {mode_codes}."
            )
        else:
            paragraphs.append(
                "Distribusi kategori SITC pada klaster ini menunjukkan konsentrasi "
                f"pada kategori-kategori {mode_codes}."
            )
    elif analysis["concentration"] == "balanced":
        mode_phrase = (
            f"{mode_codes} merupakan kategori dengan proporsi terbesar"
        )
        paragraphs.append(
            f"Meskipun {mode_phrase}, distribusi komoditas pada klaster ini secara "
            "keseluruhan relatif merata di antara berbagai kategori SITC."
        )

    if analysis.get("is_shap_important"):
        paragraphs.append(
            "Komposisi kategori SITC juga menjadi salah satu fitur yang kuat dalam "
            "membedakan karakteristik klaster ini dari kelompok lainnya."
        )

    return {
        "paragraphs": paragraphs,
        "other_distribution": analysis["others"],
        "shap_important": bool(analysis.get("is_shap_important")),
    }


def generate_geographic_narrative(
    geographic_analysis: dict[int, dict[str, Any]],
    cluster: int,
) -> dict[str, Any]:
    """Describe the unique-port distribution by province and island region."""
    analysis = geographic_analysis.get(int(cluster))
    if not analysis:
        return {
            "paragraphs": [
                "Persebaran geografis pelabuhan tidak tersedia untuk klaster ini."
            ],
            "province_shares": [],
            "region_shares": [],
            "coverage": 0.0,
        }

    paragraphs: list[str] = []
    port_count = analysis["port_count"]
    observation_count = analysis["observation_count"]
    coverage = analysis["coverage"]
    province_shares = analysis["province_shares"]
    region_shares = analysis["region_shares"]
    dominant_province = analysis["dominant_province"]
    province_level = analysis["province_level"]
    dominant_region = analysis["dominant_region"]

    if port_count:
        paragraphs.append(
            f"Klaster ini mencakup {port_count:,} pelabuhan unik dari "
            f"{observation_count:,} observasi pelabuhan × SITC."
        )

    mapped_phrase = (
        "seluruh pelabuhan unik"
        if _is_full(coverage)
        else "pelabuhan unik yang provinsinya teridentifikasi"
    )

    if not province_shares:
        paragraphs.append(
            "Persebaran provinsi pelabuhan pada klaster ini tidak dapat ditentukan "
            "karena data provinsi tidak tersedia atau tidak dikenali."
        )
    elif dominant_province and province_level == "all":
        if _is_full(coverage):
            if dominant_region:
                paragraphs.append(
                    f"Seluruh pelabuhan dalam klaster ini berada di Provinsi "
                    f"{dominant_province}, yang termasuk kawasan kepulauan "
                    f"{dominant_region}."
                )
            else:
                paragraphs.append(
                    "Seluruh pelabuhan dalam klaster ini berada di Provinsi "
                    f"{dominant_province}."
                )
        else:
            paragraphs.append(
                "Seluruh pelabuhan yang provinsinya teridentifikasi berada di "
                f"Provinsi {dominant_province}."
            )
    elif dominant_province and province_level == "strong":
        share = province_shares[0]["percentage"]
        paragraphs.append(
            "Persebaran pelabuhan pada klaster ini terkonsentrasi terutama di "
            f"Provinsi {dominant_province}, yang mencakup sekitar "
            f"{_format_percentage(share)} dari {mapped_phrase}."
        )
    else:
        top = province_shares[:NARRATIVE_TOP_PROVINCES]
        listed = ", ".join(
            f"{row['province']} ({_format_percentage(row['percentage'])})"
            for row in top
        )
        paragraphs.append(
            "Tidak ada provinsi yang mendominasi persebaran pelabuhan klaster ini. "
            f"Pelabuhan tersebar di beberapa provinsi, dengan jumlah terbesar di {listed}."
        )

    if region_shares:
        if dominant_region and province_level == "all" and _is_full(coverage):
            # Already stated together with the dominant province.
            pass
        elif dominant_region:
            top_provinces = province_shares[:2]
            if len(top_provinces) >= 2:
                utama = (
                    f"Provinsi {top_provinces[0]['province']} dan "
                    f"{top_provinces[1]['province']}"
                )
                paragraphs.append(
                    "Secara geografis, klaster ini terkonsentrasi di kawasan "
                    f"kepulauan {dominant_region}, dengan persebaran utama pada {utama}."
                )
            else:
                paragraphs.append(
                    "Secara geografis, klaster ini terkonsentrasi di kawasan "
                    f"kepulauan {dominant_region}."
                )
        else:
            top_regions = region_shares[:NARRATIVE_TOP_REGIONS]
            if len(top_regions) >= 2:
                main_regions = " dan ".join(row["region"] for row in top_regions[:2])
                paragraphs.append(
                    "Klaster ini memiliki persebaran geografis lintas kawasan "
                    f"kepulauan, terutama pada wilayah {main_regions}."
                )
                remaining = top_regions[2:]
                if remaining:
                    others = ", ".join(row["region"] for row in remaining)
                    paragraphs.append(
                        f"Anggota lainnya juga ditemukan di kawasan {others}."
                    )
            else:
                paragraphs.append(
                    "Persebaran pelabuhan klaster ini hanya terdeteksi pada kawasan "
                    f"{top_regions[0]['region']}."
                )
    elif province_shares:
        paragraphs.append(
            "Persebaran kawasan kepulauan tidak dapat ditentukan karena provinsi "
            "pelabuhan tidak dikenali."
        )

    if coverage < GEO_COVERAGE_MIN:
        paragraphs.append(
            "Analisis geografis ini hanya mencakup "
            f"{_format_percentage(coverage * 100)} dari seluruh pelabuhan unik "
            "karena sebagian provinsi pelabuhan tidak tersedia atau tidak dikenali."
        )

    return {
        "paragraphs": paragraphs,
        "province_shares": province_shares,
        "region_shares": region_shares,
        "coverage": float(coverage),
    }


def _generate_identity_summary(
    numeric_items: list[dict[str, Any]],
    categorical_analysis: dict[str, Any] | None,
    geographic_analysis: dict[str, Any] | None,
) -> str | None:
    """Optional closing paragraph built only from conditions that are met."""
    parts: list[str] = []

    strong = [
        item
        for item in numeric_items
        if item.get("mean") is not None
        and (
            item.get("comparison") in {"far_above", "far_below"}
            or item.get("is_unique_max")
        )
    ]
    if strong:
        strongest = max(strong, key=lambda item: abs(item.get("z_score") or 0.0))
        if strongest.get("comparison") == "far_above":
            parts.append(
                f"Klaster ini memiliki {strongest['label']} yang jauh di atas "
                "rata-rata keseluruhan."
            )
        elif strongest.get("comparison") == "far_below":
            parts.append(
                f"Klaster ini memiliki {strongest['label']} yang jauh di bawah "
                "rata-rata keseluruhan."
            )
        else:
            parts.append(
                f"Klaster ini mencatat rata-rata {strongest['label']} tertinggi "
                "dibandingkan klaster lainnya."
            )

    if categorical_analysis:
        modes = categorical_analysis["modes"]
        if modes:
            parts.append(
                f"Komposisi komoditas terbesarnya adalah {_sitc_list(modes)} "
                f"dengan proporsi "
                f"{_format_percentage(categorical_analysis['mode_percentage'])}."
            )

    if geographic_analysis and geographic_analysis.get("port_count"):
        dominant_region = geographic_analysis.get("dominant_region")
        if dominant_region:
            parts.append(
                f"Pelabuhannya terkonsentrasi di kawasan kepulauan {dominant_region}."
            )
        elif geographic_analysis.get("province_shares"):
            parts.append("Pelabuhannya tersebar lintas kawasan kepulauan.")

    if not parts:
        return None
    return " ".join(parts)


def generate_cluster_narrative(
    cluster: int,
    numeric_analysis: dict[str, Any],
    categorical_analysis: dict[int, dict[str, Any]],
    geographic_analysis: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Combine every section for one cluster."""
    numeric_items = numeric_analysis.get("per_cluster", {}).get(int(cluster), [])
    numeric_paragraphs = generate_numeric_narrative(numeric_analysis, cluster)
    categorical = generate_categorical_narrative(categorical_analysis, cluster)
    geographic = generate_geographic_narrative(geographic_analysis, cluster)
    summary = _generate_identity_summary(
        numeric_items,
        categorical_analysis.get(int(cluster)),
        geographic_analysis.get(int(cluster)),
    )
    geo = geographic_analysis.get(int(cluster), {})

    return {
        "cluster": int(cluster),
        "label": f"Cluster {cluster}",
        "observation_count": int(geo.get("observation_count", 0)),
        "port_count": int(geo.get("port_count", 0)),
        "numeric": numeric_paragraphs,
        "categorical": categorical,
        "geographic": geographic,
        "summary": summary,
    }


def generate_all_cluster_narratives(
    model_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate the narrative blocks for every cluster.

    A failure in the SHAP or geographic analysis never prevents the remaining
    descriptions from being produced.
    """
    clustered = (model_result or {}).get("clustered")
    if (
        clustered is None
        or not isinstance(clustered, pd.DataFrame)
        or clustered.empty
        or "Cluster" not in clustered.columns
    ):
        return []

    try:
        shap_info = analyze_shap_importance(model_result)
    except Exception as exc:  # noqa: BLE001 - narrative must degrade gracefully
        shap_info = {"available": False, "reason": str(exc), "shares": {}}

    try:
        numeric_analysis = analyze_numeric_profile(clustered, shap_info)
    except Exception:  # noqa: BLE001
        numeric_analysis = {"clusters": [], "per_cluster": {}}

    try:
        categorical_analysis = analyze_categorical_profile(clustered, shap_info)
    except Exception:  # noqa: BLE001
        categorical_analysis = {}

    try:
        geographic_analysis = analyze_geographic_profile(clustered)
    except Exception:  # noqa: BLE001
        geographic_analysis = {}

    clusters = numeric_analysis.get("clusters") or sorted(
        int(cluster) for cluster in clustered["Cluster"].dropna().unique()
    )
    return [
        generate_cluster_narrative(
            cluster, numeric_analysis, categorical_analysis, geographic_analysis
        )
        for cluster in clusters
    ]
