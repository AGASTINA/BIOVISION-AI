import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from xgboost import Booster, DMatrix


PROJECT_ROOT = Path(__file__).resolve().parent
FEATURES_PATH = PROJECT_ROOT / "outputs" / "SurgWound" / "fused_features.csv"
LABELS_PATH = PROJECT_ROOT / "data" / "SurgWound" / "clinical_labels.csv"
HEALING_MODEL_PATH = PROJECT_ROOT / "models" / "healing_xgboost.json"
INFECTION_MODEL_PATH = PROJECT_ROOT / "models" / "infection_xgboost.json"
SHAP_ROOT = PROJECT_ROOT / "outputs" / "SurgWound" / "shap"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "SurgWound" / "btis"

HEALING_WEIGHT = 0.40
INFECTION_WEIGHT = 0.40
TISSUE_WEIGHT = 0.20
INFECTION_RISK_VALUES = {"Low": 0.0, "Medium": 0.5, "High": 1.0}
HANDCRAFTED_PREFIXES = ("rgb_", "hsv_", "lab_", "glcm_")
HANDCRAFTED_SHAPE = {
    "wound_area",
    "wound_perimeter",
    "aspect_ratio",
    "circularity",
    "compactness",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _load_inputs() -> tuple[np.ndarray, list[str], list[str]]:
    feature_rows = _read_csv(FEATURES_PATH)
    label_rows = _read_csv(LABELS_PATH)
    feature_names = [name for name in feature_rows[0] if name != "filename"]
    feature_map = {row["filename"]: row for row in feature_rows}
    test_names = sorted(row["filename"] for row in label_rows if row["split"] == "test")
    if len(test_names) != 137 or len(set(test_names)) != 137:
        raise ValueError("Expected exactly 137 unique test filenames")
    matrix = np.asarray(
        [[float(feature_map[name][feature]) for feature in feature_names] for name in test_names],
        dtype=np.float32,
    )
    if matrix.shape != (137, 1563):
        raise ValueError(f"Unexpected test feature shape: {matrix.shape}")
    return matrix, test_names, feature_names


def _load_probabilities(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    healing = Booster()
    infection = Booster()
    healing.load_model(HEALING_MODEL_PATH)
    infection.load_model(INFECTION_MODEL_PATH)
    healing_probability = np.asarray(healing.predict(DMatrix(data)), dtype=float)
    infection_probability = np.asarray(infection.predict(DMatrix(data)), dtype=float)
    if healing_probability.shape != (137,) or infection_probability.shape != (137, 3):
        raise ValueError("Unexpected model probability dimensions")
    return healing_probability, infection_probability


def _load_tissue_signal(filenames: list[str], feature_names: list[str]) -> np.ndarray:
    healing_rows = _read_csv(SHAP_ROOT / "healing_shap_values.csv")
    infection_rows = _read_csv(SHAP_ROOT / "infection_shap_values.csv")
    expected_features = set(feature_names)
    healing_features = [name for name in healing_rows[0] if name != "filename"]
    if healing_features != feature_names:
        raise ValueError("Healing SHAP feature order does not match fused features")
    healing_map = {row["filename"]: np.asarray([float(row[name]) for name in feature_names]) for row in healing_rows}
    infection_values: dict[str, dict[str, dict[str, float]]] = {}
    for row in infection_rows:
        if row["feature"] not in expected_features:
            raise ValueError(f"Unexpected infection SHAP feature: {row['feature']}")
        infection_values.setdefault(row["filename"], {}).setdefault(row["class"], {})[row["feature"]] = float(row["shap_value"])

    handcrafted = [
        name for name in feature_names
        if name.startswith(HANDCRAFTED_PREFIXES) or name in HANDCRAFTED_SHAPE
    ]
    healing_signal = np.asarray([
        np.mean(np.abs(healing_map[filename][[feature_names.index(name) for name in handcrafted]]) )
        for filename in filenames
    ])
    infection_signal = np.asarray([
        np.mean([
            abs(infection_values[filename][class_name][feature])
            for class_name in ("Low", "Medium", "High")
            for feature in handcrafted
        ])
        for filename in filenames
    ])
    healing_scale = max(float(healing_signal.max()), np.finfo(float).eps)
    infection_scale = max(float(infection_signal.max()), np.finfo(float).eps)
    return 100.0 * 0.5 * (healing_signal / healing_scale + infection_signal / infection_scale)


def _write_results(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        fields = list(rows[0])
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _validate(rows: list[dict[str, float | str]]) -> None:
    if len(rows) != 137 or len({row["filename"] for row in rows}) != 137:
        raise ValueError("BTIS rows do not contain exactly one row per test filename")
    for row in rows:
        healing_sum = float(row["healed_probability"]) + float(row["not_healed_probability"])
        infection_sum = sum(float(row[f"infection_{name.lower()}_probability"]) for name in ("low", "medium", "high"))
        probabilities = [
            float(row["healed_probability"]),
            float(row["not_healed_probability"]),
            float(row["infection_low_probability"]),
            float(row["infection_medium_probability"]),
            float(row["infection_high_probability"]),
        ]
        if not all(0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("BTIS probability outside [0, 1]")
        if not np.isclose(healing_sum, 1.0, atol=1e-5) or not np.isclose(infection_sum, 1.0, atol=1e-5):
            raise ValueError("BTIS probabilities do not sum to one")
        if not np.isfinite(float(row["BTIS"])) or not 0.0 <= float(row["BTIS"]) <= 100.0:
            raise ValueError("BTIS outside declared [0, 100] scale")


def run_analysis() -> None:
    data, filenames, feature_names = _load_inputs()
    healed_probability, infection_probability = _load_probabilities(data)
    tissue_component = _load_tissue_signal(filenames, feature_names)
    not_healed_probability = healed_probability
    healed_probability = 1.0 - not_healed_probability
    infection_component = 100.0 * sum(
        infection_probability[:, index] * INFECTION_RISK_VALUES[class_name]
        for index, class_name in enumerate(("Low", "Medium", "High"))
    )
    healing_component = 100.0 * not_healed_probability
    btis = (
        HEALING_WEIGHT * healing_component
        + INFECTION_WEIGHT * infection_component
        + TISSUE_WEIGHT * tissue_component
    )
    rows = []
    for index, filename in enumerate(filenames):
        rows.append({
            "filename": filename,
            "healed_probability": float(healed_probability[index]),
            "not_healed_probability": float(not_healed_probability[index]),
            "infection_low_probability": float(infection_probability[index, 0]),
            "infection_medium_probability": float(infection_probability[index, 1]),
            "infection_high_probability": float(infection_probability[index, 2]),
            "tissue_component": float(tissue_component[index]),
            "healing_component": float(healing_component[index]),
            "infection_component": float(infection_component[index]),
            "BTIS": float(btis[index]),
        })
    _validate(rows)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_results(OUTPUT_ROOT / "btis_results.csv", rows)
    summary = {
        "predefined_formula_found": False,
        "prototype": True,
        "scale": "0-100",
        "formula": "BTIS = 0.40 * healing_component + 0.40 * infection_component + 0.20 * tissue_component",
        "component_formulas": {
            "healing_component": "100 * P(Not Healed)",
            "infection_component": "100 * (0.0 * P(Low) + 0.5 * P(Medium) + 1.0 * P(High))",
            "tissue_component": "50 * (mean_abs_handcrafted_healing_SHAP/max_test_mean_abs_handcrafted_healing_SHAP + mean_abs_handcrafted_infection_SHAP/max_test_mean_abs_handcrafted_infection_SHAP)",
        },
        "weights": {"healing": HEALING_WEIGHT, "infection": INFECTION_WEIGHT, "tissue": TISSUE_WEIGHT},
        "test_samples": len(rows),
        "btis_min": float(np.min(btis)),
        "btis_max": float(np.max(btis)),
        "btis_mean": float(np.mean(btis)),
        "btis_median": float(np.median(btis)),
        "limitations": [
            "Prototype engineering score; not clinically validated.",
            "Infection component preserves Low, Medium, and High probabilities but uses prototype ordinal weights.",
            "The infection model has High-risk F1 and recall of 0.0; high-risk BTIS interpretation is unreliable.",
            "SurgWound Healing Status is an image-level label, not longitudinal healing evidence.",
            "No clinical thresholds or prevalence claims are assigned.",
        ],
    }
    with (OUTPUT_ROOT / "btis_summary.json").open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2)
    plt.figure(figsize=(10, 6))
    plt.hist(btis, bins=20, color="#3b82f6", edgecolor="white")
    plt.xlabel("BTIS prototype score (0-100)")
    plt.ylabel("Number of test images")
    plt.title("BTIS prototype score distribution; not clinical prevalence")
    plt.tight_layout()
    plt.savefig(OUTPUT_ROOT / "btis_distribution.png", dpi=160)
    plt.close()
    print("BTIS_VALIDATION = PASS")
    print("BTIS ANALYSIS COMPLETE")
    print("predefined_formula_found: False")
    print(f"formula: {summary['formula']}")
    print(f"test_samples: {len(rows)}")
    print(f"BTIS_range: {summary['btis_min']:.6f} - {summary['btis_max']:.6f}")
    print(f"BTIS_mean: {summary['btis_mean']:.6f}")
    print(f"BTIS_median: {summary['btis_median']:.6f}")
    print(f"outputs: {OUTPUT_ROOT}")
    print("limitations: prototype/non-clinical; infection High-risk interpretation is unreliable.")


if __name__ == "__main__":
    run_analysis()
