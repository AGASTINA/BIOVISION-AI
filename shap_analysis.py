import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import shap
from xgboost import Booster


PROJECT_ROOT = Path(__file__).resolve().parent
FEATURES_PATH = PROJECT_ROOT / "outputs" / "SurgWound" / "fused_features.csv"
LABELS_PATH = PROJECT_ROOT / "data" / "SurgWound" / "clinical_labels.csv"
MODEL_ROOT = PROJECT_ROOT / "models"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "SurgWound" / "shap"
CLASS_NAMES = {"healing": ["Healed", "Not Healed"], "infection": ["Low", "Medium", "High"]}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _load_test_data() -> tuple[np.ndarray, list[str], list[str]]:
    feature_rows = _read_csv(FEATURES_PATH)
    label_rows = _read_csv(LABELS_PATH)
    feature_names = [name for name in feature_rows[0] if name != "filename"]
    feature_map = {row["filename"]: row for row in feature_rows}
    test_names = sorted(row["filename"] for row in label_rows if row["split"] == "test")
    if len(test_names) != 137 or len(set(test_names)) != 137:
        raise ValueError("Expected 137 unique test filenames")
    if set(test_names) - set(feature_map):
        raise ValueError("Test filenames are missing from fused features")
    matrix = np.asarray(
        [[float(feature_map[name][feature]) for feature in feature_names] for name in test_names],
        dtype=np.float32,
    )
    if matrix.shape != (137, 1563):
        raise ValueError(f"Unexpected test feature shape: {matrix.shape}")
    return matrix, test_names, feature_names


def _load_model(path: Path) -> Booster:
    model = Booster()
    model.load_model(path)
    return model


def _as_class_array(values: Any, sample_count: int, feature_count: int, class_count: int) -> np.ndarray:
    if isinstance(values, list):
        array = np.stack([np.asarray(value) for value in values], axis=-1)
    else:
        array = np.asarray(values)
    if class_count == 2 and array.ndim == 2:
        array = np.stack([-array, array], axis=-1)
    if array.shape != (sample_count, feature_count, class_count):
        raise ValueError(f"Unexpected SHAP shape: {array.shape}")
    return array


def _group(feature: str) -> str:
    if feature.startswith(("rgb_", "hsv_", "lab_")):
        return "handcrafted color"
    if feature.startswith("glcm_"):
        return "handcrafted texture"
    if feature in {"wound_area", "wound_perimeter", "aspect_ratio", "circularity", "compactness"}:
        return "handcrafted shape"
    if feature.startswith("efficientnet_"):
        return "EfficientNet deep features"
    return "other"


def _top_rows(values: np.ndarray, feature_names: list[str], class_name: str | None = None) -> list[dict[str, Any]]:
    mean_abs = np.mean(np.abs(values), axis=0)
    mean = np.mean(values, axis=0)
    order = np.argsort(-mean_abs)
    return [
        {
            "feature": feature_names[index],
            **({"class": class_name} if class_name is not None else {}),
            "mean_abs_shap": float(mean_abs[index]),
            "mean_shap": float(mean[index]),
        }
        for index in order
    ]


def _write_healing_values(path: Path, filenames: list[str], feature_names: list[str], values: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", *feature_names])
        writer.writerows([[filename, *row] for filename, row in zip(filenames, values)])


def _write_infection_values(path: Path, filenames: list[str], feature_names: list[str], values: np.ndarray, classes: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "class", "feature", "shap_value"])
        for sample_index, filename in enumerate(filenames):
            for class_index, class_name in enumerate(classes):
                for feature_index, feature_name in enumerate(feature_names):
                    writer.writerow([filename, class_name, feature_name, float(values[sample_index, feature_index, class_index])])


def _write_importance(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["feature", "mean_abs_shap", "mean_shap"] if "class" not in rows[0] else ["feature", "class", "mean_abs_shap", "mean_shap"]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_healing(values: np.ndarray, data: np.ndarray, names: list[str], feature_names: list[str]) -> None:
    top = _top_rows(values, feature_names)[:20]
    top_names = [row["feature"] for row in top]
    indices = [feature_names.index(name) for name in top_names]
    display_values = values[:, indices]
    plt.figure(figsize=(12, 8))
    shap.summary_plot(display_values, data[:, indices], feature_names=top_names, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_ROOT / "healing_summary.png", dpi=160, bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(10, 8))
    shap.summary_plot(display_values, data[:, indices], feature_names=top_names, plot_type="bar", max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_ROOT / "healing_bar.png", dpi=160, bbox_inches="tight")
    plt.close()


def _plot_infection(values: np.ndarray, data: np.ndarray, feature_names: list[str], classes: list[str]) -> None:
    combined = np.mean(np.abs(values), axis=2)
    top = _top_rows(combined, feature_names)[:20]
    top_names = [row["feature"] for row in top]
    indices = [feature_names.index(name) for name in top_names]
    figure, axes = plt.subplots(1, len(classes), figsize=(18, 7), squeeze=False)
    for class_index, class_name in enumerate(classes):
        plt.sca(axes[0, class_index])
        shap.summary_plot(values[:, indices, class_index], data[:, indices], feature_names=top_names, max_display=20, show=False)
        axes[0, class_index].set_title(class_name)
    figure.tight_layout()
    figure.savefig(OUTPUT_ROOT / "infection_summary.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(combined[:, indices], data[:, indices], feature_names=top_names, plot_type="bar", max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_ROOT / "infection_bar.png", dpi=160, bbox_inches="tight")
    plt.close()


def _validate(values: np.ndarray, filenames: list[str], feature_names: list[str], class_count: int) -> None:
    if len(filenames) != 137 or len(set(filenames)) != 137:
        raise ValueError("SHAP filenames are not exactly the 137 unique test filenames")
    if values.shape != (137, 1563, class_count):
        raise ValueError(f"SHAP dimensions are invalid: {values.shape}")
    if not np.isfinite(values).all() or len(feature_names) != 1563:
        raise ValueError("SHAP values or feature ordering are invalid")


def _print_top(title: str, rows: list[dict[str, Any]]) -> None:
    print(title)
    for row in rows[:20]:
        direction = "positive" if row["mean_shap"] > 0 else "negative" if row["mean_shap"] < 0 else "neutral"
        print(f"  {row['feature']} | mean_abs_shap={row['mean_abs_shap']:.6f} | {direction} | {_group(row['feature'])}")


def run_analysis() -> None:
    data, filenames, feature_names = _load_test_data()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    healing_model = _load_model(MODEL_ROOT / "healing_xgboost.json")
    infection_model = _load_model(MODEL_ROOT / "infection_xgboost.json")
    healing_values = _as_class_array(shap.TreeExplainer(healing_model)(data).values, 137, 1563, 2)
    infection_values = _as_class_array(shap.TreeExplainer(infection_model)(data).values, 137, 1563, 3)
    _validate(healing_values, filenames, feature_names, 2)
    _validate(infection_values, filenames, feature_names, 3)

    healing_rows = _top_rows(healing_values[:, :, 1], feature_names)
    infection_rows = [row for class_index, class_name in enumerate(CLASS_NAMES["infection"]) for row in _top_rows(infection_values[:, :, class_index], feature_names, class_name)]
    _write_healing_values(OUTPUT_ROOT / "healing_shap_values.csv", filenames, feature_names, healing_values[:, :, 1])
    _write_infection_values(OUTPUT_ROOT / "infection_shap_values.csv", filenames, feature_names, infection_values, CLASS_NAMES["infection"])
    _write_importance(OUTPUT_ROOT / "healing_feature_importance.csv", healing_rows)
    _write_importance(OUTPUT_ROOT / "infection_feature_importance.csv", infection_rows)
    _plot_healing(healing_values[:, :, 1], data, filenames, feature_names)
    _plot_infection(infection_values, data, feature_names, CLASS_NAMES["infection"])
    print("SHAP_VALIDATION = PASS")
    print("SHAP ANALYSIS COMPLETE")
    print(f"Healing: test samples=137, feature count=1563")
    _print_top("Healing top 20 features:", healing_rows)
    for class_index, class_name in enumerate(CLASS_NAMES["infection"]):
        _print_top(f"Infection — {class_name} top 20 features:", _top_rows(infection_values[:, :, class_index], feature_names, class_name))
    print("High infection has only 23 samples overall and zero correctly predicted High-risk test cases; interpret its SHAP values cautiously.")
    print(f"Outputs: {OUTPUT_ROOT}")


if __name__ == "__main__":
    run_analysis()
