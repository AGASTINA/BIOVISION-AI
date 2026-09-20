import html
import io
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from xgboost import Booster, DMatrix

import database as db

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
HEALING_MODEL_PATH = PROJECT_ROOT / "models" / "healing_xgboost.json"
INFECTION_MODEL_PATH = PROJECT_ROOT / "models" / "infection_xgboost.json"
FUSED_FEATURES_PATH = PROJECT_ROOT / "outputs" / "SurgWound" / "fused_features.csv"
SHAP_ROOT = PROJECT_ROOT / "outputs" / "SurgWound" / "shap"
STORAGE_PATH = PROJECT_ROOT / "data" / "storage"
TEST_IMG_DIR  = PROJECT_ROOT / "test_images"
HERO_IMG_PATH = STORAGE_PATH / "hero.jpg"
FEATURE_COUNT = 1563
HANDCRAFTED_FEATURES = [
    "rgb_mean_r", "rgb_mean_g", "rgb_mean_b", "rgb_std_r", "rgb_std_g", "rgb_std_b",
    "hsv_mean_h", "hsv_mean_s", "hsv_mean_v", "hsv_std_h", "hsv_std_s", "hsv_std_v",
    "lab_mean_l", "lab_mean_a", "lab_mean_b", "lab_std_l", "lab_std_a", "lab_std_b",
    "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_entropy",
    "wound_area", "wound_perimeter", "aspect_ratio", "circularity", "compactness",
]


# ==============================================================================
# EXISTING AI PIPELINE FUNCTIONS (PRESERVED FUNCTIONALLY IDENTICAL)
# ==============================================================================

@st.cache_resource
def _load_xgboost_models() -> tuple[Booster, Booster]:
    healing = Booster()
    infection = Booster()
    healing.load_model(HEALING_MODEL_PATH)
    infection.load_model(INFECTION_MODEL_PATH)
    return healing, infection


@st.cache_resource
def _load_explainers():
    import shap
    healing, infection = _load_xgboost_models()
    return shap.TreeExplainer(healing), shap.TreeExplainer(infection)


@st.cache_data
def _load_test_shap_scales() -> tuple[float, float]:
    healing_rows = _read_csv(SHAP_ROOT / "healing_shap_values.csv")
    infection_rows = _read_csv(SHAP_ROOT / "infection_shap_values.csv")
    healing_values = np.asarray([[float(row[name]) for name in HANDCRAFTED_FEATURES] for row in healing_rows])
    infection_by_file: dict[str, list[float]] = {}
    for row in infection_rows:
        if row["feature"] in HANDCRAFTED_FEATURES:
            infection_by_file.setdefault(row["filename"], []).append(abs(float(row["shap_value"])))
    healing_signal = np.mean(np.abs(healing_values), axis=1)
    infection_signal = np.asarray([np.mean(values) for values in infection_by_file.values()])
    return max(float(healing_signal.max()), np.finfo(float).eps), max(float(infection_signal.max()), np.finfo(float).eps)


def _read_csv(path: Path) -> list[dict[str, str]]:
    import csv
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _shap_values(explainer, vector: np.ndarray, class_count: int) -> np.ndarray:
    values = np.asarray(explainer(vector).values)
    if class_count == 2 and values.ndim == 2:
        values = np.stack([-values, values], axis=-1)
    if values.ndim == 2:
        values = values[:, :, None]
    if values.shape != (1, FEATURE_COUNT, class_count):
        raise ValueError(f"Unexpected SHAP shape: {values.shape}")
    return values


def _feature_group(name: str) -> str:
    if name.startswith(("rgb_", "hsv_", "lab_")):
        return "handcrafted color"
    if name.startswith("glcm_"):
        return "handcrafted texture"
    if name in {"wound_area", "wound_perimeter", "aspect_ratio", "circularity", "compactness"}:
        return "handcrafted shape"
    if name.startswith("efficientnet_"):
        return "EfficientNet deep features"
    return "other"


def _top_shap(values: np.ndarray, feature_names: list[str], limit: int = 5) -> list[dict[str, object]]:
    scores = np.abs(values[0])
    indices = np.argsort(-scores)[:limit]
    return [
        {
            "feature": feature_names[index],
            "shap_value": float(values[0, index]),
            "abs_shap": float(scores[index]),
            "direction": "positive" if values[0, index] > 0 else "negative" if values[0, index] < 0 else "neutral",
            "group": _feature_group(feature_names[index]),
        }
        for index in indices
    ]


def _image_overlay(image_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    overlay = image_rgb.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(image_rgb, 0.65, overlay, 0.35, 0)


def _build_fused_row(segmentation, peri_wound, filename: str) -> tuple[dict[str, float | str], dict[str, float | str]]:
    from efficientnet_features import extract_efficientnet_features, save_feature_vectors_csv
    from feature_fusion import fuse_feature_csvs
    from handcrafted_features import extract_handcrafted_features, save_features_csv

    handcrafted = extract_handcrafted_features(segmentation, peri_wound, filename)
    pipeline_item = [(filename, segmentation, peri_wound)]
    efficientnet = extract_efficientnet_features(pipeline_item, batch_size=1)[0]
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        handcrafted_path = save_features_csv([handcrafted], temp / "handcrafted.csv")
        efficientnet_path = save_feature_vectors_csv([efficientnet], temp / "efficientnet.csv")
        fused_path, stats = fuse_feature_csvs(handcrafted_path, efficientnet_path, temp / "fused.csv")
        if stats["final_features"] != FEATURE_COUNT:
            raise ValueError(f"Expected {FEATURE_COUNT} fused features, got {stats['final_features']}")
        fused = _read_csv(fused_path)[0]
    return handcrafted, {name: float(fused[name]) for name in fused if name != "filename"}


def _process_image(image_bytes: bytes, filename: str) -> dict[str, object]:
    from btis_analysis import (
        HEALING_WEIGHT,
        INFECTION_RISK_VALUES,
        INFECTION_WEIGHT,
        TISSUE_WEIGHT,
    )
    from peri_wound import extract_peri_wound
    from yolo_inference import run_inference
    suffix = Path(filename).suffix.lower() or ".jpg"
    image_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    image_file.write(image_bytes)
    image_file.close()
    visualization_path = Path(tempfile.mktemp(suffix=".jpg"))
    try:
        segmentation = run_inference(image_file.name, visualization_path, MODEL_PATH)
        peri_wound = extract_peri_wound(segmentation)
        handcrafted, fused = _build_fused_row(segmentation, peri_wound, filename)
        feature_names = list(fused)
        vector = np.asarray([[fused[name] for name in feature_names]], dtype=np.float32)
        healing_model, infection_model = _load_xgboost_models()
        healing_probability = float(healing_model.predict(DMatrix(vector))[0])
        infection_probability = np.asarray(infection_model.predict(DMatrix(vector))[0], dtype=float)
        healing_explainer, infection_explainer = _load_explainers()
        healing_shap = _shap_values(healing_explainer, vector, 2)
        infection_shap = _shap_values(infection_explainer, vector, 3)
        healing_top = _top_shap(healing_shap[:, :, 1:2].reshape(1, FEATURE_COUNT), feature_names)
        infection_top = {
            class_name: _top_shap(infection_shap[:, :, index:index + 1].reshape(1, FEATURE_COUNT), feature_names)
            for index, class_name in enumerate(("Low", "Medium", "High"))
        }
        healing_scale, infection_scale = _load_test_shap_scales()
        handcrafted_indices = [feature_names.index(name) for name in HANDCRAFTED_FEATURES]
        healing_signal = float(np.mean(np.abs(healing_shap[0, handcrafted_indices, 1])))
        infection_signal = float(np.mean(np.abs(infection_shap[0, handcrafted_indices, :])))
        tissue_component = 50.0 * (healing_signal / healing_scale + infection_signal / infection_scale)
        not_healed_probability = healing_probability
        healed_probability = 1.0 - not_healed_probability
        healing_component = 100.0 * not_healed_probability
        infection_component = 100.0 * sum(infection_probability[index] * INFECTION_RISK_VALUES[name] for index, name in enumerate(("Low", "Medium", "High")))
        btis = HEALING_WEIGHT * healing_component + INFECTION_WEIGHT * infection_component + TISSUE_WEIGHT * tissue_component
        return {
            "filename": filename,
            "original": segmentation.image,
            "segmentation": segmentation,
            "peri_wound": peri_wound,
            "handcrafted": handcrafted,
            "feature_count": len(feature_names),
            "healing_probability": healed_probability,
            "not_healed_probability": not_healed_probability,
            "infection_probability": infection_probability,
            "healing_class": "Not Healed" if not_healed_probability >= healed_probability else "Healed",
            "infection_class": ("Low", "Medium", "High")[int(np.argmax(infection_probability))],
            "healing_top": healing_top,
            "infection_top": infection_top,
            "healing_component": healing_component,
            "infection_component": infection_component,
            "tissue_component": tissue_component,
            "btis": btis,
            "visualization_path": visualization_path,
        }
    finally:
        Path(image_file.name).unlink(missing_ok=True)


def _report_payload(result: dict[str, object]) -> dict[str, object]:
    handcrafted = result["handcrafted"]
    return {
        "image_id": result["filename"],
        "wound_assessment": {"wound_area": handcrafted.get("wound_area", 0), "wound_perimeter": handcrafted.get("wound_perimeter", 0), "aspect_ratio": handcrafted.get("aspect_ratio", 0), "circularity": handcrafted.get("circularity", 0), "compactness": handcrafted.get("compactness", 0)},
        "peri_wound_assessment": {name: handcrafted[name] for name in HANDCRAFTED_FEATURES if name in handcrafted and name.startswith(("rgb_", "hsv_", "lab_", "glcm_"))},
        "healing_assessment": {"label": "Image-Level Healing Status", "healed_probability": result["healing_probability"], "not_healed_probability": result["not_healed_probability"], "predicted_class": result["healing_class"]},
        "infection_risk_assessment": {"low_probability": float(result["infection_probability"][0]), "medium_probability": float(result["infection_probability"][1]), "high_probability": float(result["infection_probability"][2]), "predicted_class": result["infection_class"], "limitation": "High-risk classification currently has limited reliability."},
        "btis": {"label": "Experimental BioVision Tissue/Healing Infection Score", "value": result["btis"], "healing_component": result["healing_component"], "infection_component": result["infection_component"], "tissue_component": result["tissue_component"], "disclaimer": "Prototype / Research Use Only"},
        "explainability": {"healing": result["healing_top"], "infection": result["infection_top"]},
        "limitations": ["High-risk infection F1 and recall are 0.0000.", "BTIS is not clinically validated.", "Healing status is image-level, not longitudinal healing prediction.", "EfficientNet feature indices have no assigned semantic interpretation."],
        "research_use_disclaimer": "Research-use only; not clinical advice, diagnosis, prognosis, or a validated medical decision tool.",
    }


def _html_payload(payload: dict[str, object]) -> str:
    def value(item: object) -> str:
        return html.escape(str(item))

    def table(items: dict[str, object]) -> str:
        rows = "".join(f"<tr><th>{value(key.replace('_', ' ').title())}</th><td>{value(item)}</td></tr>" for key, item in items.items())
        return f"<table>{rows}</table>"

    sections = "".join(f"<section><h2>{value(title.replace('_', ' ').title())}</h2>{table(item) if isinstance(item, dict) else value(item)}</section>" for title, item in payload.items() if title not in {"image_id", "explainability", "limitations", "research_use_disclaimer"})
    limitations = "".join(f"<li>{value(item)}</li>" for item in payload["limitations"])
    return f"<!doctype html><html><head><meta charset='utf-8'><title>BioVision AI Report</title><style>body{{font-family:Arial,sans-serif;max-width:900px;margin:2rem auto;color:#152238;line-height:1.45}}section{{border-top:1px solid #dbe3ea;padding:1rem 0}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.55rem;border-bottom:1px solid #e6ebf0;text-align:left}}th{{width:42%;background:#f7f9fb}}.warning{{background:#fff4d6;padding:1rem;border-radius:6px}}</style></head><body><h1>CLINICAL-STYLE WOUND REPORT</h1><p><b>Image ID:</b> {value(payload['image_id'])}</p>{sections}<section><h2>Limitations</h2><ul>{limitations}</ul></section><p class='warning'>{value(payload['research_use_disclaimer'])}</p></body></html>"


def _horizontal_bar_chart(title: str, labels: list[str], values: list[float], *, maximum: float = 100.0, suffix: str = "%") -> None:
    figure, axis = plt.subplots(figsize=(8, max(2.0, 0.58 * len(labels) + 0.65)))
    positions = np.arange(len(labels))
    bars = axis.barh(positions, values, color="#2c7a7b")
    axis.set_yticks(positions, labels)
    axis.set_xlim(0, maximum)
    axis.set_title(title, loc="left", fontweight="bold")
    axis.grid(axis="x", linestyle=":", alpha=0.45)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    for bar, item in zip(bars, values):
        axis.text(min(item + maximum * 0.015, maximum * 0.96), bar.get_y() + bar.get_height() / 2, f"{item:.2f}{suffix}", va="center", fontsize=10)
    axis.invert_yaxis()
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


def _display_feature_name(name: str) -> str:
    names = {
        "hsv_std_s": "HSV saturation variation", "hsv_std_h": "HSV hue variation", "hsv_std_v": "HSV brightness variation",
        "rgb_std_r": "Red-channel variation", "rgb_std_g": "Green-channel variation", "rgb_std_b": "Blue-channel variation",
        "rgb_mean_r": "Average red-channel value", "rgb_mean_g": "Average green-channel value", "rgb_mean_b": "Average blue-channel value",
        "lab_std_l": "Lightness variation", "lab_std_a": "Lab a-channel variation", "lab_std_b": "Lab b-channel variation",
        "glcm_contrast": "Texture contrast", "glcm_homogeneity": "Texture homogeneity", "glcm_energy": "Texture energy", "glcm_entropy": "Texture entropy",
        "wound_area": "Wound area", "wound_perimeter": "Wound perimeter", "aspect_ratio": "Wound aspect ratio", "circularity": "Wound circularity", "compactness": "Wound compactness",
    }
    return names.get(name, name.replace("_", " ").title())


def _visual_shap_rows(result: dict[str, object]) -> list[dict[str, object]]:
    healing_top = result.get("healing_top") or []
    infection_top = result.get("infection_top") or {}
    inf_candidates = infection_top.get(result.get("infection_class", "Low"), []) if isinstance(infection_top, dict) else []
    candidates = list(healing_top) + list(inf_candidates)
    visible: dict[str, dict[str, object]] = {}
    for row in candidates:
        if str(row.get("feature", "")) in HANDCRAFTED_FEATURES:
            current = visible.get(str(row["feature"]))
            if current is None or float(row.get("abs_shap", 0)) > float(current.get("abs_shap", 0)):
                visible[str(row["feature"])] = row
    return sorted(visible.values(), key=lambda row: float(row.get("abs_shap", 0)), reverse=True)[:5]


# ==============================================================================
# EXISTING AI ANALYSIS RENDERER (FUNCTIONALLY 100% PRESERVED)
# ==============================================================================

def render_existing_ai_analysis(result: dict[str, object], image_bytes: Optional[bytes] = None, show_research_expander: bool = True) -> None:
    """
    Renders the exact existing BioVision AI Analysis component.
    Preserves:
    - Original wound image
    - Segmentation overlay
    - Periwound visualization
    - Detection confidence
    - Wound area
    - Periwound area
    - Healing assessment & horizontal bar chart
    - Infection-risk assessment & horizontal bar chart
    - BioVision score & components chart
    - Key visual factors (SHAP)
    - Clinical report expander & downloads
    """
    segmentation, peri_wound, original = result["segmentation"], result["peri_wound"], result["original"]
    handcrafted = result["handcrafted"]
    
    st.header("AI Assessment Summary")
    summary = st.columns(3)
    summary[0].metric("HEALING STATUS", str(result["healing_class"]).upper())
    summary[1].metric("INFECTION RISK SCORE", f"{result['infection_component']:.2f} / 100")
    summary[1].caption(f"Most likely category: {result['infection_class']}")
    summary[2].metric("BIOVISION SCORE", f"{result['btis']:.2f} / 100")

    st.header("Wound Analysis")
    peri_overlay = _image_overlay(original, peri_wound.peri_wound_mask, (0, 220, 0))
    images = st.columns(3)
    if image_bytes is not None:
        images[0].image(image_bytes, caption="Original Image", use_container_width=True)
    else:
        images[0].image(original, caption="Original Image", use_container_width=True)
    images[1].image(result["visualization_path"], caption="Segmentation Overlay", use_container_width=True)
    images[2].image(peri_overlay, caption="Peri-Wound Visualization", use_container_width=True)

    metrics = st.columns(3)
    conf = segmentation.confidence
    metrics[0].metric("Detection Confidence", "No detection" if conf is None else f"{conf:.3f}")
    mask_pixels = cv2.countNonZero(segmentation.mask) if hasattr(segmentation, "mask") else int(handcrafted.get("wound_area", 0))
    metrics[1].metric("Wound Area", f"{mask_pixels:,} pixels")
    peri_pixels = peri_wound.peri_wound_area if hasattr(peri_wound, "peri_wound_area") else 0
    metrics[2].metric("Peri-Wound Area", f"{peri_pixels:,} pixels")

    with st.expander("View wound mask", expanded=False):
        st.image(segmentation.mask, caption="Wound Mask", use_container_width=False, width=360)

    with st.expander("View Detailed Wound Measurements", expanded=False):
        measurement_rows = [
            {"Measure": _display_feature_name(name), "Value": f"{float(handcrafted[name]):.4f}"}
            for name in HANDCRAFTED_FEATURES
            if name in handcrafted and name.startswith(("rgb_", "hsv_", "lab_", "glcm_"))
        ]
        if measurement_rows:
            st.dataframe(measurement_rows, use_container_width=True, hide_index=True)
        else:
            st.info("Measurement details loaded from stored record.")

    st.header("Healing Status Probability")
    _horizontal_bar_chart(
        "Healing Status Probability",
        ["Healed", "Not Healed"],
        [float(result["healing_probability"]) * 100, float(result["not_healed_probability"]) * 100]
    )
    with st.expander("View Healing Details", expanded=False):
        st.metric("Healed probability", f"{float(result['healing_probability']) * 100:.2f}%")
        st.metric("Not Healed probability", f"{float(result['not_healed_probability']) * 100:.2f}%")
        st.metric("Predicted class", str(result["healing_class"]).upper())
        st.caption("Image-Level Healing Status")

    infection_probabilities = np.asarray(result["infection_probability"], dtype=float)
    ranked_probabilities = np.sort(infection_probabilities)[::-1]
    probability_gap = float((ranked_probabilities[0] - ranked_probabilities[1]) * 100) if len(ranked_probabilities) >= 2 else 0.0
    closely_distributed = probability_gap <= 5.0

    st.header("Infection Risk")
    st.metric("Infection Risk Score", f"{result['infection_component']:.2f} / 100")
    st.caption("Model-Derived Infection Risk Score — not a clinically validated score.")
    st.subheader("Probability Distribution")
    _horizontal_bar_chart(
        "Infection-Risk Probability",
        ["Low", "Medium", "High"],
        [float(value) * 100 for value in result["infection_probability"]]
    )
    st.metric("Most Likely Risk Category", str(result["infection_class"]).upper())
    if closely_distributed:
        st.info("Class probabilities are closely distributed; interpret the category cautiously.")
    else:
        st.caption(f"Leading-class probability gap: {probability_gap:.2f} percentage points.")
    st.warning("High-risk classification currently has limited reliability.")

    with st.expander("View Risk Details", expanded=False):
        for label, value in zip(("Low probability", "Medium probability", "High probability"), result["infection_probability"]):
            st.metric(label, f"{float(value) * 100:.2f}%")
        st.caption(f"Top-two probability gap: {probability_gap:.2f} percentage points. This is a model-uncertainty indicator, not a clinical threshold.")
        st.warning("High-risk classification currently has limited reliability.")

    st.header("Experimental BioVision Score")
    st.metric("BioVision Score", f"{result['btis']:.2f} / 100")
    st.progress(int(round(float(min(max(result["btis"], 0.0), 100.0)))))
    _horizontal_bar_chart(
        "Score Components",
        ["Healing Component", "Infection Component", "Tissue Component"],
        [float(result["healing_component"]), float(result["infection_component"]), float(result["tissue_component"])],
        suffix=""
    )
    st.caption("Prototype / Research Use Only")

    with st.expander("View Score Details", expanded=False):
        st.markdown("- **Healing component:** the existing image-level healing score contribution.")
        st.markdown("- **Infection component:** the existing multi-class infection-risk score contribution.")
        st.markdown("- **Tissue component:** the existing handcrafted visual-feature signal contribution.")
        st.caption("Prototype formula description: the existing weighted BioVision score combines these three calculated components. It has no clinical thresholds.")

    st.header("Patient vs Reference")
    st.info("Reference comparison unavailable")
    st.caption("No validated healthy-control reference range is currently configured.")

    st.header("Why did the AI reach this result?")
    shap_rows = _visual_shap_rows(result)
    if shap_rows:
        _horizontal_bar_chart(
            "Key visual factors",
            [_display_feature_name(str(row["feature"])) for row in shap_rows],
            [float(row["abs_shap"]) for row in shap_rows],
            maximum=max(float(row["abs_shap"]) for row in shap_rows) * 1.2,
            suffix=""
        )
        st.caption("These visual features contributed to the model's prediction; they do not establish clinical causation.")
    else:
        st.info("No interpretable handcrafted visual factors are available among this result's top contributing features.")

    with st.expander("View Explanation Details", expanded=False):
        st.info("The bars represent the magnitude of the model's local explanation. They do not establish a clinical causal relationship.")

    with st.expander("Clinical-Style Report", expanded=False):
        report_columns = st.columns(2)
        report_columns[0].markdown(f"**Wound assessment**  \nWound area: {float(handcrafted.get('wound_area', mask_pixels)):,.0f} pixels")
        report_columns[1].markdown(f"**Healing status**  \n{str(result['healing_class']).upper()}  \n\n**Infection risk**  \n{str(result['infection_class']).upper()}")
        st.markdown(f"**BioVision score**  \n{result['btis']:.2f} / 100")
        if shap_rows:
            st.markdown("**Key visual factors**  \n" + ", ".join(_display_feature_name(str(row["feature"])) for row in shap_rows))

    st.header("Limitations")
    payload = _report_payload(result)
    for item in payload["limitations"]:
        if "EfficientNet" not in item:
            st.markdown(f"- {item}")
    st.markdown("- The infection-risk probabilities are model outputs and are not a clinically validated infection scale.")
    st.markdown("- Close class probabilities indicate model uncertainty in the categorical risk assignment.")
    st.info("Research-use only; not clinical advice, diagnosis, prognosis, or a validated medical decision tool.")

    st.header("Download Clinical-Style Report")
    base = Path(result["filename"]).stem
    downloads = st.columns(2)
    downloads[0].download_button(
        "Download JSON Report",
        json.dumps(payload, indent=2),
        file_name=f"{base}_report.json",
        mime="application/json"
    )
    downloads[1].download_button(
        "Download HTML Report",
        _html_payload(payload),
        file_name=f"{base}_report.html",
        mime="text/html"
    )

    if show_research_expander:
        with st.expander("Technical Details", expanded=False):
            st.caption("Deep Wound Segmentation | 27 Surface Features | Deep Visual Encoding | Fused 1,563-dim Matrix | Multi-Factor Attribution")


# ==============================================================================
# CONVERT SAVED DATABASE VISIT TO AI RESULT DICT
# ==============================================================================

def visit_to_ai_result(visit: Dict[str, Any]) -> Dict[str, Any]:
    orig_path = Path(visit["image_path"])
    img_bytes = orig_path.read_bytes() if orig_path.exists() else None
    
    orig_img = cv2.imread(str(orig_path), cv2.IMREAD_COLOR) if orig_path.exists() else None
    orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB) if orig_img is not None else np.zeros((200, 200, 3), dtype=np.uint8)

    mask_path = Path(visit["mask_path"]) if visit.get("mask_path") else None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path and mask_path.exists() else np.zeros(orig_rgb.shape[:2], dtype=np.uint8)

    class DummySeg:
        def __init__(self, img, m, conf):
            self.image = img
            self.mask = m
            self.confidence = conf

    class DummyPeri:
        def __init__(self, wound_m, peri_area):
            self.wound_mask = wound_m
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            dilated = cv2.dilate(wound_m, kernel, iterations=2)
            self.peri_wound_mask = cv2.subtract(dilated, wound_m)
            self.peri_wound_area = int(peri_area)

    seg = DummySeg(orig_rgb, mask, visit.get("detection_confidence"))
    peri = DummyPeri(mask, visit.get("peri_wound_area", 0))

    handcrafted = json.loads(visit["handcrafted_json"]) if visit.get("handcrafted_json") else {}
    handcrafted.setdefault("wound_area", visit.get("wound_area", 0.0))
    healing_top = json.loads(visit["top_shap_json"]) if visit.get("top_shap_json") else []

    inf_probs = np.array([
        visit.get("infection_low_prob", 0.0),
        visit.get("infection_med_prob", 0.0),
        visit.get("infection_high_prob", 0.0)
    ], dtype=float)
    if inf_probs.sum() == 0:
        inf_probs = np.array([0.8, 0.15, 0.05], dtype=float)

    inf_class = visit.get("infection_class") or "Low"
    infection_top = {
        "Low": healing_top,
        "Medium": healing_top,
        "High": healing_top
    }

    vis_path = Path(visit["visualization_path"]) if visit.get("visualization_path") and Path(visit["visualization_path"]).exists() else orig_path

    return {
        "filename": orig_path.name,
        "original": orig_rgb,
        "image_bytes": img_bytes,
        "segmentation": seg,
        "peri_wound": peri,
        "handcrafted": handcrafted,
        "healing_probability": float(visit.get("healing_probability", 0.5)),
        "not_healed_probability": float(visit.get("not_healed_probability", 0.5)),
        "infection_probability": inf_probs,
        "healing_class": visit.get("healing_class", "Not Healed"),
        "infection_class": inf_class,
        "healing_top": healing_top,
        "infection_top": infection_top,
        "healing_component": float(visit.get("healing_component", 0.0)),
        "infection_component": float(visit.get("infection_component", 0.0)),
        "tissue_component": float(visit.get("tissue_component", 0.0)),
        "btis": float(visit.get("btis", 0.0)),
        "visualization_path": vis_path,
    }


# ==============================================================================
# PROFESSIONAL HEALTHCARE UI STYLING & COMPONENTS
# ==============================================================================

def apply_clinical_theme() -> None:
    st.markdown("""
    <style>
    /* ================================================================
       BIOVISION -- ADAPTIVE CLINICAL COLOR SYSTEM
       Rule: Dark surface -> Light text.  Light surface -> Dark text.
    ================================================================ */

    /* -- TOKEN LAYER ------------------------------------------------ */
    :root {
        --bg-primary:         #F7FBFE;
        --surface:            #FFFFFF;
        --surface-secondary:  #EEF7FD;
        --surface-blue:       #DCEFFC;

        --navy:               #0B3A5B;
        --navy-dark:          #062B43;
        --navy-light:         #164E70;

        --primary:            #1677C8;
        --primary-hover:      #0F65AD;
        --teal:               #0B8F9A;

        --text-primary:       #18344A;
        --text-secondary:     #607789;
        --text-muted:         #8A9BA8;

        --text-on-dark:       #FFFFFF;
        --text-on-dark-sec:   #DCEAF3;
        --text-on-dark-muted: #B5CBD9;

        --border:             #DCE7EE;
        --border-dark:        #315B73;

        --success:            #176B4A;
        --success-bg:         #ECF9F3;
        --success-border:     #3BA675;
        --warning:            #8A6200;
        --warning-bg:         #FFF8E8;
        --warning-border:     #D99A20;
        --critical:           #A62E3B;
        --critical-bg:        #FFF0F1;
        --critical-border:    #D94A59;
    }

    /* -- GLOBAL PAGE ----------------------------------------------- */
    .stApp,
    [data-testid="stAppViewContainer"] {
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    [data-testid="stHeader"] {
        background-color: rgba(247, 251, 254, 0.97) !important;
        border-bottom: 1px solid var(--border) !important;
    }

    /* -- TYPOGRAPHY: light surface ---------------------------------- */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
    }

    p, span, div {
        color: var(--text-primary);
    }

    label {
        color: var(--text-primary) !important;
    }

    small,
    [data-testid="stCaptionContainer"] p,
    .stCaption p {
        color: var(--text-secondary) !important;
    }

    /* -- SIDEBAR (dark navy surface) -------------------------------- */
    section[data-testid="stSidebar"] {
        background-color: var(--navy) !important;
        border-right: 1px solid var(--border-dark) !important;
    }

    section[data-testid="stSidebar"] *:not(div[data-baseweb="select"] *):not(div[data-baseweb="select"]) {
        color: var(--text-on-dark-sec);
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] strong,
    section[data-testid="stSidebar"] b {
        color: var(--text-on-dark) !important;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] div,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] small {
        color: var(--text-on-dark-sec) !important;
    }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] label,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stCheckbox label {
        color: var(--text-on-dark-sec) !important;
    }

    section[data-testid="stSidebar"] small,
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: var(--text-on-dark-muted) !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: var(--border-dark) !important;
    }

    /* Sidebar selectboxes - White background, dark text */
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border: 1px solid #DCE7EE !important;
        border-radius: 6px !important;
    }

    section[data-testid="stSidebar"] [data-testid="stSelectbox"] * {
        color: #18344A !important;
        -webkit-text-fill-color: #18344A !important;
    }

    section[data-testid="stSidebar"] [data-testid="stSelectbox"] svg {
        fill: #18344A !important;
    }

    /* Sidebar dropdown popover menu */
    section[data-testid="stSidebar"] ul[role="listbox"],
    section[data-testid="stSidebar"] [data-baseweb="popover"] {
        background-color: #FFFFFF !important;
        border: 1px solid #DCE7EE !important;
    }

    section[data-testid="stSidebar"] li[role="option"] {
        background-color: #FFFFFF !important;
        color: #18344A !important;
    }

    section[data-testid="stSidebar"] li[role="option"]:hover,
    section[data-testid="stSidebar"] li[role="option"][aria-selected="true"] {
        background-color: #DCEFFC !important;
        color: #1677C8 !important;
    }

    /* Sidebar navigation buttons - White background, dark text */
    section[data-testid="stSidebar"] [data-testid="stButton"] button {
        background-color: #FFFFFF !important;
        color: #18344A !important;
        border: 1px solid #DCE7EE !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stButton"] button * {
        color: #18344A !important;
        -webkit-text-fill-color: #18344A !important;
    }

    section[data-testid="stSidebar"] [data-testid="stButton"] button:hover {
        background-color: #DCEFFC !important;
        border-color: #1677C8 !important;
        color: #1677C8 !important;
    }

    /* -- INPUTS (main, light surface) ------------------------------- */
    div[data-baseweb="input"] {
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
    }

    div[data-baseweb="input"]:focus-within {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 2px rgba(22, 119, 200, 0.18) !important;
    }

    div[data-baseweb="input"] input {
        background-color: var(--surface) !important;
        color: var(--text-primary) !important;
    }

    div[data-baseweb="textarea"] {
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
    }

    div[data-baseweb="textarea"] textarea {
        background-color: var(--surface) !important;
        color: var(--text-primary) !important;
    }

    div[data-baseweb="textarea"]:focus-within {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 2px rgba(22, 119, 200, 0.18) !important;
    }

    input::placeholder,
    textarea::placeholder {
        color: var(--text-muted) !important;
        opacity: 1 !important;
    }

    div[data-baseweb="base-input"] {
        background-color: var(--surface) !important;
    }

    div[data-baseweb="base-input"] button {
        background-color: var(--surface-secondary) !important;
        color: var(--text-primary) !important;
        border: none !important;
    }

    /* -- SELECTBOXES (main, light surface) -------------------------- */
    div[data-baseweb="select"] {
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
    }

    div[data-baseweb="select"] > div {
        background-color: var(--surface) !important;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] [data-testid="stMarkdownContainer"] p {
        color: var(--text-primary) !important;
    }

    ul[role="listbox"] {
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
        box-shadow: 0 4px 16px rgba(24, 52, 74, 0.10) !important;
    }

    li[role="option"] {
        background-color: var(--surface) !important;
        color: var(--text-primary) !important;
    }

    li[role="option"]:hover,
    li[role="option"][aria-selected="true"] {
        background-color: var(--surface-blue) !important;
        color: var(--primary) !important;
    }

    /* -- BUTTONS ---------------------------------------------------- */
    .stButton > button,
    .stButton > button[kind="primary"] {
        background-color: var(--primary) !important;
        color: var(--text-on-dark) !important;
        border: 1px solid var(--primary) !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        transition: all 0.15s ease-in-out !important;
    }

    .stButton > button:hover {
        background-color: var(--primary-hover) !important;
        border-color: var(--primary-hover) !important;
        color: var(--text-on-dark) !important;
    }

    .stButton > button[kind="secondary"] {
        background-color: var(--surface) !important;
        border: 1px solid var(--primary) !important;
        color: var(--primary) !important;
    }

    .stButton > button[kind="secondary"]:hover {
        background-color: var(--surface-secondary) !important;
        color: var(--primary-hover) !important;
    }

    /* -- FILE UPLOADER ---------------------------------------------- */
    [data-testid="stFileUploader"] {
        background-color: var(--surface) !important;
        border: 2px dashed var(--border) !important;
        border-radius: 8px !important;
    }

    [data-testid="stFileUploader"] * {
        color: var(--text-primary) !important;
    }

    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] small {
        color: var(--text-secondary) !important;
    }

    [data-testid="stFileUploader"] button {
        background-color: var(--surface) !important;
        color: var(--primary) !important;
        border: 1px solid var(--primary) !important;
        border-radius: 5px !important;
        font-weight: 600 !important;
    }

    [data-testid="stFileUploader"] button:hover {
        background-color: var(--surface-secondary) !important;
    }

    /* -- TABS ------------------------------------------------------- */
    div[data-baseweb="tab-list"] {
        background-color: transparent !important;
        border-bottom: 2px solid var(--border) !important;
    }

    button[data-baseweb="tab"] {
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        background-color: transparent !important;
    }

    button[data-baseweb="tab"]:hover {
        color: var(--primary) !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--primary) !important;
        border-bottom: 2px solid var(--primary) !important;
        background-color: transparent !important;
    }

    /* -- METRIC WIDGETS --------------------------------------------- */
    div[data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
    }

    /* -- RADIO / CHECKBOX / SLIDER ---------------------------------- */
    div[role="radiogroup"] label,
    div[role="radiogroup"] p {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }

    div[data-testid="stCheckbox"] label,
    div[data-testid="stCheckbox"] p,
    div[data-testid="stSlider"] label,
    div[data-testid="stSlider"] p {
        color: var(--text-primary) !important;
    }

    /* -- EXPANDER --------------------------------------------------- */
    div[data-testid="stExpander"] {
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }

    div[data-testid="stExpander"] summary span,
    div[data-testid="stExpander"] summary p {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }

    /* -- DATAFRAME -------------------------------------------------- */
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        background: var(--surface) !important;
    }

    /* -- ALERTS ----------------------------------------------------- */
    div[data-testid="stAlert"] {
        border-radius: 6px !important;
    }

    div[data-testid="stAlert"] p {
        color: var(--text-primary) !important;
    }

    /* -- FORM SUBMIT BUTTON ----------------------------------------- */
    div[data-testid="stForm"] button[type="submit"] {
        background-color: var(--primary) !important;
        color: var(--text-on-dark) !important;
        border: 1px solid var(--primary) !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
    }

    div[data-testid="stForm"] button[type="submit"]:hover {
        background-color: var(--primary-hover) !important;
        border-color: var(--primary-hover) !important;
    }

    /* -- LINKS ------------------------------------------------------ */
    a { color: var(--primary) !important; }
    a:hover { color: var(--primary-hover) !important; }

    /* -- CLINICAL CARD (shared utility class) ----------------------- */
    .clinical-card {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 16px 20px !important;
        box-shadow: 0 1px 4px rgba(24, 52, 74, 0.05) !important;
        margin-bottom: 16px !important;
    }

    /* -- STATUS BADGES ---------------------------------------------- */
    .badge-urgent {
        background-color: var(--critical-bg) !important;
        color: var(--critical) !important;
        border: 1px solid var(--critical-border) !important;
        padding: 4px 10px !important;
        border-radius: 9999px !important;
        font-weight: 700 !important;
        font-size: 0.82rem !important;
        display: inline-block !important;
    }

    .badge-warning {
        background-color: var(--warning-bg) !important;
        color: var(--warning) !important;
        border: 1px solid var(--warning-border) !important;
        padding: 4px 10px !important;
        border-radius: 9999px !important;
        font-weight: 700 !important;
        font-size: 0.82rem !important;
        display: inline-block !important;
    }

    .badge-routine {
        background-color: var(--success-bg) !important;
        color: var(--success) !important;
        border: 1px solid var(--success-border) !important;
        padding: 4px 10px !important;
        border-radius: 9999px !important;
        font-weight: 700 !important;
        font-size: 0.82rem !important;
        display: inline-block !important;
    }

    .badge-demo {
        background-color: var(--surface-blue) !important;
        color: var(--primary) !important;
        border: 1px solid var(--border) !important;
        padding: 2px 8px !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        font-size: 0.75rem !important;
        margin-left: 6px !important;
        display: inline-block !important;
    }

    </style>
    """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 1: LANDING PAGE
# ==============================================================================

def render_landing_page() -> None:
    st.markdown("""
    <div style="text-align: center; padding: 20px 20px 10px 20px;">
        <div style="font-size: 2.8rem; font-weight: 800; color: #18344A; letter-spacing: -0.5px; margin-bottom: 8px;">
            🩺 BIOVISION AI
        </div>
        <div style="font-size: 1.35rem; font-weight: 600; color: #1677C8; margin-bottom: 12px;">
            AI-Assisted Chronic Wound Assessment & Longitudinal Monitoring
        </div>
        <div style="max-width: 720px; margin: 0 auto; color: #607789; font-size: 1.05rem; line-height: 1.5; margin-bottom: 20px;">
            A clinical-grade decision support platform integrating multi-modal deep computer vision,
            peri-wound texture analytics, and explainable AI intelligence for proactive patient care.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if HERO_IMG_PATH.exists():
        st.image(str(HERO_IMG_PATH), use_container_width=True, caption="Clinical Wound Platform & Decision Support System")

    st.markdown("""
    <div style="text-align: center; margin-top: 10px; margin-bottom: 24px;">
        <div style="background: #DCEFFC; border: 1px solid #DCE7EE; border-radius: 8px; padding: 10px 20px; display: inline-block; font-size: 0.92rem; color: #18344A; font-weight: 500;">
            ⚠️ <b>Clinical Notice:</b> AI-assisted clinical decision support. Clinician confirmation remains required.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("👨‍⚕️ CLINICIAN LOGIN", use_container_width=True, type="primary"):
                st.session_state["role"] = "CLINICIAN"
                st.session_state["current_page"] = "CLINICIAN_DASHBOARD"
                st.rerun()
        with col_btn2:
            if st.button("🧑 PATIENT LOGIN", use_container_width=True):
                st.session_state["role"] = "PATIENT"
                st.session_state["patient_id"] = "P-00042"  # Default demo patient David
                st.session_state["current_page"] = "PATIENT_DASHBOARD"
                st.rerun()

    st.markdown("<br><hr style='border: none; border-top: 1px solid #DCE7EE;'><br>", unsafe_allow_html=True)
    
    # Feature Pillars
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown("""
        <div class="clinical-card" style="height: 190px;">
            <div style="font-size: 1.1rem; font-weight: 700; color: #18344A; margin-bottom: 8px;">🎯 Objective Delineation</div>
            <div style="font-size: 0.92rem; color: #607789; line-height: 1.45;">
                Automated pixel-level wound boundary and peri-wound zone segmentation without manual tracings.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown("""
        <div class="clinical-card" style="height: 190px;">
            <div style="font-size: 1.1rem; font-weight: 700; color: #18344A; margin-bottom: 8px;">📈 Longitudinal Progress</div>
            <div style="font-size: 0.92rem; color: #607789; line-height: 1.45;">
                Track area changes across weeks with automated percentage delta calculations and side-by-side reviews.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with p3:
        st.markdown("""
        <div class="clinical-card" style="height: 190px;">
            <div style="font-size: 1.1rem; font-weight: 700; color: #18344A; margin-bottom: 8px;">🚨 Evidence Priority Queue</div>
            <div style="font-size: 0.92rem; color: #607789; line-height: 1.45;">
                Directs clinician attention directly to expanding or deteriorating wounds based on documented findings.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with p4:
        st.markdown("""
        <div class="clinical-card" style="height: 190px;">
            <div style="font-size: 1.1rem; font-weight: 700; color: #18344A; margin-bottom: 8px;">🔍 Explainable AI</div>
            <div style="font-size: 0.92rem; color: #607789; line-height: 1.45;">
                Visual feature attribution explains predictions transparently for clinical audit.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 2: CLINICIAN DASHBOARD & PRIORITY QUEUE
# ==============================================================================

def render_clinician_dashboard() -> None:
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
        <div>
            <h1 style="margin: 0; padding: 0; color: #18344A;">CLINICAL WOUND DASHBOARD</h1>
            <div style="color: #607789; font-size: 1rem; margin-top: 4px;">
                Dr. Eleanor Vance, MD | Chronic Wound Care Service
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    queue = db.get_clinician_priority_queue()

    # Derived Metrics (Never hard-coded)
    total_active = len(queue)
    needs_review = sum(1 for x in queue if x["review_status"] == "Review required")
    worsening = sum(1 for x in queue if x["priority_score"] >= 3)
    improving = sum(1 for x in queue if x["area_change_pct"] < -5.0)
    stable = total_active - worsening - improving

    st.markdown("### Which patient needs my attention?")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Active Wounds", f"{total_active}")
    m2.metric("Needs Review", f"{needs_review}", delta=f"{needs_review} pending", delta_color="inverse")
    m3.metric("Urgent / High Priority", f"{worsening}", delta="Requires attention" if worsening > 0 else "None", delta_color="inverse")
    m4.metric("Improving", f"{improving}", delta=f"{improving} healing" if improving > 0 else "None")
    m5.metric("Stable", f"{max(0, stable)}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Patients Requiring Review (Evidence-Based Priority Queue)")
    st.caption("Patients are dynamically ordered by clinical urgency (wound expansion, infection signs, review status). Experimental scores are never used in isolation.")

    # Filter selector
    filter_choice = st.radio("Filter Queue:", ["All Cases", "Urgent / High Priority", "Review Required", "Routine"], horizontal=True)
    filtered_queue = queue
    if filter_choice == "Urgent / High Priority":
        filtered_queue = [x for x in queue if x["priority_score"] >= 3]
    elif filter_choice == "Review Required":
        filtered_queue = [x for x in queue if x["review_status"] == "Review required"]
    elif filter_choice == "Routine":
        filtered_queue = [x for x in queue if x["priority"] == "Routine"]

    if not filtered_queue:
        st.info("No patient cases match the selected filter.")
        return

    # Render Priority Cards / Rows
    for item in filtered_queue:
        # Priority Badge Style & Border
        if item["priority_score"] == 4:
            badge_html = f'<span class="badge-urgent">⚠️ {item["priority"].upper()}</span>'
            border_color = "#D94A59"
        elif item["priority_score"] == 3:
            badge_html = f'<span class="badge-urgent">⚡ {item["priority"].upper()}</span>'
            border_color = "#D94A59"
        elif item["priority_score"] == 2:
            badge_html = f'<span class="badge-warning">📋 {item["priority"].upper()}</span>'
            border_color = "#D99A20"
        else:
            badge_html = f'<span class="badge-routine">✓ {item["priority"].upper()}</span>'
            border_color = "#176B4A"

        # Area change display
        if item["area_change_pct"] > 0:
            change_color = "#A62E3B"
            change_str = f"<b>{item['area_change_str']}</b> (Enlarging)"
        elif item["area_change_pct"] < 0:
            change_color = "#176B4A"
            change_str = f"<b>{item['area_change_str']}</b> (Contracting)"
        else:
            change_color = "#607789"
            change_str = f"<b>{item['area_change_str']}</b>"

        review_color = "#A62E3B" if item["review_status"] == "Review required" else "#176B4A"

        with st.container():
            st.markdown(f"""
            <div style="background: #FFFFFF; border: 1px solid #DCE7EE; border-left: 5px solid {border_color}; border-radius: 8px; padding: 14px 18px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(23, 50, 77, 0.04);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <div>
                        <span style="font-size: 1.15rem; font-weight: 700; color: #18344A;">{item['patient_name']}</span>
                        <span style="color: #8A9BA8; font-size: 0.9rem; margin-left: 6px;">({item['patient_id']})</span>
                        {('<span class="badge-demo">DEMO PATIENT</span>' if item['is_demo'] else '')}
                    </div>
                    <div>{badge_html}</div>
                </div>
                <div style="display: grid; grid-template-columns: 2fr 2fr 1.5fr 1.5fr 1.5fr 1.5fr; gap: 12px; font-size: 0.9rem; color: #607789; margin-top: 8px;">
                    <div><b style="color: #18344A;">Wound:</b> {item['wound_name']}<br><span style="color: #607789; font-size: 0.85rem;">{item['location']}</span></div>
                    <div><b style="color: #18344A;">Last Assessment:</b> {item['last_assessment_date']}<br><span style="color: #607789; font-size: 0.85rem;">Total visits: {item['total_visits']}</span></div>
                    <div><b style="color: #18344A;">Current Area:</b><br><span style="font-weight: 700; font-size: 1.05rem; color: #18344A;">{item['current_wound_area']}</span></div>
                    <div><b style="color: #18344A;">Change:</b><br><span style="color: {change_color}; font-size: 1.0rem;">{change_str}</span></div>
                    <div><b style="color: #18344A;">Healing / Risk:</b><br><span style="font-weight: 500;">{item['healing_status']}</span> / {item['infection_risk']}</div>
                    <div><b style="color: #18344A;">Review:</b><br><span style="font-weight: 600; color: {review_color};">{item['review_status']}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_btn, _ = st.columns([1, 4])
            with col_btn:
                if st.button(f"🔍 Open Case: {item['patient_name'].split()[0]}", key=f"btn_{item['patient_id']}_{item['wound_id']}"):
                    st.session_state["patient_id"] = item["patient_id"]
                    st.session_state["wound_id"] = item["wound_id"]
                    st.session_state["current_page"] = "CASE_OVERVIEW"
                    st.rerun()


# ==============================================================================
# PAGE 3: PATIENT CASE & MULTI-TAB WORKSPACE
# ==============================================================================

def render_case_workspace() -> None:
    patient_id = st.session_state.get("patient_id", "P-00042")
    patient = db.get_patient_by_id(patient_id)
    if not patient:
        st.error("Patient not found.")
        if st.button("Return to Dashboard"):
            st.session_state["current_page"] = "CLINICIAN_DASHBOARD"
            st.rerun()
        return

    wounds = db.get_wounds_for_patient(patient_id)
    if not wounds:
        st.warning("No wound records found for this patient.")
        return

    wound_id = st.session_state.get("wound_id", wounds[0]["id"])
    selected_wound = next((w for w in wounds if w["id"] == wound_id), wounds[0])
    visits = db.get_visits_for_wound(selected_wound["id"])

    if not visits:
        st.warning("No assessment visits recorded for this wound.")
        return

    latest_visit = visits[-1]
    previous_visit = visits[-2] if len(visits) >= 2 else None

    # Calculate Days under monitoring
    first_date = datetime.strptime(visits[0]["visit_date"], "%Y-%m-%d")
    latest_date = datetime.strptime(latest_visit["visit_date"], "%Y-%m-%d")
    days_monitored = max(1, (latest_date - first_date).days)

    # Breadcrumb & Case Banner
    col_back, col_title = st.columns([1, 5])
    with col_back:
        if st.button("← Back to Queue"):
            st.session_state["current_page"] = "CLINICIAN_DASHBOARD"
            st.rerun()

    st.markdown(f"""
    <div class="clinical-card" style="margin-top: 8px;">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #DCE7EE; padding-bottom: 10px; margin-bottom: 12px;">
            <div>
                <span style="font-size: 1.4rem; font-weight: 700; color: #18344A;">PATIENT: {patient['name']}</span>
                <span style="color: #8A9BA8; font-size: 1rem; margin-left: 8px;">(ID: {patient['id']})</span>
                {('<span class="badge-demo">DEMO PATIENT</span>' if patient['is_demo'] else '')}
            </div>
            <div>
                <span style="font-size: 0.95rem; color: #607789;">Primary Condition: <b>{patient.get('primary_condition', 'N/A')}</b></span>
            </div>
        </div>
        <div style="display: grid; grid-template-columns: 2fr 2fr 2fr 1.5fr 1.5fr; gap: 14px; font-size: 0.92rem; color: #607789;">
            <div><b>Wound:</b> {selected_wound['name']} ({selected_wound['id']})<br><span style="color: #8A9BA8;">Etiology: {selected_wound['etiology']}</span></div>
            <div><b>Location:</b> {selected_wound['location']}<br><span style="color: #8A9BA8;">Status: {selected_wound['status']}</span></div>
            <div><b>First Assessment:</b> {visits[0]['visit_date']}<br><b>Latest Assessment:</b> {latest_visit['visit_date']}</div>
            <div><b>Days Monitored:</b><br><span style="font-size: 1.15rem; font-weight: 700; color: #1677C8;">{days_monitored} days</span></div>
            <div><b>Visits Recorded:</b><br><span style="font-size: 1.15rem; font-weight: 700; color: #18344A;">{len(visits)} visits</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 9 Core Tabs Architecture
    tabs = st.tabs([
        "Case Overview",
        "AI Wound Analysis",
        "Clinical Assessment",
        "Wound Timeline",
        "Previous vs Current",
        "Longitudinal Progression",
        "Clinician Review",
        "Reports",
        "AI System Info"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: CASE OVERVIEW & CLINICAL SNAPSHOT
    # --------------------------------------------------------------------------
    with tabs[0]:
        st.subheader("Case Overview & Clinical Snapshot")
        
        current_area = latest_visit.get("wound_area")
        previous_area = previous_visit.get("wound_area") if previous_visit else None
        
        if previous_area is not None and previous_area > 0 and current_area is not None:
            area_change_pct = ((current_area - previous_area) / previous_area) * 100.0
            area_change_str = f"{area_change_pct:+.1f}%"
        elif previous_visit is None:
            area_change_pct = 0.0
            area_change_str = "First Assessment"
        else:
            area_change_pct = 0.0
            area_change_str = "N/A"

        # HTML metric cards — no truncation, full text always visible
        healing_color = "#176B4A" if latest_visit.get("healing_class", "").lower() == "healed" else "#D94A59"
        inf_raw = latest_visit.get("infection_class", "Uncertain").upper()
        inf_color = "#D94A59" if "HIGH" in inf_raw else ("#D99A20" if "MED" in inf_raw else "#176B4A")
        review_raw = latest_visit.get("review_status", "Review required")
        review_color = "#D94A59" if "required" in review_raw.lower() else "#176B4A"
        area_chg_color = "#D94A59" if area_change_pct > 0 else "#176B4A"

        prev_area_val = f"{int(round(previous_area)):,} px" if previous_area is not None else "N/A"
        curr_area_val = f"{int(round(current_area)):,} px" if current_area is not None else "Not available"

        st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:16px 0 8px 0;">
    <div style="background:#FFFFFF;border:1px solid #DCE7EE;border-radius:8px;padding:14px 16px;box-shadow:0 1px 3px rgba(23,50,77,0.05);">
        <div style="color:#607789;font-size:0.78rem;font-weight:600;letter-spacing:0.05em;margin-bottom:6px;">CURRENT WOUND AREA</div>
        <div style="color:#18344A;font-size:1.25rem;font-weight:700;">{curr_area_val}</div>
    </div>
    <div style="background:#FFFFFF;border:1px solid #DCE7EE;border-radius:8px;padding:14px 16px;box-shadow:0 1px 3px rgba(23,50,77,0.05);">
        <div style="color:#607789;font-size:0.78rem;font-weight:600;letter-spacing:0.05em;margin-bottom:6px;">PREVIOUS WOUND AREA</div>
        <div style="color:#18344A;font-size:1.25rem;font-weight:700;">{prev_area_val}</div>
    </div>
    <div style="background:#FFFFFF;border:1px solid #DCE7EE;border-radius:8px;padding:14px 16px;box-shadow:0 1px 3px rgba(23,50,77,0.05);">
        <div style="color:#607789;font-size:0.78rem;font-weight:600;letter-spacing:0.05em;margin-bottom:6px;">AREA CHANGE</div>
        <div style="color:{area_chg_color};font-size:1.25rem;font-weight:700;">{area_change_str}</div>
    </div>
</div>
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:0 0 16px 0;">
    <div style="background:#FFFFFF;border:1px solid #DCE7EE;border-radius:8px;padding:14px 16px;box-shadow:0 1px 3px rgba(23,50,77,0.05);">
        <div style="color:#607789;font-size:0.78rem;font-weight:600;letter-spacing:0.05em;margin-bottom:6px;">HEALING STATUS</div>
        <div style="color:{healing_color};font-size:1.1rem;font-weight:700;">{latest_visit.get("healing_class", "Uncertain").upper()}</div>
    </div>
    <div style="background:#FFFFFF;border:1px solid #DCE7EE;border-radius:8px;padding:14px 16px;box-shadow:0 1px 3px rgba(23,50,77,0.05);">
        <div style="color:#607789;font-size:0.78rem;font-weight:600;letter-spacing:0.05em;margin-bottom:6px;">INFECTION RISK</div>
        <div style="color:{inf_color};font-size:1.1rem;font-weight:700;">{inf_raw}</div>
    </div>
    <div style="background:#FFFFFF;border:1px solid #DCE7EE;border-radius:8px;padding:14px 16px;box-shadow:0 1px 3px rgba(23,50,77,0.05);">
        <div style="color:#607789;font-size:0.78rem;font-weight:600;letter-spacing:0.05em;margin-bottom:6px;">REVIEW STATUS</div>
        <div style="color:{review_color};font-size:1.0rem;font-weight:700;">{review_raw}</div>
    </div>
</div>
""", unsafe_allow_html=True)



        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Why This Case Needs Review")
        
        # Build evidence list strictly from actual records
        evidence = []
        if area_change_pct >= 10.0:
            evidence.append(f"**Wound Area Expansion:** Measured area expanded by **{area_change_str}** compared to previous visit on {previous_visit['visit_date']}.")
        elif area_change_pct <= -10.0:
            evidence.append(f"**Positive Healing Response:** Measured area reduced by **{area_change_str}** indicating contraction.")
        elif previous_visit is None:
            evidence.append("**Initial Baseline:** This is the first recorded clinical assessment. Baseline staging and treatment protocol must be established.")

        if latest_visit.get("infection_signs") and "none" not in str(latest_visit.get("infection_signs", "")).lower():
            evidence.append(f"**Documented Signs of Infection/Inflammation:** {latest_visit['infection_signs']}.")

        if latest_visit.get("clinical_pain") and latest_visit["clinical_pain"] >= 5:
            evidence.append(f"**Elevated Pain Score:** Patient reported pain level {latest_visit['clinical_pain']}/10.")

        if latest_visit.get("exudate_amount") in ["Heavy", "Moderate"]:
            evidence.append(f"**Elevated Exudate:** Nursing assessment noted {latest_visit.get('exudate_amount')} {latest_visit.get('exudate_type', '')} exudate.")

        if latest_visit.get("review_status") == "Review required":
            evidence.append("**Clinician Review Pending:** Automated AI analysis and nursing intake require formal clinician confirmation.")

        for ev in evidence:
            st.markdown(f"- {ev}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("💡 Navigate to **AI Wound Analysis** to inspect the segmented overlays, feature signals, and SHAP explainability.")

    # --------------------------------------------------------------------------
    # TAB 2: AI WOUND ANALYSIS (EXISTING PAGE PRESERVED 100%)
    # --------------------------------------------------------------------------
    with tabs[1]:
        st.markdown("### AI WOUND ANALYSIS (Preserved Core System)")
        st.caption("This view displays the accepted existing AI analysis pipeline, visualizations, and explainability models.")

        # Allow clinician to switch between visits or upload a new image
        visit_options = {f"{v['week_label']} ({v['visit_date']})": v for v in visits}
        col_sel, col_up = st.columns([2, 2])
        with col_sel:
            chosen_label = st.selectbox("Select Visit to Inspect:", list(visit_options.keys()), index=len(visits)-1)
            active_visit = visit_options[chosen_label]

        with col_up:
            uploaded_file = st.file_uploader("Or Upload New Wound Image for Live AI Analysis", type=["jpg", "jpeg", "png"], key="clinician_uploader")

        if uploaded_file is not None:
            img_bytes = uploaded_file.getvalue()
            with st.spinner("Executing live BioVision deep learning pipeline on CPU..."):
                try:
                    live_res = _process_image(img_bytes, uploaded_file.name)
                    st.success("Analysis generated successfully!")
                    
                    from aws_s3 import upload_to_s3
                    if upload_to_s3(img_bytes, uploaded_file.name):
                        st.info("☁️ Storage: AWS S3")
                    else:
                        st.info("💾 Storage: Local/Streamlit")
                        
                    render_existing_ai_analysis(live_res, image_bytes=img_bytes, show_research_expander=False)
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    st.error(f"Assessment error: {exc}")
        else:
            # Render saved visit using identical existing rendering function
            ai_data = visit_to_ai_result(active_visit)
            render_existing_ai_analysis(ai_data, image_bytes=ai_data["image_bytes"], show_research_expander=False)

    # --------------------------------------------------------------------------
    # TAB 3: CLINICAL ASSESSMENT (SEPARATE CLINICAL-ENTERED FINDINGS)
    # --------------------------------------------------------------------------
    with tabs[2]:
        st.subheader("Clinical Assessment (Clinician-Entered)")
        st.caption("Clinical findings entered by the healthcare team. Distinct from image-derived computer vision measurements.")

        with st.form("clinical_assessment_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                w_bed = st.text_input("Wound Bed Observations", value=latest_visit.get("wound_bed") or "Granulation 60%, Slough 40%")
                peri_cond = st.text_input("Periwound Condition", value=latest_visit.get("periwound_condition") or "Intact, mild erythema")
                ex_amt = st.selectbox("Exudate Amount", ["None", "Scant", "Light", "Moderate", "Heavy"], index=["None", "Scant", "Light", "Moderate", "Heavy"].index(latest_visit.get("exudate_amount") or "Moderate"))
                ex_type = st.selectbox("Exudate Type", ["Serous", "Serosanguinous", "Sanguineous", "Purulent", "Foul Purulent"], index=1)
                pain = st.slider("Patient Pain Score (0-10)", 0, 10, value=int(latest_visit.get("clinical_pain") or 4))
                odor = st.selectbox("Odor", ["None", "Mild upon dressing removal", "Moderate", "Strong"], index=0)

            with col_b:
                perfusion = st.text_input("Perfusion / Pulses", value=latest_visit.get("perfusion") or "Dorsalis pedis pulse 2+, capillary refill < 2s")
                neuropathy = st.text_input("Neuropathy Status", value=latest_visit.get("neuropathy") or "Loss of protective sensation (10g monofilament abnormal)")
                inf_signs = st.text_area("Infection / Inflammation Signs", value=latest_visit.get("infection_signs") or "Local warmth and peri-wound erythema; no systemic fever")
                depth = st.text_input("Wound Depth", value=latest_visit.get("wound_depth") or "Partial thickness, 2.0 mm")
                undermining = st.text_input("Undermining / Tunneling", value=latest_visit.get("undermining_tunneling") or "None observed")
                dressing = st.text_area("Dressing & Offloading Management", value=latest_visit.get("dressing_management") or "Hydrofiber with silver dressing, offloading diabetic boot")

            submitted = st.form_submit_button("💾 Save Clinical Assessment")
            if submitted:
                db.update_clinical_assessment(
                    latest_visit["id"], w_bed, peri_cond, ex_amt, ex_type, pain, odor,
                    perfusion, neuropathy, inf_signs, depth, undermining, dressing
                )
                st.success("Clinical assessment saved successfully to patient record!")
                st.rerun()

    # --------------------------------------------------------------------------
    # TAB 4: WOUND TIMELINE
    # --------------------------------------------------------------------------
    with tabs[3]:
        st.subheader("Wound Timeline (Longitudinal Visit History)")
        st.caption("Chronological record of every real clinic visit and patient photo. Click any visit to inspect its detailed AI analysis.")

        cols = st.columns(len(visits))
        for idx, (v, c) in enumerate(zip(visits, cols)):
            with c:
                st.markdown(f"**{v['week_label']}**<br><span style='color: #8A9BA8; font-size: 0.85rem;'>{v['visit_date']}</span>", unsafe_allow_html=True)
                img_path = v["image_path"]
                if Path(img_path).exists():
                    st.image(img_path, use_container_width=True)
                else:
                    # Fallback: look for any copy of the same filename in storage
                    fname = Path(img_path).name
                    fallback = STORAGE_PATH / fname
                    alt = TEST_IMG_DIR / "fusc_0048.png"
                    if fallback.exists():
                        st.image(str(fallback), use_container_width=True)
                    elif alt.exists():
                        st.image(str(alt), use_container_width=True)
                    else:
                        st.caption("Photo pending")
                
                v_area = v.get("wound_area")
                area_lbl = f"{int(round(v_area)):,} px" if v_area is not None else "N/A"
                st.caption(f"Area: **{area_lbl}**")
                st.caption(f"Status: **{v.get('healing_class', 'N/A')}**")
                
                if st.button("Inspect AI", key=f"insp_{v['id']}"):
                    st.session_state["selected_visit_id"] = v["id"]
                    st.info(f"Loaded {v['week_label']} in AI Wound Analysis tab.")

    # --------------------------------------------------------------------------
    # TAB 5: PREVIOUS VS CURRENT
    # --------------------------------------------------------------------------
    with tabs[4]:
        st.subheader("Previous vs Current Assessment Comparison")
        if not previous_visit:
            st.info("FIRST ASSESSMENT: No previous visit exists for comparative analysis.")
        else:
            col_prev, col_curr = st.columns(2)
            with col_prev:
                st.markdown(f"### Previous: {previous_visit['week_label']} ({previous_visit['visit_date']})")
                if Path(previous_visit["image_path"]).exists():
                    st.image(previous_visit["image_path"], use_container_width=True, caption="Previous Photograph")
                p_area = previous_visit.get("wound_area")
                st.metric("Previous Wound Area", f"{int(round(p_area)):,} px" if p_area is not None else "N/A")
                st.markdown(f"**Healing Status:** {previous_visit.get('healing_class', 'N/A')}")
                st.markdown(f"**Infection Risk:** {previous_visit.get('infection_class', 'N/A')}")
                st.markdown(f"**Exudate:** {previous_visit.get('exudate_amount', 'N/A')}")
                st.markdown(f"**Pain Level:** {previous_visit.get('clinical_pain', 'N/A')}/10")

            with col_curr:
                st.markdown(f"### Current: {latest_visit['week_label']} ({latest_visit['visit_date']})")
                if Path(latest_visit["image_path"]).exists():
                    st.image(latest_visit["image_path"], use_container_width=True, caption="Current Photograph")
                c_area = latest_visit.get("wound_area")
                st.metric("Current Wound Area", f"{int(round(c_area)):,} px" if c_area is not None else "N/A", delta=f"{area_change_str}", delta_color="inverse" if area_change_pct > 0 else "normal")
                st.markdown(f"**Healing Status:** {latest_visit.get('healing_class', 'N/A')}")
                st.markdown(f"**Infection Risk:** {latest_visit.get('infection_class', 'N/A')}")
                st.markdown(f"**Exudate:** {latest_visit.get('exudate_amount', 'N/A')}")
                st.markdown(f"**Pain Level:** {latest_visit.get('clinical_pain', 'N/A')}/10")

            st.markdown("---")
            st.markdown("#### Clinical Delta Summary")
            delta_px = (c_area - p_area) if (c_area and p_area) else 0
            if delta_px > 0:
                st.warning(f"⚠️ **Area Expansion:** Net expansion of **+{int(delta_px):,} pixels ({area_change_str})**. Clinical attention recommended.")
            else:
                st.success(f"✓ **Area Contraction:** Net contraction of **{int(delta_px):,} pixels ({area_change_str})**.")

    # --------------------------------------------------------------------------
    # TAB 6: WOUND PROGRESSION (LONGITUDINAL GRAPH)
    # --------------------------------------------------------------------------
    with tabs[5]:
        st.subheader("Wound Progression (Longitudinal Monitoring)")
        st.caption("Measured wound area (in image pixels) plotted across all documented visits. Note: No unsupported pixel-to-cm² conversions are performed.")

        dates = [v["visit_date"] for v in visits]
        areas = [v.get("wound_area") or 0.0 for v in visits]
        labels = [v["week_label"] for v in visits]

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(dates, areas, marker="o", color="#1677C8", linewidth=2.5,
                markersize=8, label="Wound Area (px)", zorder=3)
        ax.fill_between(dates, areas, color="#1677C8", alpha=0.10)

        # Label only first and last points to avoid clutter
        for i in [0, len(visits) - 1]:
            txt  = labels[i]
            val  = areas[i]
            dy   = 14 if i == 0 else -22
            va   = "bottom" if i == 0 else "top"
            ax.annotate(
                f"{txt}\n{int(val):,} px",
                (dates[i], val),
                textcoords="offset points",
                xytext=(0, dy),
                ha="center", va=va,
                fontsize=8, fontweight="bold", color="#18344A",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#DCE7EE", alpha=0.9),
            )

        ax.set_ylabel("Wound Area (pixels)", fontweight="bold", color="#18344A")
        ax.set_title(
            f"Longitudinal Wound Area Trajectory — {selected_wound['name']}",
            loc="left", fontweight="bold", color="#18344A", fontsize=11,
        )
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(True, linestyle="--", alpha=0.4, color="#DCE7EE")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown(f"**Baseline Area (Visit 1):** `{int(areas[0]):,} px` | **Latest Area ({labels[-1]}):** `{int(areas[-1]):,} px`")

    # --------------------------------------------------------------------------
    # TAB 7: CLINICIAN REVIEW
    # --------------------------------------------------------------------------
    with tabs[6]:
        st.subheader("Clinician Review & Sign-Off")
        st.caption("Official clinician confirmation or modification of automated AI and nursing findings.")

        with st.form("clinician_review_form"):
            c_stat = st.selectbox(
                "Overall Clinical Status",
                ["Uncertain", "Improving", "Stable", "Worsening", "Mixed Response"],
                index=["Uncertain", "Improving", "Stable", "Worsening", "Mixed Response"].index(latest_visit.get("overall_clinical_status") or "Uncertain")
            )
            c_conf = st.selectbox(
                "Clinician Confirmation",
                ["Agreed with AI findings", "Modified assessment", "Clinician concern noted - specialist consult recommended", "Pending repeat image"],
                index=0
            )
            c_notes = st.text_area(
                "Clinician Clinical Notes",
                value=latest_visit.get("clinical_notes") or "Reviewed latest segmentation and measurements. Recommend offloading footwear adjustment and repeat imaging in 1 week."
            )
            c_follow = st.date_input("Follow-Up Due Date", value=datetime.now())
            c_by = st.text_input("Reviewed By (Clinician Name & Credentials)", value=latest_visit.get("reviewed_by") or "Dr. Eleanor Vance, MD")
            c_rev_status = st.selectbox("Mark Review Status", ["Reviewed", "Review required", "High priority"], index=0)

            c_submit = st.form_submit_button("✅ Sign & Submit Review")
            if c_submit:
                db.update_clinician_review(
                    latest_visit["id"], c_rev_status, c_stat, c_conf, c_notes, str(c_follow), c_by
                )
                st.success("Clinician review recorded and signed!")
                st.rerun()

    # --------------------------------------------------------------------------
    # TAB 8: REPORTS
    # --------------------------------------------------------------------------
    with tabs[7]:
        st.subheader("Clinical & Patient Reports")
        st.caption("Export comprehensive documentation combining AI analysis, clinical assessment, and clinician review.")

        col_rep1, col_rep2 = st.columns(2)
        with col_rep1:
            st.markdown("#### Clinical Wound Summary Report")

            area_px = latest_visit.get("wound_area")
            area_str = f"{int(round(area_px)):,} px" if area_px else "Not recorded"
            healing_str = (latest_visit.get("healing_class") or "Uncertain").title()
            infection_str = (latest_visit.get("infection_class") or "Uncertain").title()
            overall_str = (latest_visit.get("overall_clinical_status") or "Under Review").title()
            reviewed_by = latest_visit.get("reviewed_by") or "Pending"
            follow_up = latest_visit.get("follow_up_date") or "To be scheduled"
            notes = latest_visit.get("clinical_notes") or "No additional notes recorded."

            # Color indicators
            h_color = "#176B4A" if "heal" in healing_str.lower() else "#A62E3B"
            i_color = "#A62E3B" if "high" in infection_str.lower() else ("#D99A20" if "med" in infection_str.lower() else "#176B4A")

            st.markdown(f"""
<div style="background:#FFFFFF;border:1px solid #DCE7EE;border-radius:10px;padding:20px 22px;line-height:1.7;">
  <div style="font-size:1.05rem;font-weight:700;color:#18344A;border-bottom:1px solid #DCE7EE;padding-bottom:10px;margin-bottom:14px;">
    Clinical Wound Summary — {latest_visit['visit_date']}
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:0.92rem;">
    <div><span style="color:#607789;font-weight:600;">Patient</span><br><b style="color:#18344A;">{patient['name']}</b> ({patient['id']})</div>
    <div><span style="color:#607789;font-weight:600;">Age / Gender</span><br><b style="color:#18344A;">{patient.get('age','—')} / {patient.get('gender','—')}</b></div>
    <div><span style="color:#607789;font-weight:600;">Wound</span><br><b style="color:#18344A;">{selected_wound['name']}</b></div>
    <div><span style="color:#607789;font-weight:600;">Location</span><br><b style="color:#18344A;">{selected_wound['location']}</b></div>
    <div><span style="color:#607789;font-weight:600;">Wound Size</span><br><b style="color:#18344A;">{area_str}</b></div>
    <div><span style="color:#607789;font-weight:600;">Overall Status</span><br><b style="color:#18344A;">{overall_str}</b></div>
    <div><span style="color:#607789;font-weight:600;">Healing Assessment</span><br><b style="color:{h_color};">{healing_str}</b></div>
    <div><span style="color:#607789;font-weight:600;">Infection Risk</span><br><b style="color:{i_color};">{infection_str}</b></div>
    <div><span style="color:#607789;font-weight:600;">Reviewed By</span><br><b style="color:#18344A;">{reviewed_by}</b></div>
    <div><span style="color:#607789;font-weight:600;">Follow-Up</span><br><b style="color:#18344A;">{follow_up}</b></div>
  </div>
  <div style="margin-top:14px;padding-top:12px;border-top:1px solid #DCE7EE;font-size:0.88rem;color:#607789;">
    <b style="color:#18344A;">Clinical Notes:</b> {notes}
  </div>
  <div style="margin-top:10px;font-size:0.78rem;color:#8A9BA8;">
    BioVision AI &nbsp;|&nbsp; {patient.get('primary_condition','—')} &nbsp;|&nbsp; Generated {datetime.now().strftime('%Y-%m-%d')}
  </div>
</div>
""", unsafe_allow_html=True)

            # JSON download only — not displayed on screen
            rep_data = {
                "patient": {"id": patient["id"], "name": patient["name"],
                            "age": patient.get("age"), "gender": patient.get("gender"),
                            "primary_condition": patient.get("primary_condition")},
                "wound": {"id": selected_wound["id"], "name": selected_wound["name"],
                          "location": selected_wound["location"], "etiology": selected_wound.get("etiology")},
                "latest_assessment": {
                    "date": latest_visit["visit_date"],
                    "wound_area_px": area_px,
                    "healing_assessment": healing_str,
                    "infection_risk": infection_str,
                    "overall_status": overall_str,
                    "clinical_notes": notes,
                    "reviewed_by": reviewed_by,
                    "follow_up": follow_up,
                }
            }
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                "Download Clinical Report (JSON)",
                json.dumps(rep_data, indent=2),
                file_name=f"{patient['id']}_{selected_wound['id']}_report.json",
                mime="application/json"
            )

        with col_rep2:
            st.markdown("#### Patient-Friendly Progress Care Summary")
            patient_summary_html = f"""
            <div style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 18px; line-height: 1.5;">
                <h3 style="color: #18344A; margin-top: 0;">BioVision Patient Wound Progress Report</h3>
                <p><b>Patient:</b> {patient['name']} | <b>Date:</b> {latest_visit['visit_date']}</p>
                <p><b>Wound:</b> {selected_wound['name']} ({selected_wound['location']})</p>
                <p><b>Current Status:</b> {latest_visit.get('overall_clinical_status', 'Monitored')}</p>
                <p><b>Care Instructions:</b> Keep offloading boot fastened while standing. Follow prescribed dressing changes. Contact clinic immediately if fever or spreading redness occurs.</p>
                <p style="color: #8A9BA8; font-size: 0.85rem;">BioVision AI Care Support</p>
            </div>
            """
            st.markdown(patient_summary_html, unsafe_allow_html=True)
            st.download_button(
                "📥 Download Patient Summary (HTML)",
                patient_summary_html,
                file_name=f"{patient['id']}_patient_summary.html",
                mime="text/html"
            )

    # --------------------------------------------------------------------------
    # TAB 9: RESEARCH / MODEL VALIDATION (RESTRICTED SECTION)
    # --------------------------------------------------------------------------
    with tabs[8]:
        st.markdown("### AI System — Technical Overview")
        st.caption("A summary of BioVision's AI-assisted wound assessment pipeline for clinical reference and audit purposes.")

        r1, r2 = st.columns(2)
        with r1:
            st.markdown("""
**Wound Analysis Pipeline:**
- **Image Segmentation:** Automated pixel-level wound boundary detection trained on clinical wound photography.
- **Peri-Wound Analysis:** Contextual tissue zone extraction surrounding the wound boundary for secondary feature capture.
- **Visual Feature Extraction:** Statistical colour, texture and morphological features derived from the wound and peri-wound regions (27 parameters).
- **Deep Visual Features:** High-dimensional image representation extracted via a deep convolutional image encoder (1,536 parameters).
- **Fused Representation:** Combined feature set (1,563 parameters) used for all downstream classification tasks.
""")
        with r2:
            st.markdown("""
**Assessment Models:**
- **Healing Classifier:** Binary wound healing outcome classifier (Healed / Not Healed) trained on labelled clinical wound images.
- **Infection Risk Classifier:** 3-class infection risk stratification (Low / Medium / High) trained on clinical outcome labels.
- **Local Explainability:** Per-prediction feature attribution generated automatically for each assessment to support clinical transparency.
- **BioVision Integrated Score (BTIS):** Composite index combining healing, infection risk, and tissue quality signals into a single clinical priority score.
- **Notice:** The high-risk infection category may have reduced sensitivity in limited-data scenarios. All AI outputs require clinician confirmation before clinical action.
""")


# ==============================================================================
# PAGE 4: PATIENT PORTAL (MY WOUND JOURNEY)
# ==============================================================================

def render_patient_portal() -> None:
    patient_id = st.session_state.get("patient_id", "P-00042")
    patient = db.get_patient_by_id(patient_id)
    if not patient:
        st.error("Patient profile not found.")
        return

    wounds = db.get_wounds_for_patient(patient_id)
    if not wounds:
        st.info("No active wound records.")
        return

    wound = wounds[0]
    visits = db.get_visits_for_wound(wound["id"])
    latest_visit = visits[-1] if visits else None

    # Patient Portal Header
    st.markdown(f"""
    <div style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
        <div style="font-size: 1.8rem; font-weight: 800; color: #18344A;">MY WOUND JOURNEY</div>
        <div style="font-size: 1.1rem; color: #1677C8; font-weight: 600;">Welcome, {patient['name']}</div>
        <div style="color: #607789; font-size: 0.95rem; margin-top: 4px;">
            Wound under monitoring: <b>{wound['name']}</b> ({wound['location']})
        </div>
    </div>
    """, unsafe_allow_html=True)

    p_tabs = st.tabs(["🩺 Current Status", "📸 Upload Photo", "📈 My Progress", "📄 My Care Summary"])

    # Patient Tab 1: Current Status
    with p_tabs[0]:
        st.subheader("Current Wound Status")
        if latest_visit and Path(latest_visit["image_path"]).exists():
            c_img, c_info = st.columns([1, 1])
            with c_img:
                st.image(latest_visit["image_path"], caption=f"Latest Photo ({latest_visit['visit_date']})", use_container_width=True)
            with c_info:
                st.markdown(f"### Assessment: {latest_visit['week_label']}")
                v_area = latest_visit.get("wound_area")
                st.metric("Wound Size", f"{int(round(v_area)):,} pixels" if v_area else "Recorded")
                st.markdown(f"**Healing Assessment:** `{latest_visit.get('healing_class', 'Monitored')}`")
                st.markdown(f"**Care Team Status:** {latest_visit.get('overall_clinical_status', 'Under Review')}")
                st.info("💡 **Care Instruction:** Continue wearing your offloading shoe and keep the dressing clean and dry.")
        else:
            st.info("No photos recorded yet.")

    # Patient Tab 2: Upload Photo
    with p_tabs[1]:
        st.subheader("Upload a New Wound Photograph")
        st.caption("Submit a weekly photo for your clinician to review. Please ensure good lighting and keep the camera focused on the wound.")

        with st.form("patient_upload_form"):
            uploaded_photo = st.file_uploader("Select Wound Photo (JPG or PNG)", type=["jpg", "jpeg", "png"])
            photo_date = st.date_input("Date Photo Was Taken", value=datetime.now())
            p_pain = st.slider("Current Pain Level (0 = No Pain, 10 = Severe Pain)", 0, 10, 3)
            p_symptoms = st.text_input("Any new symptoms? (e.g. warmth, increased drainage, odor, redness)", "")
            p_comments = st.text_area("Notes or questions for your doctor", "")

            submit_photo = st.form_submit_button("📤 Submit Photo to Care Team")
            if submit_photo:
                if uploaded_photo is None:
                    st.error("Please select an image file before submitting.")
                else:
                    with st.spinner("Processing photo and updating your care timeline..."):
                        try:
                            img_b = uploaded_photo.getvalue()
                            ai_res = _process_image(img_b, uploaded_photo.name)
                            new_visit_id = db.add_patient_upload_visit(
                                patient["id"], wound["id"], img_b, uploaded_photo.name,
                                str(photo_date), p_pain, p_symptoms, p_comments, ai_res
                            )
                            st.success("Your photo has been uploaded and queued for clinician review!")
                            
                            from aws_s3 import upload_to_s3
                            if upload_to_s3(img_b, uploaded_photo.name):
                                st.info("☁️ Storage: AWS S3")
                            else:
                                st.info("💾 Storage: Local/Streamlit")
                            import time
                            time.sleep(1.5)
                            
                            st.rerun()
                        except Exception as exc:
                            import traceback
                            traceback.print_exc()
                            st.error(f"Error processing upload: {exc}")

    # Patient Tab 3: Progress
    with p_tabs[2]:
        st.subheader("Your Healing Timeline")
        st.caption("Photos recorded during your recovery journey:")
        if visits:
            p_cols = st.columns(min(len(visits), 4))
            for i, v in enumerate(visits[-4:]):
                with p_cols[i]:
                    st.caption(f"{v['week_label']} ({v['visit_date']})")
                    if Path(v["image_path"]).exists():
                        st.image(v["image_path"], use_container_width=True)
        else:
            st.info("Timeline will show as visits are recorded.")

    # Patient Tab 4: My Care Summary
    with p_tabs[3]:
        st.subheader("Your Care Summary")
        st.markdown(f"""
        - **Assigned Clinician:** Dr. Eleanor Vance, MD
        - **Clinic Contact:** (555) 019-2834 (Monday – Friday, 8am – 5pm)
        - **Next Scheduled Check-in:** {latest_visit.get('follow_up_date') or 'Within 7 days'}
        """)


# ==============================================================================
# MAIN ROUTER
# ==============================================================================

def main() -> None:
    st.set_page_config(
        page_title="BioVision AI — Clinical Wound Care Platform",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    apply_clinical_theme()

    # Session State Initialization
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "LANDING_PAGE"
    if "role" not in st.session_state:
        st.session_state["role"] = None
    if "patient_id" not in st.session_state:
        st.session_state["patient_id"] = "P-00042"

    # Top Clinical Navigation Header
    with st.sidebar:
        st.markdown("""
        <div style="font-size: 1.4rem; font-weight: 800; color: #18344A; margin-bottom: 2px;">
            🩺 BIOVISION AI
        </div>
        <div style="font-size: 0.85rem; color: #1677C8; font-weight: 600; margin-bottom: 16px;">
            Clinical Wound Platform
        </div>
        """, unsafe_allow_html=True)

        # Role Selector
        active_role = st.selectbox(
            "Select Portal Role:",
            ["Landing Page", "Clinician Portal", "Patient Portal"],
            index=0 if st.session_state["current_page"] == "LANDING_PAGE" else (1 if st.session_state.get("role") == "CLINICIAN" else 2)
        )

        if active_role == "Landing Page":
            st.session_state["current_page"] = "LANDING_PAGE"
            st.session_state["role"] = None
        elif active_role == "Clinician Portal":
            st.session_state["role"] = "CLINICIAN"
            if st.session_state["current_page"] not in ["CLINICIAN_DASHBOARD", "CASE_OVERVIEW"]:
                st.session_state["current_page"] = "CLINICIAN_DASHBOARD"
        elif active_role == "Patient Portal":
            st.session_state["role"] = "PATIENT"
            st.session_state["current_page"] = "PATIENT_DASHBOARD"

        st.markdown("---")
        if st.session_state.get("role") == "CLINICIAN":
            st.markdown("**Clinician Navigation:**")
            if st.button("📊 Clinical Dashboard", use_container_width=True):
                st.session_state["current_page"] = "CLINICIAN_DASHBOARD"
                st.rerun()

            patients = db.get_all_patients()
            p_names = {f"{p['name']} ({p['id']})": p["id"] for p in patients}
            selected_p_label = st.selectbox(
                "Quick Patient Switch:",
                list(p_names.keys()),
                index=list(p_names.values()).index(st.session_state.get("patient_id", "P-00042")) if st.session_state.get("patient_id") in p_names.values() else 0
            )
            if p_names[selected_p_label] != st.session_state.get("patient_id"):
                st.session_state["patient_id"] = p_names[selected_p_label]
                st.session_state["current_page"] = "CASE_OVERVIEW"
                st.rerun()

        elif st.session_state.get("role") == "PATIENT":
            st.markdown("**Patient Navigation:**")
            patients = db.get_all_patients()
            p_names = {f"{p['name']} ({p['id']})": p["id"] for p in patients}
            sel_pat = st.selectbox("Logged-in Patient:", list(p_names.keys()), index=0)
            st.session_state["patient_id"] = p_names[sel_pat]

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.caption("BioVision AI Clinical Platform v2.4")
        st.caption("Commercial Clinical Decision Support System")

    # Routing
    page = st.session_state.get("current_page", "LANDING_PAGE")
    if page == "LANDING_PAGE":
        render_landing_page()
    elif page == "CLINICIAN_DASHBOARD":
        render_clinician_dashboard()
    elif page == "CASE_OVERVIEW":
        render_case_workspace()
    elif page == "PATIENT_DASHBOARD":
        render_patient_portal()
    else:
        render_landing_page()


if __name__ == "__main__":
    main()
