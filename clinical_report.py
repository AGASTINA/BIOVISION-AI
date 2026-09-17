import csv
import html
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT / "outputs" / "SurgWound"
BTIS_PATH = ROOT / "btis" / "btis_results.csv"
PREDICTIONS_PATH = ROOT / "prediction_results.csv"
HANDCRAFTED_PATH = ROOT / "handcrafted_features.csv"
HEALING_SHAP_PATH = ROOT / "shap" / "healing_feature_importance.csv"
INFECTION_SHAP_PATH = ROOT / "shap" / "infection_feature_importance.csv"
BTIS_SUMMARY_PATH = ROOT / "btis" / "btis_summary.json"
INFECTION_METRICS_PATH = ROOT / "infection_metrics.json"
REPORT_ROOT = ROOT / "reports"

INDEX_FIELDS = [
    "filename",
    "predicted_healing",
    "healed_probability",
    "not_healed_probability",
    "predicted_infection",
    "infection_low_probability",
    "infection_medium_probability",
    "infection_high_probability",
    "BTIS",
    "report_json",
    "report_html",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _feature_group(feature: str) -> str:
    if feature.startswith(("rgb_", "hsv_", "lab_")):
        return "handcrafted color"
    if feature.startswith("glcm_"):
        return "handcrafted texture"
    if feature in {"wound_area", "wound_perimeter", "aspect_ratio", "circularity", "compactness"}:
        return "handcrafted shape"
    if feature.startswith("efficientnet_"):
        return "EfficientNet deep features"
    return "other"


def _top_features(rows: list[dict[str, str]], limit: int = 5) -> list[dict[str, str | float]]:
    return [
        {
            "feature": row["feature"],
            "mean_abs_shap": float(row["mean_abs_shap"]),
            "mean_shap": float(row["mean_shap"]),
            "direction": "positive" if float(row["mean_shap"]) > 0 else "negative" if float(row["mean_shap"]) < 0 else "neutral",
            "group": _feature_group(row["feature"]),
        }
        for row in rows[:limit]
    ]


def _html_list(items: list[dict[str, str | float]]) -> str:
    return "<ul>" + "".join(
        f"<li><code>{html.escape(str(item['feature']))}</code>: mean |SHAP| {float(item['mean_abs_shap']):.6f}; {item['direction']}; {item['group']}</li>"
        for item in items
    ) + "</ul>"


def _html_report(report: dict[str, object]) -> str:
    healing = report["healing_assessment"]
    infection = report["infection_risk_assessment"]
    btis = report["btis"]
    wound = report["wound_assessment"]
    peri = report["peri_wound_assessment"]
    explain = report["explainability"]
    return f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><title>BioVision report {html.escape(str(report['image_id']))}</title>
<style>body{{font-family:Arial,sans-serif;max-width:900px;margin:2rem auto;line-height:1.45}}section{{border-top:1px solid #ddd;padding:1rem 0}}code{{background:#f3f4f6;padding:.1rem .25rem}}.warning{{background:#fff4d6;padding:1rem}}</style></head>
<body><h1>BioVision Clinical Research Report</h1><p><strong>Image ID:</strong> {html.escape(str(report['image_id']))}</p>
<section><h2>Wound Assessment</h2><p>{html.escape(str(wound['description']))}</p><pre>{html.escape(json.dumps(wound['measurements'], indent=2))}</pre></section>
<section><h2>Peri-wound Assessment</h2><p>{html.escape(str(peri['description']))}</p><pre>{html.escape(json.dumps(peri['measurements'], indent=2))}</pre></section>
<section><h2>Healing Assessment</h2><p>Image-level healing status classification: <strong>{html.escape(str(healing['predicted_class']))}</strong></p><pre>{html.escape(json.dumps(healing, indent=2))}</pre></section>
<section><h2>Infection-risk Assessment</h2><p>Three-class infection-risk output: <strong>{html.escape(str(infection['predicted_class']))}</strong></p><pre>{html.escape(json.dumps(infection, indent=2))}</pre></section>
<section><h2>Experimental BioVision Tissue/Healing Infection Score (prototype)</h2><pre>{html.escape(json.dumps(btis, indent=2))}</pre></section>
<section><h2>Explainability</h2><h3>Healing</h3>{_html_list(explain['healing_top_features'])}<h3>Infection risk</h3>{''.join(f'<h4>{html.escape(str(class_name))}</h4>{_html_list(items)}' for class_name, items in explain['infection_top_features'].items())}</section>
<section class=\"warning\"><h2>Model Limitations</h2><ul>{''.join(f'<li>{html.escape(str(item))}</li>' for item in report['model_limitations'])}</ul></section>
<section><h2>Research-use Disclaimer</h2><p>{html.escape(str(report['research_use_disclaimer']))}</p></section>
</body></html>"""


def generate_reports() -> None:
    btis_rows = _read_csv(BTIS_PATH)
    prediction_rows = {row["filename"]: row for row in _read_csv(PREDICTIONS_PATH)}
    handcrafted_rows = {row["filename"]: row for row in _read_csv(HANDCRAFTED_PATH)}
    healing_shap = _top_features(_read_csv(HEALING_SHAP_PATH))
    infection_shap_rows = _read_csv(INFECTION_SHAP_PATH)
    infection_shap = {
        class_name: _top_features([row for row in infection_shap_rows if row["class"] == class_name])
        for class_name in ("Low", "Medium", "High")
    }
    btis_summary = json.loads(BTIS_SUMMARY_PATH.read_text(encoding="utf-8"))
    infection_metrics = json.loads(INFECTION_METRICS_PATH.read_text(encoding="utf-8"))
    high_metrics = infection_metrics["test"]["classification_report"]["High"]
    json_root = REPORT_ROOT / "json"
    html_root = REPORT_ROOT / "html"
    json_root.mkdir(parents=True, exist_ok=True)
    html_root.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, str]] = []

    for btis in btis_rows:
        filename = btis["filename"]
        prediction = prediction_rows[filename]
        features = handcrafted_rows[filename]
        measurements = {
            name: float(features[name])
            for name in (
                "wound_area",
                "wound_perimeter",
                "aspect_ratio",
                "circularity",
                "compactness",
                "rgb_mean_r",
                "rgb_mean_g",
                "rgb_mean_b",
                "hsv_mean_h",
                "hsv_mean_s",
                "hsv_mean_v",
                "glcm_contrast",
                "glcm_homogeneity",
                "glcm_energy",
                "glcm_entropy",
            )
        }
        report = {
            "image_id": filename,
            "wound_assessment": {
                "description": "Measured wound-mask geometry and masked visual features; these measurements are not diagnoses.",
                "measurements": {name: measurements[name] for name in ("wound_area", "wound_perimeter", "aspect_ratio", "circularity", "compactness")},
            },
            "peri_wound_assessment": {
                "description": "Available masked color and texture measurements from the peri-wound region; these measurements are not diagnoses.",
                "measurements": {name: measurements[name] for name in measurements if name not in {"wound_area", "wound_perimeter", "aspect_ratio", "circularity", "compactness"}},
            },
            "healing_assessment": {
                "description": "Image-level healing status classification, not future healing prediction.",
                "healed_probability": float(btis["healed_probability"]),
                "not_healed_probability": float(btis["not_healed_probability"]),
                "predicted_class": prediction["predicted_healing"],
            },
            "infection_risk_assessment": {
                "description": "Three-class infection-risk classification; not a binary infection/no-infection determination.",
                "low_probability": float(btis["infection_low_probability"]),
                "medium_probability": float(btis["infection_medium_probability"]),
                "high_probability": float(btis["infection_high_probability"]),
                "predicted_class": prediction["predicted_infection"],
            },
            "btis": {
                "label": "Experimental BioVision Tissue/Healing Infection Score (prototype)",
                "value": float(btis["BTIS"]),
                "healing_component": float(btis["healing_component"]),
                "infection_component": float(btis["infection_component"]),
                "tissue_component": float(btis["tissue_component"]),
                "formula": btis_summary["formula"],
                "clinical_validation": False,
            },
            "explainability": {
                "healing_top_features": healing_shap,
                "infection_top_features": infection_shap,
            },
            "model_limitations": [
                f"High-risk infection F1 = {high_metrics['f1-score']:.4f}; high-risk recall = {high_metrics['recall']:.4f}.",
                "High-risk predictions must not be presented as clinically reliable.",
                "The infection model preserves Low, Medium, and High outputs but has weak minority-class performance.",
                "SurgWound Healing Status is an image-level label, not longitudinal healing evidence.",
            ],
            "research_use_disclaimer": "Research-use only. This automated report is not clinical advice, diagnosis, prognosis, or a validated medical decision tool. BTIS is an experimental non-clinical prototype.",
        }
        json_path = json_root / f"{Path(filename).stem}.json"
        html_path = html_root / f"{Path(filename).stem}.html"
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        html_path.write_text(_html_report(report), encoding="utf-8")
        index_rows.append({
            "filename": filename,
            "predicted_healing": prediction["predicted_healing"],
            "healed_probability": btis["healed_probability"],
            "not_healed_probability": btis["not_healed_probability"],
            "predicted_infection": prediction["predicted_infection"],
            "infection_low_probability": btis["infection_low_probability"],
            "infection_medium_probability": btis["infection_medium_probability"],
            "infection_high_probability": btis["infection_high_probability"],
            "BTIS": btis["BTIS"],
            "report_json": str(json_path.relative_to(ROOT)),
            "report_html": str(html_path.relative_to(ROOT)),
        })

    index_path = REPORT_ROOT / "report_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(index_rows)
    _validate_outputs(index_rows, json_root, html_root)
    print("REPORT_VALIDATION = PASS")
    print(f"reports_generated: {len(index_rows)}")
    print(f"json_directory: {json_root}")
    print(f"html_directory: {html_root}")
    print(f"index: {index_path}")
    print("limitations: research-use only; BTIS is prototype/non-clinical; High-risk infection predictions are not clinically reliable.")


def _validate_outputs(index_rows: list[dict[str, str]], json_root: Path, html_root: Path) -> None:
    if len(index_rows) != 137 or len({row["filename"] for row in index_rows}) != 137:
        raise ValueError("Report index does not contain exactly 137 unique filenames")
    for row in index_rows:
        report = json.loads((json_root / f"{Path(row['filename']).stem}.json").read_text(encoding="utf-8"))
        required = {"image_id", "healing_assessment", "infection_risk_assessment", "btis", "explainability", "model_limitations", "research_use_disclaimer"}
        if not required <= set(report) or not (html_root / f"{Path(row['filename']).stem}.html").is_file():
            raise ValueError(f"Incomplete report: {row['filename']}")
        if "healed_probability" not in report["healing_assessment"] or not {"low_probability", "medium_probability", "high_probability"} <= set(report["infection_risk_assessment"]):
            raise ValueError(f"Missing probabilities: {row['filename']}")
        disclaimer = report["research_use_disclaimer"].lower()
        if "prototype" not in report["btis"]["label"].lower() or "not clinical advice" not in disclaimer:
            raise ValueError(f"Missing BTIS prototype disclaimer: {row['filename']}")
        if not any("high-risk" in limitation.lower() for limitation in report["model_limitations"]):
            raise ValueError(f"Missing infection limitation: {row['filename']}")


if __name__ == "__main__":
    generate_reports()
