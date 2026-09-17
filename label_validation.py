import csv
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = ("filename", "healing_outcome", "infection_status")


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return reader.fieldnames or [], list(reader)


def _distribution(rows: list[dict[str, str]], column: str) -> dict[str, int]:
    return dict(sorted(Counter(row.get(column, "") for row in rows).items()))


def validate_clinical_labels(
    labels_path: str | Path = "outputs/clinical_labels.csv",
    fused_features_path: str | Path = "outputs/fused_features.csv",
) -> dict[str, Any]:
    labels_path = Path(labels_path).resolve()
    fused_features_path = Path(fused_features_path).resolve()
    report: dict[str, Any] = {
        "labels_path": labels_path,
        "file_exists": labels_path.is_file(),
        "required_columns": list(REQUIRED_COLUMNS),
        "missing_columns": [],
        "duplicate_filenames": [],
        "missing_labels": [],
        "healing_outcome_distribution": {},
        "infection_status_distribution": {},
        "fused_features_exists": fused_features_path.is_file(),
        "matched_samples": [],
        "unmatched_label_samples": [],
        "unmatched_feature_samples": [],
    }
    if not report["file_exists"]:
        return report

    columns, rows = _read_csv(labels_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in columns]
    report["missing_columns"] = missing_columns
    if missing_columns:
        return report

    filenames = [row["filename"] for row in rows]
    filename_counts = Counter(filenames)
    report["duplicate_filenames"] = sorted(
        filename for filename, count in filename_counts.items() if count > 1
    )
    report["missing_labels"] = [
        {
            "filename": row["filename"],
            "missing_columns": [
                column for column in REQUIRED_COLUMNS[1:] if not row[column].strip()
            ],
        }
        for row in rows
        if any(not row[column].strip() for column in REQUIRED_COLUMNS[1:])
    ]
    report["healing_outcome_distribution"] = _distribution(rows, "healing_outcome")
    report["infection_status_distribution"] = _distribution(rows, "infection_status")

    if not report["fused_features_exists"]:
        return report
    fused_columns, fused_rows = _read_csv(fused_features_path)
    if "filename" not in fused_columns:
        report["fused_features_missing_filename"] = True
        return report

    label_ids = set(filenames)
    fused_ids = {row["filename"] for row in fused_rows}
    report["matched_samples"] = sorted(label_ids & fused_ids)
    report["unmatched_label_samples"] = sorted(label_ids - fused_ids)
    report["unmatched_feature_samples"] = sorted(fused_ids - label_ids)
    return report
