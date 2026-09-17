import csv
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


DATASET_ROOT = Path("data/clinical_dataset")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
METADATA_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".json", ".txt"}
CANDIDATE_FIELDS = (
    "filename",
    "image_id",
    "patient_id",
    "healing",
    "healing_outcome",
    "healing_status",
    "infection",
    "infection_status",
    "wound_status",
    "outcome",
    "diagnosis",
    "severity",
    "follow_up",
    "followup",
)


def _normalise(value: Any) -> str:
    return str(value).strip().lower()


def _rows_from_delimited(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as data_file:
        sample = data_file.read(4096)
        data_file.seek(0)
        dialect = csv.excel_tab if path.suffix.lower() == ".tsv" else csv.Sniffer().sniff(sample)
        reader = csv.DictReader(data_file, dialect=dialect)
        return reader.fieldnames or [], list(reader)


def _json_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("records", data.get("data", [data]))
    if not isinstance(data, list):
        return [], []
    rows = [item for item in data if isinstance(item, dict)]
    fields = sorted({key for row in rows for key in row})
    return fields, [{key: str(row.get(key, "")) for key in fields} for row in rows]


def _xlsx_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    namespace = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("main:si", namespace):
                shared_strings.append("".join(node.text or "" for node in item.iter() if node.tag.endswith("}t")))
        sheet_name = next(name for name in archive.namelist() if name.startswith("xl/worksheets/sheet"))
        root = ElementTree.fromstring(archive.read(sheet_name))
        matrix: list[list[str]] = []
        for row in root.findall(".//main:row", namespace):
            values: list[str] = []
            for cell in row.findall("main:c", namespace):
                value = cell.find("main:v", namespace)
                text = value.text if value is not None else ""
                if cell.get("t") == "s" and text:
                    text = shared_strings[int(text)]
                values.append(text or "")
            matrix.append(values)
    if not matrix:
        return [], []
    fields = matrix[0]
    return fields, [dict(zip(fields, values)) for values in matrix[1:]]


def _read_metadata(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt"}:
        return _rows_from_delimited(path)
    if suffix == ".json":
        return _json_rows(path)
    if suffix == ".xlsx":
        return _xlsx_rows(path)
    return [], []


def _column_matches(columns: list[str]) -> list[str]:
    candidates = {_normalise(field): field for field in columns}
    return [candidates[field] for field in CANDIDATE_FIELDS if field in candidates]


def audit_clinical_dataset(dataset_root: str | Path = DATASET_ROOT) -> dict[str, Any]:
    root = Path(dataset_root)
    report: dict[str, Any] = {
        "dataset_directory_exists": root.is_dir(),
        "image_count": 0,
        "metadata_file_count": 0,
        "metadata_files": [],
        "metadata_columns": {},
        "metadata_record_counts": {},
        "possible_image_metadata_matches": [],
        "candidate_clinical_fields": {},
        "healing_labels_exist": False,
        "infection_labels_exist": False,
        "images_with_both_labels": 0,
        "missing_clinical_labels": [],
        "duplicate_image_ids": [],
        "duplicate_filenames": [],
    }
    if not root.is_dir():
        return report

    images = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    metadata_files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in METADATA_EXTENSIONS]
    image_names = {_normalise(path.name) for path in images}
    report["image_count"] = len(images)
    report["metadata_file_count"] = len(metadata_files)
    all_filenames: list[str] = []
    all_image_ids: list[str] = []
    matched: set[str] = set()

    for path in metadata_files:
        try:
            columns, rows = _read_metadata(path)
        except (
            OSError,
            ValueError,
            KeyError,
            IndexError,
            csv.Error,
            json.JSONDecodeError,
            zipfile.BadZipFile,
            ElementTree.ParseError,
        ):
            columns, rows = [], []
        relative_path = str(path.relative_to(root))
        candidates = _column_matches(columns)
        report["metadata_files"].append(relative_path)
        report["metadata_columns"][relative_path] = columns
        report["metadata_record_counts"][relative_path] = len(rows)
        report["candidate_clinical_fields"][relative_path] = candidates

        normalised_columns = {_normalise(column): column for column in columns}
        filename_column = normalised_columns.get("filename")
        image_id_column = normalised_columns.get("image_id")
        healing_column = next((normalised_columns.get(field) for field in ("healing", "healing_outcome", "healing_status") if field in normalised_columns), None)
        infection_column = next((normalised_columns.get(field) for field in ("infection", "infection_status") if field in normalised_columns), None)
        for row in rows:
            filename = _normalise(row.get(filename_column, "")) if filename_column else ""
            image_id = _normalise(row.get(image_id_column, "")) if image_id_column else ""
            if filename:
                all_filenames.append(filename)
                if filename in image_names or Path(filename).name in image_names:
                    matched.add(filename)
            if image_id:
                all_image_ids.append(image_id)
            if healing_column:
                report["healing_labels_exist"] |= bool(row.get(healing_column, "").strip())
            if infection_column:
                report["infection_labels_exist"] |= bool(row.get(infection_column, "").strip())
            if healing_column and infection_column and row.get(healing_column, "").strip() and row.get(infection_column, "").strip():
                report["images_with_both_labels"] += 1
            missing = [field for field, column in (("healing", healing_column), ("infection", infection_column)) if column and not row.get(column, "").strip()]
            if missing and filename:
                report["missing_clinical_labels"].append({"filename": filename, "missing": missing})

    report["possible_image_metadata_matches"] = sorted(matched)
    report["duplicate_image_ids"] = sorted(item for item, count in Counter(all_image_ids).items() if count > 1)
    report["duplicate_filenames"] = sorted(item for item, count in Counter(all_filenames).items() if count > 1)
    return report


def print_audit(report: dict[str, Any]) -> None:
    if not report["dataset_directory_exists"]:
        print("Clinical dataset directory not found.")
        return
    print(f"Images: {report['image_count']}")
    print(f"Metadata files: {report['metadata_file_count']}")
    print(f"Metadata columns: {report['metadata_columns']}")
    print(f"Metadata records: {report['metadata_record_counts']}")
    print(f"Candidate clinical fields: {report['candidate_clinical_fields']}")
    print(f"Possible image-to-metadata matches: {report['possible_image_metadata_matches']}")
    print(f"Healing labels exist: {report['healing_labels_exist']}")
    print(f"Infection labels exist: {report['infection_labels_exist']}")
    print(f"Images with both labels: {report['images_with_both_labels']}")
    print(f"Missing clinical labels: {report['missing_clinical_labels']}")
    print(f"Duplicate image IDs: {report['duplicate_image_ids']}")
    print(f"Duplicate filenames: {report['duplicate_filenames']}")


if __name__ == "__main__":
    print_audit(audit_clinical_dataset())
