import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from label_validation import REQUIRED_COLUMNS


OUTPUTS_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = OUTPUTS_DIR / "clinical_labels_template.csv"
SOURCE_PATH = OUTPUTS_DIR / "clinical_labels_source.csv"
FUSED_FEATURES_PATH = OUTPUTS_DIR / "fused_features.csv"
CLINICAL_LABELS_PATH = OUTPUTS_DIR / "clinical_labels.csv"


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return reader.fieldnames or [], list(reader)


def _validate_source(
    rows: list[dict[str, str]],
    columns: list[str],
    fused_rows: list[dict[str, str]],
) -> list[str]:
    errors: list[str] = []
    if columns != list(REQUIRED_COLUMNS):
        errors.append(
            "clinical_labels_source.csv must contain exactly: "
            + ",".join(REQUIRED_COLUMNS)
        )

    filenames = [row.get("filename", "") for row in rows]
    missing_filenames = [str(index + 2) for index, filename in enumerate(filenames) if not filename.strip()]
    if missing_filenames:
        errors.append(f"missing filenames on CSV rows: {', '.join(missing_filenames)}")

    duplicates = sorted(filename for filename, count in Counter(filenames).items() if filename and count > 1)
    if duplicates:
        errors.append(f"duplicate filenames: {', '.join(duplicates)}")

    missing_healing = [row.get("filename", "") for row in rows if not row.get("healing_outcome", "").strip()]
    if missing_healing:
        errors.append(f"missing healing_outcome for: {', '.join(missing_healing)}")

    missing_infection = [row.get("filename", "") for row in rows if not row.get("infection_status", "").strip()]
    if missing_infection:
        errors.append(f"missing infection_status for: {', '.join(missing_infection)}")

    label_ids = {filename for filename in filenames if filename}
    fused_ids = {row.get("filename", "") for row in fused_rows}
    unmatched_labels = sorted(label_ids - fused_ids)
    unmatched_features = sorted(fused_ids - label_ids)
    if unmatched_labels:
        errors.append(f"filenames not found in fused_features.csv: {', '.join(unmatched_labels)}")
    if unmatched_features:
        errors.append(f"fused-feature filenames without clinical labels: {', '.join(unmatched_features)}")
    return errors


def prepare_clinical_labels() -> None:
    if not TEMPLATE_PATH.is_file():
        print(f"Required template is missing: {TEMPLATE_PATH}")
        return
    template_columns, _ = _read_csv(TEMPLATE_PATH)
    if template_columns != list(REQUIRED_COLUMNS):
        print("The clinical label template does not contain the required columns.")
        return

    if not SOURCE_PATH.is_file():
        print(
            "A genuine clinical annotation file is required at "
            f"{SOURCE_PATH}. Stop without creating clinical_labels.csv."
        )
        return
    if not FUSED_FEATURES_PATH.is_file():
        print(f"Fused feature file is missing: {FUSED_FEATURES_PATH}")
        return

    source_columns, source_rows = _read_csv(SOURCE_PATH)
    fused_columns, fused_rows = _read_csv(FUSED_FEATURES_PATH)
    if "filename" not in fused_columns:
        print("fused_features.csv must contain a filename column.")
        return

    errors = _validate_source(source_rows, source_columns, fused_rows)
    if errors:
        print("Clinical label validation failed:")
        for error in errors:
            print(f"- {error}")
        return

    shutil.copyfile(SOURCE_PATH, CLINICAL_LABELS_PATH)
    print(f"Validated genuine labels and created {CLINICAL_LABELS_PATH} without altering values.")


if __name__ == "__main__":
    prepare_clinical_labels()
