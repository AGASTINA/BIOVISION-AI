import csv
from pathlib import Path
from typing import Iterable


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or "filename" not in reader.fieldnames:
            raise ValueError(f"CSV must contain a filename column: {path}")
        rows = list(reader)
    return reader.fieldnames, rows


def _validate_rows(
    path: Path,
    rows: Iterable[dict[str, str]],
) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        image_id = row["filename"]
        if not image_id:
            raise ValueError(f"Empty filename in {path}")
        if image_id in indexed:
            raise ValueError(f"Duplicate filename in {path}: {image_id}")
        indexed[image_id] = row
    return indexed


def fuse_feature_csvs(
    handcrafted_path: str | Path,
    efficientnet_path: str | Path,
    output_path: str | Path,
) -> tuple[Path, dict[str, int]]:
    handcrafted_path = Path(handcrafted_path)
    efficientnet_path = Path(efficientnet_path)
    handcrafted_fields, handcrafted_rows = _read_rows(handcrafted_path)
    efficientnet_fields, efficientnet_rows = _read_rows(efficientnet_path)
    handcrafted = _validate_rows(handcrafted_path, handcrafted_rows)
    efficientnet = _validate_rows(efficientnet_path, efficientnet_rows)

    handcrafted_ids = set(handcrafted)
    efficientnet_ids = set(efficientnet)
    if handcrafted_ids != efficientnet_ids:
        missing_from_efficientnet = sorted(handcrafted_ids - efficientnet_ids)
        missing_from_handcrafted = sorted(efficientnet_ids - handcrafted_ids)
        raise ValueError(
            "Mismatched image IDs: "
            f"missing from EfficientNet={missing_from_efficientnet}, "
            f"missing from handcrafted={missing_from_handcrafted}"
        )

    efficientnet_feature_fields = [field for field in efficientnet_fields if field != "filename"]
    handcrafted_feature_fields = [field for field in handcrafted_fields if field != "filename"]
    if set(handcrafted_feature_fields) & set(efficientnet_feature_fields):
        raise ValueError("Feature column names overlap between input CSVs")

    fieldnames = ["filename", *handcrafted_feature_fields, *efficientnet_feature_fields]
    fused_rows: list[dict[str, str]] = []
    missing_values = 0
    for image_id in sorted(handcrafted_ids):
        fused_row = {"filename": image_id}
        fused_row.update({field: handcrafted[image_id][field] for field in handcrafted_feature_fields})
        fused_row.update({field: efficientnet[image_id][field] for field in efficientnet_feature_fields})
        missing_values += sum(value == "" for value in fused_row.values())
        fused_rows.append(fused_row)

    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fused_rows)

    return destination, {
        "matched_images": len(fused_rows),
        "handcrafted_features": len(handcrafted_feature_fields),
        "efficientnet_features": len(efficientnet_feature_fields),
        "final_features": len(fieldnames) - 1,
        "missing_values": missing_values,
    }
