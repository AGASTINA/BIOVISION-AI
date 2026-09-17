import base64
import csv
from collections import Counter
from pathlib import Path
from typing import Any

from surgwound_inspect import SPLITS, _load_records


OUTPUT_COLUMNS = ("filename", "healing_outcome", "infection_status", "split")
HEALING_FIELD = "Healing Status"
INFECTION_FIELD = "Infection Risk Assessment"
VALID_HEALING = {"Healed", "Not Healed"}
VALID_INFECTION = {"Low", "Medium", "High"}


def _collect_images() -> dict[str, dict[str, Any]]:
    images: dict[str, dict[str, Any]] = {}
    for split, path in SPLITS.items():
        _, records = _load_records(path)
        for record in records:
            filename = str(record.get("image_name", "")).strip()
            if not filename or Path(filename).name != filename:
                raise ValueError(f"Invalid image filename: {filename!r}")
            entry = images.setdefault(
                filename,
                {
                    "split": split,
                    "image": None,
                    "healing": set(),
                    "infection": set(),
                },
            )
            if entry["split"] != split:
                raise ValueError(f"Image appears in multiple splits: {filename}")

            image_data = record.get("image")
            if entry["image"] is None:
                if not isinstance(image_data, str) or not image_data:
                    raise ValueError(f"Missing embedded image data: {filename}")
                entry["image"] = image_data

            field = record.get("field")
            answer = record.get("answer")
            if field == HEALING_FIELD:
                entry["healing"].add(str(answer))
            elif field == INFECTION_FIELD:
                entry["infection"].add(str(answer))

    return images


def _validate_images(images: dict[str, dict[str, Any]]) -> None:
    if len(images) != 686:
        raise ValueError(f"Expected 686 unique images, found {len(images)}")

    for filename, entry in images.items():
        if entry["image"] is None:
            raise ValueError(f"Missing image data: {filename}")
        if len(entry["healing"]) != 1 or not entry["healing"] <= VALID_HEALING:
            raise ValueError(f"Invalid or inconsistent healing label: {filename} -> {entry['healing']}")
        if len(entry["infection"]) != 1 or not entry["infection"] <= VALID_INFECTION:
            raise ValueError(f"Invalid or inconsistent infection label: {filename} -> {entry['infection']}")

    split_sets = {
        split: {filename for filename, entry in images.items() if entry["split"] == split}
        for split in SPLITS
    }
    for left, left_names in split_sets.items():
        for right, right_names in split_sets.items():
            if left < right and left_names & right_names:
                raise ValueError(f"Split overlap between {left} and {right}")


def prepare_dataset() -> None:
    images = _collect_images()
    _validate_images(images)
    output_root = Path(__file__).resolve().parent / "data" / "SurgWound"
    images_root = output_root / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for filename in sorted(images):
        entry = images[filename]
        image_bytes = base64.b64decode(entry["image"], validate=True)
        (images_root / filename).write_bytes(image_bytes)
        rows.append(
            {
                "filename": filename,
                "healing_outcome": next(iter(entry["healing"])),
                "infection_status": next(iter(entry["infection"])),
                "split": entry["split"],
            }
        )

    labels_path = output_root / "clinical_labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    written_rows = list(csv.DictReader(labels_path.open(newline="", encoding="utf-8")))
    filenames = [row["filename"] for row in written_rows]
    missing_images = [row["filename"] for row in written_rows if not (images_root / row["filename"]).is_file()]
    missing_healing = [row["filename"] for row in written_rows if not row["healing_outcome"]]
    missing_infection = [row["filename"] for row in written_rows if not row["infection_status"]]
    duplicate_filenames = sorted(name for name, count in Counter(filenames).items() if count > 1)
    split_sets = {
        split: {row["filename"] for row in written_rows if row["split"] == split}
        for split in SPLITS
    }
    overlap = sum(
        len(split_sets[left] & split_sets[right])
        for index, left in enumerate(SPLITS)
        for right in list(SPLITS)[index + 1 :]
    )
    if missing_images or missing_healing or missing_infection or duplicate_filenames or overlap:
        raise ValueError(
            "Written dataset validation failed: "
            f"missing_images={missing_images}, missing_healing={missing_healing}, "
            f"missing_infection={missing_infection}, duplicate_filenames={duplicate_filenames}, "
            f"split_overlap={overlap}"
        )

    print(f"total_extracted_images: {len(written_rows)}")
    print(f"train_images: {len(split_sets['train'])}")
    print(f"validation_images: {len(split_sets['validation'])}")
    print(f"test_images: {len(split_sets['test'])}")
    print(f"total_clinical_label_rows: {len(written_rows)}")
    print(f"healing_status_distribution: {dict(Counter(row['healing_outcome'] for row in written_rows))}")
    print(f"infection_risk_distribution: {dict(Counter(row['infection_status'] for row in written_rows))}")
    print(f"missing_images: {missing_images}")
    print(f"duplicate_filenames: {duplicate_filenames}")
    print(f"missing_healing_labels: {missing_healing}")
    print(f"missing_infection_labels: {missing_infection}")
    print(f"split_overlap: {overlap}")
    print(f"all_686_images_prepared: {len(written_rows) == 686}")


if __name__ == "__main__":
    prepare_dataset()
