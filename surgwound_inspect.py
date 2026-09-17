import base64
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DATASET_DIR = Path(__file__).resolve().parent / "data" / "SurgWound"
SPLITS = {
    "train": DATASET_DIR / "train_question.json",
    "validation": DATASET_DIR / "val_question.json",
    "test": DATASET_DIR / "test_question.json",
}
TARGET_FIELDS = {"Healing Status", "Infection Risk Assessment"}


def _load_records(path: Path) -> tuple[Any, list[dict[str, Any]]]:
    text = path.read_text(encoding="utf-8")
    if not text.lstrip().startswith("["):
        text = "[\n    {\n" + text
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload, [record for record in payload if isinstance(record, dict)]
    if isinstance(payload, dict):
        records = payload.get("data", payload.get("records", []))
        return payload, [record for record in records if isinstance(record, dict)]
    return payload, []


def _image_representation(record: dict[str, Any]) -> str:
    image = record.get("image")
    if isinstance(image, str):
        try:
            decoded = base64.b64decode(image, validate=True)
            if decoded:
                return f"embedded base64 image data ({len(decoded)} decoded bytes)"
        except ValueError:
            pass
        return "string image value"
    if image is None:
        return "no image field"
    return f"embedded {type(image).__name__} image value"


def _record_fields(records: list[dict[str, Any]]) -> list[str]:
    return sorted({field for record in records for field in record})


def inspect_dataset() -> dict[str, Any]:
    report: dict[str, Any] = {
        "dataset_directory": str(DATASET_DIR),
        "split_counts": {},
        "top_level_types": {},
        "top_level_fields": {},
        "record_fields": {},
        "image_representations": {},
        "image_names_by_split": {},
        "all_image_names": set(),
        "field_values": {},
        "task_types": set(),
        "answer_types": set(),
        "answer_values_by_field": defaultdict(set),
        "target_records": defaultdict(list),
        "record_ids_by_image": defaultdict(list),
    }
    for split, path in SPLITS.items():
        payload, records = _load_records(path)
        report["split_counts"][split] = len(records)
        report["top_level_types"][split] = type(payload).__name__
        report["top_level_fields"][split] = sorted(payload) if isinstance(payload, dict) else []
        report["record_fields"][split] = _record_fields(records)
        image_names: set[str] = set()
        representations: Counter[str] = Counter()
        for record in records:
            image_name = str(record.get("image_name", "")).strip()
            if image_name:
                image_names.add(image_name)
                report["all_image_names"].add(image_name)
                report["record_ids_by_image"][image_name].append(record.get("id", ""))
            representations[_image_representation(record)] += 1
            if "task_type" in record:
                report["task_types"].add(str(record["task_type"]))
            answer = record.get("answer")
            report["answer_types"].add(type(answer).__name__)
            field = str(record.get("field", ""))
            report["answer_values_by_field"][field].add(str(answer))
            if field in TARGET_FIELDS:
                report["target_records"][field].append(
                    {"image_name": image_name, "answer": answer, "id": record.get("id", "")}
                )
        report["image_names_by_split"][split] = image_names
        report["image_representations"][split] = dict(representations)
    return report


def _json_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def print_report(report: dict[str, Any]) -> None:
    print(f"Dataset directory: {report['dataset_directory']}")
    print(f"Records by split: {report['split_counts']}")
    print(f"Top-level JSON types: {report['top_level_types']}")
    print(f"Top-level fields: {report['top_level_fields']}")
    print(f"Record fields by split: {report['record_fields']}")
    print(f"Image representations by split: {report['image_representations']}")
    print(f"Total unique wound images: {len(report['all_image_names'])}")
    print(f"Image filename/ID examples: {sorted(report['all_image_names'])[:20]}")
    print(f"Task types: {sorted(report['task_types'])}")
    print(f"Answer value types: {sorted(report['answer_types'])}")
    print("Unique answer values by field:")
    for field in sorted(report["answer_values_by_field"]):
        print(f"  {field}: {sorted(report['answer_values_by_field'][field])}")
    for field in sorted(TARGET_FIELDS):
        records = report["target_records"].get(field, [])
        print(f"{field} record count: {len(records)}")
        print(f"{field} value distribution: {dict(sorted(Counter(str(record['answer']) for record in records).items()))}")
    print("Same-image record references:")
    duplicate_images = {
        image: ids for image, ids in report["record_ids_by_image"].items() if len(ids) > 1
    }
    print(f"  images_with_multiple_records: {len(duplicate_images)}")
    print(f"  example: {dict(list(duplicate_images.items())[:5])}")
    split_sets = report["image_names_by_split"]
    print(f"Image overlap train/validation: {len(split_sets['train'] & split_sets['validation'])}")
    print(f"Image overlap train/test: {len(split_sets['train'] & split_sets['test'])}")
    print(f"Image overlap validation/test: {len(split_sets['validation'] & split_sets['test'])}")
    for field in sorted(TARGET_FIELDS):
        values_by_image: dict[str, set[str]] = defaultdict(set)
        for record in report["target_records"].get(field, []):
            values_by_image[record["image_name"]].add(str(record["answer"]))
        print(f"{field} image mapping examples: {dict(list(values_by_image.items())[:5])}")
    healing = {record["image_name"] for record in report["target_records"].get("Healing Status", [])}
    infection = {record["image_name"] for record in report["target_records"].get("Infection Risk Assessment", [])}
    print(f"Images with both target labels: {len(healing & infection)}")
    print(f"Missing healing labels among dataset images: {len(report['all_image_names'] - healing)}")
    print(f"Missing infection labels among dataset images: {len(report['all_image_names'] - infection)}")
    print("Image extraction: decode each record's embedded base64 `image` field; `image_name` supplies the filename.")
    print(f"Mapping image_filename -> healing_status -> infection_risk available: {healing <= report['all_image_names'] and infection <= report['all_image_names'] and bool(healing & infection)}")


if __name__ == "__main__":
    print_report(inspect_dataset())
