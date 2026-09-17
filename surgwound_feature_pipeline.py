import argparse
import csv
import gc
import os
import tempfile
import time
from collections import Counter
from pathlib import Path

import cv2

from efficientnet_features import extract_efficientnet_features
from feature_fusion import fuse_feature_csvs
from handcrafted_features import FEATURE_NAMES, extract_handcrafted_features
from peri_wound import extract_peri_wound
from yolo_inference import run_inference


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = PROJECT_ROOT / "data" / "SurgWound" / "images"
LABELS_PATH = PROJECT_ROOT / "data" / "SurgWound" / "clinical_labels.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "SurgWound"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _validate_inputs(image_paths: list[Path], label_rows: list[dict[str, str]]) -> None:
    image_names = {path.name for path in image_paths}
    label_names = [row.get("filename", "") for row in label_rows]
    if len(image_paths) == 686 and len(label_rows) != 686:
        raise ValueError(f"Expected 686 clinical labels, found {len(label_rows)}")
    if len(label_names) != len(set(label_names)):
        raise ValueError("Clinical labels contain duplicate filenames")
    if not image_names <= set(label_names):
        raise ValueError(
            f"Images without labels: {sorted(image_names - set(label_names))}"
        )


def _open_csv_writer(path: Path, fieldnames: list[str]):
    file_handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
    writer.writeheader()
    file_handle.flush()
    return file_handle, writer


def _validate_feature_csvs(
    handcrafted_path: Path,
    efficientnet_path: Path,
    fused_path: Path,
    expected_names: set[str],
) -> dict[str, object]:
    handcrafted_rows = _read_rows(handcrafted_path)
    efficientnet_rows = _read_rows(efficientnet_path)
    fused_rows = _read_rows(fused_path)
    fused_names = [row["filename"] for row in fused_rows]
    missing_values = sum(value == "" for row in fused_rows for value in row.values())
    duplicate_names = sorted(
        name for name, count in Counter(fused_names).items() if count > 1
    )
    fused_name_set = set(fused_names)
    return {
        "handcrafted_rows": len(handcrafted_rows),
        "efficientnet_rows": len(efficientnet_rows),
        "fused_rows": len(fused_rows),
        "unique_filenames": len(fused_name_set),
        "duplicate_filenames": duplicate_names,
        "missing_values": missing_values,
        "feature_columns": len(fused_rows[0]) - 1 if fused_rows else 0,
        "missing_feature_names": sorted(expected_names - fused_name_set),
        "unexpected_feature_names": sorted(fused_name_set - expected_names),
    }


def _promote_outputs(staging_root: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for name in (
        "handcrafted_features.csv",
        "efficientnet_features.csv",
        "fused_features.csv",
    ):
        os.replace(staging_root / name, output_root / name)
    staging_root.rmdir()


def run_pipeline(
    max_images: int | None = None,
    batch_size: int = 8,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, object]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    started = time.perf_counter()
    all_image_paths = sorted(IMAGE_ROOT.glob("*.jpg"))
    label_rows = _read_rows(LABELS_PATH)
    image_paths = all_image_paths[:max_images] if max_images is not None else all_image_paths
    _validate_inputs(image_paths, label_rows)
    expected_names = {path.name for path in image_paths}
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix="SurgWound_", dir=output_root.parent))
    errors: list[str] = []
    detection_count = 0
    peri_wound_count = 0
    processed_count = 0
    handcrafted_handle = None
    efficientnet_handle = None

    try:
        for batch_start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[batch_start : batch_start + batch_size]
            pipeline_items = []
            batch_handcrafted_rows = []

            for image_path in batch_paths:
                try:
                    visualization = tempfile.NamedTemporaryFile(
                        suffix=".jpg",
                        delete=False,
                    )
                    visualization_path = Path(visualization.name)
                    visualization.close()
                    try:
                        segmentation = run_inference(
                            image_path,
                            output_path=visualization_path,
                            model_path=MODEL_PATH,
                        )
                    finally:
                        visualization_path.unlink(missing_ok=True)
                    if cv2.countNonZero(segmentation.mask) > 0:
                        detection_count += 1
                except Exception as error:
                    errors.append(f"{image_path.name} [YOLO]: {error}")
                    print(errors[-1], flush=True)
                    continue

                try:
                    peri_wound = extract_peri_wound(segmentation)
                    if peri_wound.peri_wound_area > 0:
                        peri_wound_count += 1
                except Exception as error:
                    errors.append(f"{image_path.name} [peri-wound]: {error}")
                    print(errors[-1], flush=True)
                    continue

                try:
                    batch_handcrafted_rows.append(
                        extract_handcrafted_features(
                            segmentation,
                            peri_wound,
                            image_path.name,
                        )
                    )
                    pipeline_items.append((image_path.name, segmentation, peri_wound))
                except Exception as error:
                    errors.append(f"{image_path.name} [handcrafted]: {error}")
                    print(errors[-1], flush=True)

            if errors:
                continue

            try:
                batch_efficientnet_rows = extract_efficientnet_features(
                    pipeline_items,
                    batch_size=batch_size,
                )
            except Exception as error:
                for filename, _, _ in pipeline_items:
                    errors.append(f"{filename} [EfficientNet]: {error}")
                    print(errors[-1], flush=True)
                continue

            if handcrafted_handle is None:
                handcrafted_handle, handcrafted_writer = _open_csv_writer(
                    staging_root / "handcrafted_features.csv",
                    ["filename", *FEATURE_NAMES],
                )
                efficientnet_handle, efficientnet_writer = _open_csv_writer(
                    staging_root / "efficientnet_features.csv",
                    list(batch_efficientnet_rows[0]),
                )

            for row in batch_handcrafted_rows:
                handcrafted_writer.writerow(row)
            for row in batch_efficientnet_rows:
                efficientnet_writer.writerow(row)
            handcrafted_handle.flush()
            efficientnet_handle.flush()
            processed_count += len(batch_paths)
            print(f"Processed {processed_count}/{len(image_paths)}", flush=True)

            del pipeline_items, batch_handcrafted_rows, batch_efficientnet_rows
            gc.collect()

        if errors or processed_count != len(image_paths):
            return {
                "ready": False,
                "errors": errors,
                "processed_count": processed_count,
                "elapsed": time.perf_counter() - started,
            }

        handcrafted_handle.close()
        efficientnet_handle.close()
        handcrafted_handle = None
        efficientnet_handle = None
        fused_path, _ = fuse_feature_csvs(
            staging_root / "handcrafted_features.csv",
            staging_root / "efficientnet_features.csv",
            staging_root / "fused_features.csv",
        )
        stats = _validate_feature_csvs(
            staging_root / "handcrafted_features.csv",
            staging_root / "efficientnet_features.csv",
            fused_path,
            expected_names,
        )
        stats["ready"] = not any(
            stats[key]
            for key in (
                "duplicate_filenames",
                "missing_values",
                "missing_feature_names",
                "unexpected_feature_names",
            )
        )
        if stats["ready"]:
            _promote_outputs(staging_root, output_root)
        return {
            **stats,
            "detection_count": detection_count,
            "peri_wound_count": peri_wound_count,
            "errors": errors,
            "elapsed": time.perf_counter() - started,
        }
    finally:
        if handcrafted_handle is not None:
            handcrafted_handle.close()
        if efficientnet_handle is not None:
            efficientnet_handle.close()
        if staging_root.exists():
            for path in staging_root.iterdir():
                path.unlink()
            staging_root.rmdir()
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    result = run_pipeline(args.max_images, args.batch_size, args.output_root)
    print(f"pipeline_result: {result}", flush=True)


if __name__ == "__main__":
    main()
