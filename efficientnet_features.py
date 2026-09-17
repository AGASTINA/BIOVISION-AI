import csv
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3

from peri_wound import PeriWoundResult
from yolo_inference import WoundSegmentationResult


PipelineInput = tuple[str, WoundSegmentationResult, PeriWoundResult]


@lru_cache(maxsize=1)
def _load_feature_extractor() -> tuple[nn.Module, object]:
    weights = EfficientNet_B3_Weights.DEFAULT
    model = efficientnet_b3(weights=weights)
    model.classifier = nn.Identity()
    model.eval()
    return model, weights.transforms()


def _masked_image(
    segmentation: WoundSegmentationResult,
    peri_wound: PeriWoundResult,
) -> Image.Image:
    region = (peri_wound.wound_mask > 0) | (peri_wound.peri_wound_mask > 0)
    image = np.where(region[..., None], segmentation.image, 0)
    return Image.fromarray(image.astype(np.uint8), mode="RGB")


def extract_efficientnet_features(
    items: Sequence[PipelineInput],
    batch_size: int = 8,
    device: str | torch.device = "cpu",
) -> list[dict[str, str | float]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    model, transform = _load_feature_extractor()
    model = model.to(device)
    rows: list[dict[str, str | float]] = []
    feature_offset = 0

    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            batch_items = items[start : start + batch_size]
            batch = torch.stack(
                [transform(_masked_image(segmentation, peri_wound)) for _, segmentation, peri_wound in batch_items]
            ).to(device)
            features: Tensor = model(batch).cpu()

            for item, vector in zip(batch_items, features):
                values = vector.tolist()
                if feature_offset == 0:
                    feature_offset = len(values)
                row = {"filename": item[0]}
                row.update({f"efficientnet_{index:04d}": float(value) for index, value in enumerate(values)})
                rows.append(row)

    return rows


def save_feature_vectors_csv(
    rows: Iterable[dict[str, str | float]],
    output_path: str | Path,
) -> Path:
    rows = list(rows)
    feature_names = [
        key for key in rows[0] if key != "filename"
    ] if rows else []
    fieldnames = ["filename", *feature_names]
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return destination
