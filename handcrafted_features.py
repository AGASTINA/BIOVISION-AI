import csv
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from peri_wound import PeriWoundResult
from yolo_inference import WoundSegmentationResult


FEATURE_NAMES = [
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "rgb_std_r",
    "rgb_std_g",
    "rgb_std_b",
    "hsv_mean_h",
    "hsv_mean_s",
    "hsv_mean_v",
    "hsv_std_h",
    "hsv_std_s",
    "hsv_std_v",
    "lab_mean_l",
    "lab_mean_a",
    "lab_mean_b",
    "lab_std_l",
    "lab_std_a",
    "lab_std_b",
    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_entropy",
    "wound_area",
    "wound_perimeter",
    "aspect_ratio",
    "circularity",
    "compactness",
]


def _masked_statistics(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pixels = image[mask > 0].astype(np.float64)
    if pixels.size == 0:
        pixels = image.reshape(-1, image.shape[-1]).astype(np.float64)
    return pixels.mean(axis=0), pixels.std(axis=0)


def _glcm_features(image: np.ndarray, mask: np.ndarray) -> tuple[float, ...]:
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    quantized = (grayscale.astype(np.uint16) * 32 // 256).astype(np.uint8)
    levels = 32
    glcm = np.zeros((levels, levels), dtype=np.float64)
    valid = mask > 0
    directions = ((0, 1), (-1, 1), (-1, 0), (-1, -1))

    for row_delta, column_delta in directions:
        source_rows = slice(max(0, -row_delta), min(image.shape[0], image.shape[0] - row_delta))
        source_columns = slice(max(0, -column_delta), min(image.shape[1], image.shape[1] - column_delta))
        target_rows = slice(max(0, row_delta), min(image.shape[0], image.shape[0] + row_delta))
        target_columns = slice(max(0, column_delta), min(image.shape[1], image.shape[1] + column_delta))
        pair_mask = valid[source_rows, source_columns] & valid[target_rows, target_columns]
        source_values = quantized[source_rows, source_columns][pair_mask]
        target_values = quantized[target_rows, target_columns][pair_mask]
        for source_value, target_value in zip(source_values, target_values):
            glcm[source_value, target_value] += 1
            glcm[target_value, source_value] += 1

    total = glcm.sum()
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0
    probabilities = glcm / total
    difference = np.abs(np.arange(levels)[:, None] - np.arange(levels)[None, :])
    contrast = float(np.sum((difference**2) * probabilities))
    homogeneity = float(np.sum(probabilities / (1 + difference)))
    energy = float(np.sqrt(np.sum(probabilities**2)))
    nonzero = probabilities > 0
    entropy = float(-np.sum(probabilities[nonzero] * np.log2(probabilities[nonzero])))
    return contrast, homogeneity, energy, entropy


def _shape_features(mask: np.ndarray) -> tuple[float, ...]:
    binary_mask = (mask > 0).astype(np.uint8)
    area = float(cv2.countNonZero(binary_mask))
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    if not contours or area == 0:
        return area, perimeter, 0.0, 0.0, 0.0

    x, y, width, height = cv2.boundingRect(np.vstack(contours))
    del x, y
    aspect_ratio = float(width / height) if height else 0.0
    circularity = float(4 * np.pi * area / (perimeter**2)) if perimeter else 0.0
    compactness = float(perimeter**2 / area)
    return area, perimeter, aspect_ratio, circularity, compactness


def extract_handcrafted_features(
    segmentation: WoundSegmentationResult,
    peri_wound: PeriWoundResult | None = None,
    image_name: str | None = None,
) -> dict[str, str | float]:
    if peri_wound is None:
        raise ValueError("peri_wound result is required for color and texture features")

    peri_wound_mask = peri_wound.peri_wound_mask
    rgb_mean, rgb_std = _masked_statistics(segmentation.image, peri_wound_mask)
    hsv_mean, hsv_std = _masked_statistics(
        cv2.cvtColor(segmentation.image, cv2.COLOR_RGB2HSV),
        peri_wound_mask,
    )
    lab_mean, lab_std = _masked_statistics(
        cv2.cvtColor(segmentation.image, cv2.COLOR_RGB2LAB),
        peri_wound_mask,
    )
    glcm = _glcm_features(segmentation.image, peri_wound_mask)
    shape = _shape_features(segmentation.mask)
    values = [
        *rgb_mean,
        *rgb_std,
        *hsv_mean,
        *hsv_std,
        *lab_mean,
        *lab_std,
        *glcm,
        *shape,
    ]
    row: dict[str, str | float] = dict(zip(FEATURE_NAMES, map(float, values)))
    if image_name is not None:
        row = {"filename": image_name, **row}
    return row


def save_features_csv(
    rows: Iterable[dict[str, str | float]],
    output_path: str | Path,
) -> Path:
    rows = list(rows)
    fieldnames = ["filename", *FEATURE_NAMES] if rows and "filename" in rows[0] else FEATURE_NAMES
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return destination
