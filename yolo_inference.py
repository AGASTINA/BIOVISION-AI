from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO


ImageArray = NDArray[np.uint8]


@dataclass(frozen=True)
class WoundSegmentationResult:
    image: ImageArray
    mask: ImageArray
    confidence: float | None


@lru_cache(maxsize=1)
def _load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def _save_visualization(
    image: ImageArray,
    mask: ImageArray,
    output_path: Path,
) -> None:
    visualization = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mask_pixels = mask.astype(bool)

    overlay = visualization.copy()
    overlay[mask_pixels] = (0, 0, 255)
    visualization = cv2.addWeighted(visualization, 0.7, overlay, 0.3, 0)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(visualization, contours, -1, (0, 255, 255), 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), visualization):
        raise OSError(f"Could not save visualization to {output_path}")


def run_inference(
    image_path: str | Path,
    output_path: str | Path = "outputs/yolo_inference.jpg",
    model_path: str | Path | None = None,
    mask_threshold: float = 0.5,
) -> WoundSegmentationResult:
    project_root = Path(__file__).resolve().parent
    resolved_model_path = Path(model_path) if model_path else project_root / "models" / "best.pt"
    resolved_model_path = resolved_model_path.resolve()
    if not resolved_model_path.is_file():
        raise FileNotFoundError(f"YOLO model not found: {resolved_model_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = _load_model(str(resolved_model_path))(image, verbose=False)[0]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    confidence: float | None = None

    if result.masks is not None and len(result.masks.data) > 0:
        detected_masks = result.masks.data.detach().cpu().numpy()
        mask = np.any(detected_masks > mask_threshold, axis=0).astype(np.uint8)
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask *= 255

    if result.boxes is not None and len(result.boxes.conf) > 0:
        confidence = float(result.boxes.conf.detach().cpu().max().item())

    _save_visualization(image_rgb, mask, Path(output_path).resolve())
    return WoundSegmentationResult(image_rgb, mask, confidence)
