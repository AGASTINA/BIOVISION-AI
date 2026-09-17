from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from yolo_inference import ImageArray, WoundSegmentationResult


@dataclass(frozen=True)
class PeriWoundResult:
    wound_mask: ImageArray
    dilated_mask: ImageArray
    peri_wound_mask: ImageArray
    peri_wound_area: int


def _save_visualization(
    image: ImageArray,
    wound_mask: ImageArray,
    peri_wound_mask: ImageArray,
    output_path: Path,
) -> None:
    visualization = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    overlay = visualization.copy()
    overlay[peri_wound_mask > 0] = (0, 220, 0)
    overlay[wound_mask > 0] = (0, 0, 255)
    visualization = cv2.addWeighted(visualization, 0.65, overlay, 0.35, 0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), visualization):
        raise OSError(f"Could not save visualization to {output_path}")


def extract_peri_wound(
    segmentation: WoundSegmentationResult,
    dilation_radius: int = 15,
    output_path: str | Path | None = None,
) -> PeriWoundResult:
    if dilation_radius < 1:
        raise ValueError("dilation_radius must be at least 1")

    wound_mask = (segmentation.mask > 0).astype(np.uint8) * 255
    kernel_size = 2 * dilation_radius + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    dilated_mask = cv2.dilate(wound_mask, kernel)
    peri_wound_mask = cv2.subtract(dilated_mask, wound_mask)
    peri_wound_area = int(cv2.countNonZero(peri_wound_mask))

    if output_path is not None:
        _save_visualization(
            segmentation.image,
            wound_mask,
            peri_wound_mask,
            Path(output_path).resolve(),
        )

    return PeriWoundResult(
        wound_mask=wound_mask,
        dilated_mask=dilated_mask,
        peri_wound_mask=peri_wound_mask,
        peri_wound_area=peri_wound_area,
    )
