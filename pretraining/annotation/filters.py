"""Filtering utilities for annotation pipeline."""

from typing import List, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


def filter_small_person_boxes(
    boxes: List[Dict[str, Any]],
    image_shape: tuple[int, int],
    min_ratio: float,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split detections into kept and discarded based on diagonal ratio.

    Purpose
    -------
    Remove exceedingly small person detections that are unlikely to be
    meaningful while leaving other classes untouched.

    Inputs
    ------
    boxes: list[dict]
        Detections produced by the model. Only entries labelled ``person`` are
        considered for filtering.
    image_shape: tuple[int, int]
        Image height and width used to compute the full-image diagonal.
    min_ratio: float
        Minimum allowed ratio between the box diagonal and the image diagonal.

    Outputs
    -------
    tuple[list[dict], list[dict]]
        First list contains boxes that pass the filter, second list holds
        discarded boxes along with their computed ratio for logging.
    """

    kept: List[Dict[str, Any]] = []
    discarded: List[Dict[str, Any]] = []
    img_h, img_w = image_shape[:2]
    img_diag = float(np.hypot(img_w, img_h))
    for b in boxes:
        label = b.get("label", "").lower()
        if label != "person" and b.get("cls") != 0:
            # Only filter persons; other classes are kept verbatim.
            kept.append(b)
            continue
        x1, y1, x2, y2 = b["bbox"]
        bb_diag = float(np.hypot(x2 - x1, y2 - y1))
        ratio = bb_diag / img_diag
        if ratio < min_ratio:
            logger.debug(
                "Discarding person detection with diagonal ratio %.6f (threshold %.6f)",
                ratio,
                min_ratio,
            )
            discarded.append({"bbox": b["bbox"], "ratio": ratio})
        else:
            kept.append(b)
    return kept, discarded