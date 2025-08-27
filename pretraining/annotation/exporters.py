"""Export classes for annotation pipeline - DatasetExporter and CvatExporter."""

import json
from pathlib import Path
from typing import List, Dict, Any
import cv2
import logging

# Import status tracking
from status import track

logger = logging.getLogger(__name__)


class DatasetExporter:
    """Save frames, labels and debug information."""

    def __init__(self, config):
        """Prepare output directories and debug log.

        Purpose
        -------
        Ensure required folders exist before any frames are processed, avoiding
        repetitive checks inside the hot save loop.

        Inputs
        ------
        config: ExportConfig
            Carries the output directory path.

        Outputs
        -------
        None
            Directories are created and a debug file handle is opened.
        """
        self.root = Path(config.output_dir)
        self.img_dir = self.root / "images"
        self.lbl_dir = self.root / "labels"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.lbl_dir.mkdir(parents=True, exist_ok=True)
        self.debug_path = self.root / "debug.jsonl"
        self.debug_file = self.debug_path.open("w", encoding="utf-8")

    @track
    def save(self, item: Dict[str, Any], boxes: List[Dict[str, Any]]):
        """Write one frame and its annotations to disk.

        Purpose
        -------
        Persist data in a YOLO-friendly layout while also logging rich
        information for potential debugging or later analysis.

        Inputs
        ------
        item: dict
            Metadata and pixel data for the frame.
        boxes: list[dict]
            Bounding boxes associated with the frame.

        Outputs
        -------
        None
            Files are written to the export directory and debug log.
        """
        frame = item["frame"]
        idx = item["frame_idx"]
        video_name = Path(item["video"]).stem
        img_name = f"{video_name}_{idx:06d}.jpg"
        lbl_name = f"{video_name}_{idx:06d}.txt"
        logger.debug("Exporting frame %d as %s with %d labels", idx, img_name, len(boxes))
        cv2.imwrite(str(self.img_dir / img_name), frame)
        with open(self.lbl_dir / lbl_name, "w", encoding="utf-8") as f:
            for b in boxes:
                x1, y1, x2, y2 = b["bbox"]
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w / 2
                cy = y1 + h / 2
                f.write(f"{b['cls']} {cx} {cy} {w} {h}\n")
        debug = {
            "image": img_name,
            "labels": boxes,
            "video": item["video"],
            "frame_idx": idx,
        }
        if item.get("discarded_boxes"):
            # Including discarded detections in the log aids post-run auditing
            # without cluttering the exported dataset.
            debug["discarded_boxes"] = item["discarded_boxes"]
        # JSON Lines format allows streaming large debug logs without
        # constructing a massive in-memory structure.
        self.debug_file.write(json.dumps(debug) + "\n")

    def close(self):
        """Flush and close the debug log."""
        self.debug_file.close()
        

class CvatExporter:
    """Save frames and annotations in CVAT XML format."""

    def __init__(self, config):
        """Create an image directory and buffer annotations.

        Purpose
        -------
        CVAT expects a single XML with all boxes, so we accumulate annotations
        in memory and defer writing until :meth:`close`.

        Inputs
        ------
        config: ExportConfig
            Specifies the output directory.

        Outputs
        -------
        None
            Directories are created and an in-memory list is initialised.
        """
        self.root = Path(config.output_dir)
        self.img_dir = self.root / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.annotations: List[Dict[str, Any]] = []

    @track
    def save(self, item: Dict[str, Any], boxes: List[Dict[str, Any]]):
        """Append annotation metadata for one frame.

        Purpose
        -------
        Images are written immediately to avoid large memory use, while box
        metadata is stored for later XML serialization.

        Inputs
        ------
        item: dict
            Frame metadata including pixel data and identifiers.
        boxes: list[dict]
            Bounding boxes detected in the frame.

        Outputs
        -------
        None
            Updates the internal ``annotations`` list and saves an image.
        """
        frame = item["frame"]
        idx = item["frame_idx"]
        video_name = Path(item["video"]).stem
        img_name = f"{video_name}_{idx:06d}.jpg"
        cv2.imwrite(str(self.img_dir / img_name), frame)
        h, w = frame.shape[:2]
        self.annotations.append({
            "name": img_name,
            "width": w,
            "height": h,
            "boxes": boxes,
        })

    def close(self):
        """Write accumulated annotations to ``annotations.xml``."""
        import xml.etree.ElementTree as ET

        root_el = ET.Element("annotations")
        for i, img in enumerate(self.annotations):
            img_el = ET.SubElement(
                root_el,
                "image",
                id=str(i),
                name=img["name"],
                width=str(img["width"]),
                height=str(img["height"]),
            )
            for b in img["boxes"]:
                x1, y1, x2, y2 = b["bbox"]
                box_el = ET.SubElement(
                    img_el,
                    "box",
                    label=str(b.get("label", b.get("cls", "0"))),
                    xtl=str(x1),
                    ytl=str(y1),
                    xbr=str(x2),
                    ybr=str(y2),
                    occluded="0",
                )
                ET.SubElement(box_el, "attribute", name="confidence").text = str(
                    b.get("conf", 0)
                )

        tree = ET.ElementTree(root_el)
        tree.write(self.root / "annotations.xml", encoding="utf-8", xml_declaration=True)