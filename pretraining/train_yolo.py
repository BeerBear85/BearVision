import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Tuple
import random
import yaml

from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def prepare_dataset(source_dir: Path, val_ratio: float) -> Tuple[Path, Path]:
    """Split a flat folder of images/labels into YOLO train/val structure.

    Parameters
    ----------
    source_dir: Path
        Directory containing images and matching YOLO-format ``.txt`` files.
    val_ratio: float
        Fraction of items to place in the validation split.

    Returns
    -------
    data_root: Path
        Root directory containing the ``train`` and ``val`` folders.
    yaml_path: Path
        Path to the generated ``data.yaml`` file.
    """
    images = [p for p in source_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        raise FileNotFoundError(f"No images found in {source_dir}")
    random.shuffle(images)
    split_idx = int(len(images) * (1 - val_ratio))
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    data_root = Path(tempfile.mkdtemp(prefix="yolo_data_"))
    for split, items in {"train": train_imgs, "val": val_imgs}.items():
        img_dir = data_root / split / "images"
        lbl_dir = data_root / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img in items:
            label = img.with_suffix(".txt")
            if not label.exists():
                continue
            shutil.copy2(img, img_dir / img.name)
            shutil.copy2(label, lbl_dir / label.name)

    yaml_path = data_root / "data.yaml"
    data_cfg = {
        "path": str(data_root),
        "train": "train/images",
        "val": "val/images",
        "names": ["wakeboard"],
    }
    with yaml_path.open("w") as f:
        yaml.safe_dump(data_cfg, f)
    return data_root, yaml_path


def train(data_dir: Path, model: str, epochs: int, batch: int, imgsz: int,
          device: str, val_ratio: float, onnx_out: Path) -> None:
    """Fine-tune a YOLO model and export to ONNX."""
    _, data_yaml = prepare_dataset(data_dir, val_ratio)
    yolo_model = YOLO(model)
    yolo_model.train(data=str(data_yaml), epochs=epochs, batch=batch, imgsz=imgsz, device=device)
    best_pt = Path(yolo_model.trainer.save_dir) / "weights" / "best.pt"
    trained = YOLO(str(best_pt))
    trained.export(format="onnx", imgsz=imgsz, path=str(onnx_out))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on a custom dataset")
    parser.add_argument("data_dir", type=Path, help="Directory with images and YOLO .txt labels")
    parser.add_argument("--model", default="yolov8x.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--device", default=None, help="Device to use (e.g. 0 or 'cpu')")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--onnx-out", type=Path, default=Path("yolov8_finetuned.onnx"),
                        help="Output path for ONNX model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data_dir, args.model, args.epochs, args.batch, args.imgsz,
          args.device, args.val_ratio, args.onnx_out)
