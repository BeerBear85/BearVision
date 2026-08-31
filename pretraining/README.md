# Offline YOLO workspace

This directory is for offline dataset annotation and YOLO fine-tuning only.
Nothing here is part of the Edge or server runtime, and runtime modules must not
import it.

## Environment

```bash
uv sync --locked --extra training --extra gui
```

## Annotation

```bash
uv run python pretraining/annotation/annotation_gui_pyqt.py
```

## Fine-tuning

```bash
uv run python pretraining/run_training_gui.py
```

The resulting model must be reviewed and deliberately configured for the
runtime. Training must never happen on an Edge or server process.

Run the offline workflow tests explicitly:

```bash
uv run ruff check pretraining tests/offline_yolo
uv run pytest tests/offline_yolo
```
