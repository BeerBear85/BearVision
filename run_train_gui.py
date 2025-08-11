#!/usr/bin/env python3
"""Launcher script for the YOLO training GUI."""

import sys
from pathlib import Path

# Add pretraining directory to path so we can import the GUI
sys.path.insert(0, str(Path(__file__).parent / "pretraining"))

from train_yolo_gui import main

if __name__ == "__main__":
    main()