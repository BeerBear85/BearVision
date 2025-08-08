#!/usr/bin/env python3
"""Simple launcher script for the YOLOv8 training GUI."""

import sys
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from train_yolo_gui import main

if __name__ == "__main__":
    main()