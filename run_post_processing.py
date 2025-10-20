#!/usr/bin/env python3
"""Command-line interface for post-processing pipeline.

This script provides an easy-to-use CLI for running the virtual cameraman
post-processing pipeline on wakeboard videos.

Usage Examples
--------------
Basic usage with defaults:
    python run_post_processing.py input.mp4 output.json

Custom YOLO model and scaling:
    python run_post_processing.py input.mp4 output.json --yolo yolov8m.pt --scaling 1.8

Advanced settings:
    python run_post_processing.py input.mp4 output.json \\
        --yolo yolov8l.pt \\
        --scaling 2.0 \\
        --confidence 0.6 \\
        --cutoff 3.0 \\
        --frame-skip 2 \\
        --verbose
"""

import argparse
import logging
import sys
from pathlib import Path

# Add code/modules to Python path
MODULE_DIR = Path(__file__).resolve().parent / 'code' / 'modules'
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from PostProcessingConfig import PostProcessingConfig
from PostProcessingPipeline import PostProcessingPipeline


def setup_logging(verbose: bool = False):
    """Configure logging for the CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Post-processing pipeline for wakeboard video virtual cameraman',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  python run_post_processing.py input.mp4 output.json

  # Custom YOLO model and scaling factor
  python run_post_processing.py input.mp4 output.json --yolo yolov8m.pt --scaling 1.8

  # Process every other frame with verbose logging
  python run_post_processing.py input.mp4 output.json --frame-skip 2 --verbose

  # Advanced: custom smoothing and confidence
  python run_post_processing.py input.mp4 output.json \\
      --yolo yolov8l.pt \\
      --scaling 2.0 \\
      --confidence 0.6 \\
      --cutoff 3.0 \\
      --device cuda

YOLO Model Options:
  yolov8n.pt - Nano (fastest, least accurate)
  yolov8s.pt - Small (balanced)
  yolov8m.pt - Medium (recommended)
  yolov8l.pt - Large (slow, most accurate)

Output:
  Creates a JSON file with:
  - Per-frame bounding boxes (clamped to frame boundaries)
  - Fixed box size for entire clip
  - Smoothed trajectory points
  - Original YOLO detections
  - Video metadata and configuration
        """
    )

    # Required arguments
    parser.add_argument(
        'input_video',
        type=str,
        help='Path to input video file'
    )
    parser.add_argument(
        'output_json',
        type=str,
        help='Path to output JSON metadata file'
    )

    # YOLO detection settings
    yolo_group = parser.add_argument_group('YOLO Detection Settings')
    yolo_group.add_argument(
        '--yolo',
        type=str,
        default='yolov8n.pt',
        metavar='MODEL',
        help='YOLO model weights file (default: yolov8n.pt)'
    )
    yolo_group.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        metavar='THRESH',
        help='Detection confidence threshold 0.0-1.0 (default: 0.5)'
    )
    yolo_group.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device for YOLO inference (default: cpu)'
    )

    # Bounding box settings
    bbox_group = parser.add_argument_group('Bounding Box Settings')
    bbox_group.add_argument(
        '--scaling',
        type=float,
        default=1.5,
        metavar='FACTOR',
        help='Box size scaling factor (default: 1.5, range: 1.0-3.0)'
    )
    bbox_group.add_argument(
        '--aspect-ratio',
        type=float,
        default=None,
        metavar='RATIO',
        help='Target aspect ratio (width/height), e.g., 1.0 for square, 1.78 for 16:9'
    )
    bbox_group.add_argument(
        '--no-preserve-aspect',
        action='store_true',
        help='Do not preserve detected aspect ratio when scaling'
    )

    # Trajectory smoothing settings
    traj_group = parser.add_argument_group('Trajectory Smoothing Settings')
    traj_group.add_argument(
        '--cutoff',
        type=float,
        default=2.0,
        metavar='HZ',
        help='Low-pass filter cutoff frequency in Hz (default: 2.0, 0=disable)'
    )
    traj_group.add_argument(
        '--sample-rate',
        type=float,
        default=None,
        metavar='FPS',
        help='Sampling rate in FPS (default: video FPS / frame_skip)'
    )

    # Processing options
    proc_group = parser.add_argument_group('Processing Options')
    proc_group.add_argument(
        '--frame-skip',
        type=int,
        default=1,
        metavar='N',
        help='Process every Nth frame (default: 1, every frame)'
    )
    proc_group.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info('='*60)
    logger.info('Post-Processing Pipeline CLI')
    logger.info('='*60)

    try:
        # Create configuration
        config = PostProcessingConfig(
            input_video=args.input_video,
            output_json=args.output_json,
            yolo_model=args.yolo,
            confidence_threshold=args.confidence,
            scaling_factor=args.scaling,
            preserve_aspect_ratio=not args.no_preserve_aspect,
            target_aspect_ratio=args.aspect_ratio,
            cutoff_hz=args.cutoff,
            sample_rate=args.sample_rate,
            frame_skip=args.frame_skip,
            device=args.device,
            verbose=args.verbose
        )

        # Validate configuration
        config.validate()

        # Run pipeline
        pipeline = PostProcessingPipeline(config)
        result = pipeline.run()

        # Print summary
        logger.info('')
        logger.info('='*60)
        logger.info('SUCCESS!')
        logger.info('='*60)
        logger.info(f"Input video:       {args.input_video}")
        logger.info(f"Output metadata:   {result['output_json']}")
        logger.info(f"Total frames:      {result['total_frames']}")
        logger.info(f"Detections found:  {result['num_detections']}")
        logger.info(f"Trajectory length: {result['trajectory_length']}")
        logger.info(f"Fixed box size:    {result['fixed_box_size']['width']:.1f} x "
                   f"{result['fixed_box_size']['height']:.1f} px")
        logger.info('='*60)

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
