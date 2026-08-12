"""
Edge device entry point running lightweight detection alongside the GoPro.

This is the main entry point for the Edge Application with state machine control.
It uses the refactored modular architecture with proper state management.
"""

import logging
import sys
import argparse
from pathlib import Path

# Add module paths
MODULE_DIR = Path(__file__).resolve().parents[1] / "modules"
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
sys.path.append(str(MODULE_DIR))
sys.path.append(str(SRC_DIR))

from edge_application import EdgeApplicationStateMachine
from EdgeStateMachine import ApplicationState
from EdgeApplicationConfig import EdgeApplicationConfig


logger = logging.getLogger(__name__)


def main():
    """
    Main entry point for the Edge Application.

    This initializes and runs the Edge Application with the state machine
    architecture as specified in the design document.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='BearVision Edge Application - Automatic wakeboard clip generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Use the versioned default config
  python edge_main.py

  # Use specific YAML config
  python edge_main.py --config config/edge.yaml
        '''
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to a versioned edge YAML file. Defaults to config/edge.yaml'
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("=" * 70)
    logger.info("Starting Edge Application with State Machine")
    logger.info("=" * 70)

    # Determine config file path
    repo_root = Path(__file__).resolve().parents[2]

    if args.config:
        # User specified config file
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = repo_root / config_path
    else:
        config_path = repo_root / "config" / "edge.yaml"
        logger.info("Using default YAML config: config/edge.yaml")

    # Load configuration
    config = EdgeApplicationConfig()

    if config.load_from_file(str(config_path)):
        logger.info(f"Configuration loaded from {config_path}")
        config.print_config()
    else:
        logger.warning("Using default configuration values")

    def status_callback(state: ApplicationState, message: str):
        """Status callback for logging state transitions and status updates."""
        logger.info(f"[{state.value.upper()}] {message}")

    # Create and run the Edge Application with state machine and config
    app = EdgeApplicationStateMachine(
        status_callback=status_callback,
        config=config
    )

    try:
        app.run()
    except KeyboardInterrupt:
        logger.info("\nShutdown requested by user...")
    except Exception as e:
        logger.error(f"Fatal error in Edge Application: {e}", exc_info=True)
    finally:
        app.shutdown()
        logger.info("=" * 70)
        logger.info("Edge Application terminated")
        logger.info("=" * 70)


if __name__ == "__main__":
    main()
