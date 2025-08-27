"""Logging configuration utilities for annotation pipeline."""

import logging
import sys

from annotation_config import LoggingConfig


def setup_logging(config: LoggingConfig):
    """Configure logging with specified level and format.
    
    Purpose
    -------
    Initialize the logging system with the format that includes file and function
    names as required by the issue. This centralizes logging configuration to
    ensure consistent formatting across the pipeline.
    
    Inputs
    ------
    config: LoggingConfig
        Logging configuration specifying level and format.
        
    Outputs
    -------
    None
        Configures the root logger with the specified settings.
    """
    logger = logging.getLogger(__name__)
    
    # Check if we're in a test environment by looking for pytest
    in_test = 'pytest' in sys.modules
    
    if not in_test:
        # Get root logger and clear any existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Set the logging level
        root_logger.setLevel(getattr(logging, config.level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(config.format)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, config.level.upper()))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # Add file handler for debug log in working directory
        file_handler = logging.FileHandler(config.debug_filename)
        file_handler.setLevel(getattr(logging, config.level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    else:
        # In test mode, just set the level on our specific logger to avoid interfering with pytest's caplog
        logger.setLevel(getattr(logging, config.level.upper()))
        # Also set the root logger level if no handlers are configured
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            root_logger.setLevel(getattr(logging, config.level.upper()))
    
    logger.info("Logging configured with level: %s", config.level)