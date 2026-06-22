"""
Logging utility for ASV pi-control.
Provides consistent logging across all modules.
"""

import logging
import os
from src.config import LOG_LEVEL, LOG_FILE, LOG_TO_FILE, LOG_TO_CONSOLE


def setup_logger(name: str = "asv") -> logging.Logger:
    """Create and configure a logger instance.
    
    Args:
        name: Logger name (usually the module name).
    
    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    # Avoid duplicate handlers on re-init
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s %(name)-12s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if LOG_TO_CONSOLE:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if LOG_TO_FILE:
        log_dir = os.path.dirname(LOG_FILE)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
