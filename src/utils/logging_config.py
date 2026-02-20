"""Centralized logging configuration for alphaBT."""
import logging
import os

LOGGER_NAME = 'alphaBT'
DEFAULT_FORMAT = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def setup_logging(
    log_file: str = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """Configure the alphaBT root logger.

    Parameters:
        log_file: Path to log file. If provided, adds FileHandler.
        console_level: Level for console output (INFO for backtest, WARNING for tuning).
        file_level: Level for file output (captures everything by default).

    Returns:
        The configured root logger.
    """
    root_logger = logging.getLogger(LOGGER_NAME)
    root_logger.setLevel(logging.DEBUG)  # Allow all levels; handlers filter

    # Avoid duplicate handlers on re-init
    root_logger.handlers.clear()

    formatter = logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    # File handler (if path provided)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(file_level)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the alphaBT namespace.

    Usage:
        from utils.logging_config import get_logger
        logger = get_logger(__name__)
    """
    return logging.getLogger(f'{LOGGER_NAME}.{name}')
