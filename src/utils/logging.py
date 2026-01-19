"""Centralized logging configuration module.

This module provides functions to configure Python's standard logging module
with consistent formatting, file/console handlers, and module-specific loggers.
"""

import logging
from pathlib import Path
from typing import Optional


# Track if logging has been configured to avoid reconfiguration
_logging_configured = False


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure root logger with consistent formatting and handlers.

    This function sets up the root logger with:
    - A console handler (stdout) with the specified log level
    - An optional file handler if log_file is provided
    - Consistent formatting: [YYYY-MM-DD HH:MM:SS] [LEVEL] [MODULE] Message

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            Defaults to "INFO".
        log_file: Optional path to log file. If provided, creates a file handler
            and ensures the directory exists. Defaults to None.

    Example:
        >>> setup_logging(log_level="DEBUG", log_file="logs/app.log")
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
    global _logging_configured

    # Avoid reconfiguring if already set up
    if _logging_configured:
        return

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (always added)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        # Ensure directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """Get module-specific logger.

    This function returns a logger instance for the specified module name.
    The logger inherits configuration from the root logger set up by
    `setup_logging()`.

    Args:
        name: Logger name, typically `__name__` of the calling module.

    Returns:
        Logger instance configured with the root logger settings.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing data")
        >>> logger.debug("Detailed debug information")
    """
    return logging.getLogger(name)


def setup_experiment_logging(experiment_dir: str) -> logging.Logger:
    """Create experiment-specific logger with log file.

    This function sets up logging for an experiment run, creating a log file
    at `{experiment_dir}/logs/training.log`.

    Args:
        experiment_dir: Path to experiment directory.

    Returns:
        Logger instance for the experiment.

    Example:
        >>> logger = setup_experiment_logging("experiments/runs/exp_001")
        >>> logger.info("Starting experiment")
    """
    experiment_path = Path(experiment_dir)
    log_file = experiment_path / "logs" / "training.log"

    # Ensure experiment directory exists
    experiment_path.mkdir(parents=True, exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger()
    
    # Check if logging was already configured
    global _logging_configured
    if _logging_configured:
        # Logging already configured, just add file handler
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file handler already exists for this file
        file_handler_exists = any(
            isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path.absolute())
            for h in root_logger.handlers
        )
        
        if not file_handler_exists:
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
    else:
        # First time setup, configure logging with file handler
        setup_logging(log_level="INFO", log_file=str(log_file))

    # Return experiment-specific logger
    return get_logger("experiment")
