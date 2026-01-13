"""Unit tests for logging module."""

import logging
import tempfile
from pathlib import Path

import pytest

from src.utils.logging import (
    get_logger,
    setup_experiment_logging,
    setup_logging,
)


@pytest.fixture
def reset_logging():
    """Reset logging configuration before each test."""
    # Reset root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.WARNING)

    # Reset module flag
    import src.utils.logging

    src.utils.logging._logging_configured = False

    yield

    # Cleanup after test
    root_logger.handlers.clear()
    src.utils.logging._logging_configured = False


def test_setup_logging_console_handler(reset_logging):
    """Test that console handler is added to root logger."""
    setup_logging(log_level="INFO")

    root_logger = logging.getLogger()
    handlers = root_logger.handlers

    assert len(handlers) >= 1
    assert any(isinstance(h, logging.StreamHandler) for h in handlers)


def test_setup_logging_file_handler(reset_logging):
    """Test that file handler is created when log_file provided."""
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test.log"
        setup_logging(log_level="INFO", log_file=str(log_file))

        root_logger = logging.getLogger()
        handlers = root_logger.handlers

        assert len(handlers) >= 2
        assert any(isinstance(h, logging.FileHandler) for h in handlers)
        assert log_file.exists()


def test_setup_logging_creates_directory(reset_logging):
    """Test that log file directory is created if it doesn't exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "nested" / "dir" / "test.log"
        setup_logging(log_level="INFO", log_file=str(log_file))

        assert log_file.parent.exists()
        assert log_file.exists()


def test_get_logger_returns_logger(reset_logging):
    """Test that get_logger returns a logger with correct name."""
    setup_logging()
    logger = get_logger("test_module")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"


def test_logger_format(reset_logging, capsys):
    """Test that log message format matches expected pattern."""
    setup_logging(log_level="INFO")
    logger = get_logger("test_module")
    logger.info("Test message")

    captured = capsys.readouterr()
    # Logging output goes to stderr by default
    output = captured.err + captured.out

    # Check format: [YYYY-MM-DD HH:MM:SS] [LEVEL] [MODULE] Message
    assert "[INFO]" in output
    assert "[test_module]" in output
    assert "Test message" in output


def test_setup_experiment_logging(reset_logging):
    """Test experiment logger creates log file in correct directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        experiment_dir = Path(temp_dir) / "experiment_001"
        logger = setup_experiment_logging(str(experiment_dir))

        assert isinstance(logger, logging.Logger)
        assert logger.name == "experiment"

        log_file = experiment_dir / "logs" / "training.log"
        assert log_file.exists()


def test_log_levels(reset_logging, capsys):
    """Test different log levels work correctly."""
    setup_logging(log_level="DEBUG")
    logger = get_logger("test")

    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    captured = capsys.readouterr()
    # Logging output goes to stderr by default
    output = captured.err + captured.out

    assert "Debug message" in output
    assert "Info message" in output
    assert "Warning message" in output
    assert "Error message" in output


def test_setup_logging_idempotent(reset_logging):
    """Test that setup_logging can be called multiple times without issues."""
    setup_logging(log_level="INFO")
    handler_count_1 = len(logging.getLogger().handlers)

    # Reset flag to allow reconfiguration
    import src.utils.logging

    src.utils.logging._logging_configured = False

    setup_logging(log_level="DEBUG")
    handler_count_2 = len(logging.getLogger().handlers)

    # Should have same number of handlers (old ones cleared)
    assert handler_count_1 == handler_count_2


def test_logger_inherits_root_config(reset_logging, capsys):
    """Test that module loggers inherit root logger configuration."""
    setup_logging(log_level="INFO")
    logger1 = get_logger("module1")
    logger2 = get_logger("module2")

    logger1.info("Message from module1")
    logger2.info("Message from module2")

    captured = capsys.readouterr()
    # Logging output goes to stderr by default
    output = captured.err + captured.out

    assert "Message from module1" in output
    assert "Message from module2" in output
    assert "[module1]" in output
    assert "[module2]" in output


def test_file_handler_encoding(reset_logging):
    """Test that file handler uses UTF-8 encoding."""
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test.log"
        setup_logging(log_file=str(log_file))

        logger = get_logger("test")
        logger.info("Test message with émojis 🎉")

        # Read file and check encoding
        content = log_file.read_text(encoding="utf-8")
        assert "Test message with émojis 🎉" in content
