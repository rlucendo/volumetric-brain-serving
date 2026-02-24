"""
Structured logging configuration for the API and Inference Engine.
"""
import logging
import sys
from typing import Any

import structlog


def setup_logger(json_format: bool = False) -> structlog.BoundLogger:
    """
    Configures and returns a structured logger.
    
    Args:
        json_format: If True, outputs logs as JSON (best for production/cloud).
                     If False, outputs formatted text (best for local development).
                     
    Returns:
        A configured structlog logger instance.
    """
    # Shared processors for both JSON and Console output
    shared_processors: list[Any] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_format:
        # Production mode: strict JSON rendering
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Local development mode: colorful and readable console output
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Set base Python logging level to INFO to filter out debug noise from third-party libs
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )

    return structlog.get_logger("neuroseg_api")

# Export a default instance to be imported across the app
logger = setup_logger(json_format=False) # Will be toggled to True in prod via config later