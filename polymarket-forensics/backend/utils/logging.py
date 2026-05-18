"""Structured logging.

Uses ``structlog`` for JSON output suitable for log aggregators. In dev
mode (LOG_FORMAT=console) it renders human-readable colored output instead.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog


def setup_logging(level: str = "INFO") -> None:
    """Initialize stdlib logging and structlog. Idempotent."""
    log_format = os.getenv("LOG_FORMAT", "console").lower()

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.StackInfoRenderer(),
        timestamper,
    ]

    if log_format == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True, sort_keys=False)

    structlog.configure(
        processors=shared_processors + [
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO),
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )
    for noisy in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
