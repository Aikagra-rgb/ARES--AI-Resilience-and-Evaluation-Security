"""
ARES Structured JSON Logging and Contextual Tracing.

Formats standard Python logging records as structured JSON documents and
propagates correlation_id and organization_id across async request contexts
using contextvars.
"""

from __future__ import annotations

import contextvars
from datetime import datetime, timezone
import json
import logging
import sys
from typing import Any, Dict, Optional

# Context variables for request tracing
_CORRELATION_ID_CTX: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "ares_correlation_id", default=None
)
_ORGANIZATION_ID_CTX: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "ares_organization_id", default=None
)


def get_correlation_id() -> Optional[str]:
    """Retrieve the current async context's correlation ID."""
    return _CORRELATION_ID_CTX.get()


def set_correlation_id(correlation_id: Optional[str]) -> None:
    """Set the current async context's correlation ID."""
    _CORRELATION_ID_CTX.set(correlation_id)


def get_organization_id() -> Optional[str]:
    """Retrieve the current async context's organization ID."""
    return _ORGANIZATION_ID_CTX.get()


def set_organization_id(organization_id: Optional[str]) -> None:
    """Set the current async context's organization ID."""
    _ORGANIZATION_ID_CTX.set(organization_id)


class StructuredJsonFormatter(logging.Formatter):
    """
    Formats standard library log records into newline-delimited JSON.
    Automatically enriches records with correlation_id and organization_id
    from the current execution context.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Context enrichment
        cid = getattr(record, "correlation_id", None) or get_correlation_id()
        if cid:
            log_obj["correlation_id"] = cid

        org_id = getattr(record, "organization_id", None) or get_organization_id()
        if org_id:
            log_obj["organization_id"] = org_id

        # Attach exception details if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        # Attach custom extra fields if provided
        standard_attrs = {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "message", "correlation_id", "organization_id"
        }
        extras = {k: v for k, v in record.__dict__.items() if k not in standard_attrs and not k.startswith("_")}
        if extras:
            log_obj["extra"] = extras

        return json.dumps(log_obj, ensure_ascii=False)


def configure_structured_logging(
    log_level: str = "INFO",
    json_format: bool = True,
) -> None:
    """
    Configure the root logger with either StructuredJsonFormatter or human-readable text.
    """
    root_logger = logging.getLogger()
    level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(level)

    # Avoid duplicate handlers on re-configuration
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(StructuredJsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] [%(name)s] [%(funcName)s:%(lineno)d] %(message)s"
            )
        )

    root_logger.addHandler(handler)
