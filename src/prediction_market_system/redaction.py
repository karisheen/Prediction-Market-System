"""Redact credential-bearing external text before it reaches audit or error sinks."""

from __future__ import annotations

import logging
import re
from typing import Any

_WEBHOOK = re.compile(r"(?i)(/api(?:/v\d+)?/webhooks/[^/\s?#]+/)[^/\s?#\"'<>]+")
_USERINFO = re.compile(r"(?i)(https?://)[^/\s@]+@")
_BEARER = re.compile(r"(?i)(\b(?:bearer|basic)\s+)[A-Za-z0-9._~+/=-]+")
_SECRET_VALUE = re.compile(
    r"(?i)([\"']?(?:[a-z0-9]+[_-])*(?:authorization|proxy-authorization|"
    r"api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|password|passwd|"
    r"credential|client[_-]?secret|webhook[_-]?url|signature)[\"']?\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s&,;}]+)"
)


def redact_secrets(text: str) -> str:
    """Keep useful diagnostics without URL, header, or key/value credentials."""
    text = _WEBHOOK.sub(r"\1[REDACTED]", text)
    text = _USERINFO.sub(r"\1[REDACTED]@", text)
    text = _BEARER.sub(r"\1[REDACTED]", text)
    return _SECRET_VALUE.sub(r"\1[REDACTED]", text)


def redact_payload(value: Any) -> Any:
    """Sanitize nested persisted diagnostics without corrupting JSON encoding."""
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]" if _SECRET_VALUE.match(f"{key}=value") else redact_payload(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_payload(item) for item in value]
    return redact_secrets(value) if isinstance(value, str) else value


class CredentialRedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_secrets(record.getMessage())
        record.args = ()
        return True


def protect_http_logs() -> None:
    """Retain HTTP diagnostics while redacting credentials before handler formatting."""
    logger = logging.getLogger("httpx")
    if not any(isinstance(item, CredentialRedactionFilter) for item in logger.filters):
        logger.addFilter(CredentialRedactionFilter())
