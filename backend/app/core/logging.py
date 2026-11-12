import os
import re
import logging
import structlog

_REDACTED = "[REDACTED]"

# Field names whose values are always secret (R-37). Matched case-insensitively
# after normalising '-' to '_'.
_SECRET_KEYS = {
    "authorization", "api_key", "apikey", "x_api_key", "password", "passwd",
    "secret", "token", "access_token", "refresh_token", "jwt", "jwt_secret",
    "redis_password", "neo4j_password", "chroma_auth_token", "cookie",
}

_VALUE_PATTERNS = [
    # ?key=XXX / &key=XXX (Gemini style) and other secret-looking query params
    (re.compile(r"([?&](?:key|api_key|apikey|token|access_token|password)=)[^&\s]*", re.I), r"\1" + _REDACTED),
    # Authorization: Bearer xxx / bare "Bearer xxx"
    (re.compile(r"(\bbearer\s+)[A-Za-z0-9._~+/=-]+", re.I), r"\1" + _REDACTED),
    # scheme://user:pass@host  -> scheme://user:[REDACTED]@host
    (re.compile(r"([a-z][a-z0-9+.-]*://[^:/@\s]*:)[^@\s/]+@", re.I), r"\1" + _REDACTED + "@"),
    # key=value / key: value pairs inside free text
    (re.compile(r"\b(password|passwd|secret|api[_-]?key|x-api-key|token|jwt)(\s*[=:]\s*)[^\s,;&]+", re.I),
     r"\1\2" + _REDACTED),
]


def sanitize_error_msg(text: str) -> str:
    """Remove credentials (API keys, bearer tokens, URL passwords) from text before logging."""
    if not text:
        return ""
    text = str(text)
    for pattern, repl in _VALUE_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def _is_secret_key(key: str) -> bool:
    return str(key).lower().replace("-", "_") in _SECRET_KEYS


def _redact(value):
    if isinstance(value, str):
        return sanitize_error_msg(value)
    if isinstance(value, dict):
        return {k: (_REDACTED if _is_secret_key(k) else _redact(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_redact(v) for v in value)
    return value


def redact_processor(logger, method_name, event_dict):
    """structlog processor: redact secret keys and secret-looking values."""
    for key in list(event_dict.keys()):
        if _is_secret_key(key):
            event_dict[key] = _REDACTED
        else:
            event_dict[key] = _redact(event_dict[key])
    return event_dict


def setup_logging():
    log_format = os.getenv("LOG_FORMAT", "json").lower()

    processors = [
        structlog.contextvars.merge_contextvars,  # binds request_id per request
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        redact_processor,  # after format_exc_info so tracebacks are redacted too
    ]

    if log_format == "console":
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(format="%(message)s", level=logging.INFO)


def get_logger(name: str = None):
    return structlog.get_logger(name)
