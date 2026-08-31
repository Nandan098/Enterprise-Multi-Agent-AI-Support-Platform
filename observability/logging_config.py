from __future__ import annotations

import json
import logging
import time

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        for key in ("route", "latency_ms", "retrieval_count", "validation"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload)

def configure_logging() -> logging.Logger:
    logger = logging.getLogger("ace_project")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    return logger

logger = configure_logging()
