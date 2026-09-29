"""Call the spend scorer: the CAI model endpoint if configured, otherwise
in-process (MiniLM + Chronos-2 on the local store, or the n-gram and
seasonal-naive stand-ins if torch is not installed, so the app still runs on a
light host)."""

from __future__ import annotations

import logging

import requests

from spend.config import settings
from spend.scoring import score

logger = logging.getLogger(__name__)
_local = None


def endpoint_configured() -> bool:
    return bool(settings()["endpoint"]["url"])


def call_endpoint(req: dict, timeout: int = 120) -> dict:
    ep = settings()["endpoint"]
    headers = {"Content-Type": "application/json"}
    if ep["api_key"]:
        headers["Authorization"] = f"Bearer {ep['api_key']}"
    resp = requests.post(ep["url"], json={"accessKey": ep["access_key"], "request": req},
                         headers=headers, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    if isinstance(body, dict) and "response" in body:
        return body["response"]
    return body


def local_context(source: str = "auto"):
    global _local
    if _local is None:
        from spend.scoring import load_context

        try:
            import chronos  # noqa: F401
            import sentence_transformers  # noqa: F401

            _local = load_context(source)
        except ImportError:
            logger.warning("sentence-transformers / chronos not installed: using the n-gram and seasonal-naive "
                           "stand-ins")
            _local = load_context(source, stub=True)
    return _local


def reset_local() -> None:
    global _local
    _local = None


def spend_request(req: dict) -> tuple[dict, str]:
    """Returns (response, source) where source is 'endpoint' or 'in-process'."""
    if endpoint_configured():
        return call_endpoint(req), "endpoint"
    return score(req, local_context()), "in-process"
