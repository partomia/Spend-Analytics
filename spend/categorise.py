"""Merchant categorisation: MCC -> SEED -> EMBEDDING (MiniLM kNN) -> FALLBACK.

normalise() defines the merchant key. cde/jobs/build_silver.py computes the
same key in Spark (merchant_key()); tests/test_cde_jobs.py checks they agree.
"""

from __future__ import annotations

import re

_SEPARATORS = re.compile(r"[*#/_.-]")
_REFERENCE = re.compile(r"\b\d{3,}\b")                 # reference numbers, terminal ids
_SUFFIX = re.compile(r"\b(pvt|ltd|private|limited|india|in)\b")
_SPACES = re.compile(r"\s+")


def normalise(text: str | None) -> str | None:
    if text is None:
        return None
    t = text.lower()
    t = _SEPARATORS.sub(" ", t)
    t = _REFERENCE.sub(" ", t)
    t = _SUFFIX.sub(" ", t)
    t = _SPACES.sub(" ", t).strip()
    return t or None
