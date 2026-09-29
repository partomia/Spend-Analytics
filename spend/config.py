"""Load config/spend.yaml and config/policy.yaml with env var overrides.

Any scalar in spend.yaml can be overridden with SPEND_<SECTION>_<KEY>, e.g.
SPEND_STORAGE_BACKEND=parquet or SPEND_IMPALA_PASSWORD=... . Database and table
names are templates resolved after overrides, so changing the prefix renames
everything.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(os.environ.get("SPEND_CONFIG_DIR", ROOT / "config"))


def _coerce(value: str, like):
    if isinstance(like, bool):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(like, int):
        return int(value)
    if isinstance(like, float):
        return float(value)
    return value


def _apply_env(cfg: dict) -> dict:
    for section, values in cfg.items():
        if not isinstance(values, dict):
            continue
        for key, current in values.items():
            env = os.environ.get(f"SPEND_{section}_{key}".upper())
            if env is not None:
                values[key] = _coerce(env, current)
    return cfg


def _resolve_names(cfg: dict) -> dict:
    dbs = cfg["databases"]
    prefix = dbs["prefix"]
    for key in ("bronze", "silver", "gold", "ref"):
        dbs[key] = dbs[key].format(prefix=prefix)
    cfg["tables"] = {k: v.format(**dbs) for k, v in cfg["tables"].items()}
    return cfg


@lru_cache(maxsize=1)
def settings() -> dict:
    with open(CONFIG_DIR / "spend.yaml") as f:
        cfg = yaml.safe_load(f)
    return _resolve_names(_apply_env(cfg))


@lru_cache(maxsize=1)
def policy() -> dict:
    with open(CONFIG_DIR / "policy.yaml") as f:
        return yaml.safe_load(f)


def table(key: str) -> str:
    return settings()["tables"][key]
