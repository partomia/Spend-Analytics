#!/usr/bin/env python3
"""
Run a .sql file on CDW Impala statement by statement (the SPEND_IMPALA_*
connection from config/spend.yaml and .env), printing each statement's row
count and time, and the rows of each SELECT. Statements are split on ';' at
line ends; '--' comment lines are dropped.

  set -a; source .env; set +a
  python scripts/run_impala_sql.py sql/dataviz_views.sql
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SHOW_ROWS = 20


def statements(text: str) -> list[str]:
    body = "\n".join(line for line in text.splitlines() if not line.strip().startswith("--"))
    return [s.strip().rstrip(";") for s in body.split(";\n") if s.strip().rstrip(";").strip()]


def main() -> int:
    from spend.storage import get_storage

    if len(sys.argv) != 2:
        sys.exit(__doc__)
    storage = get_storage("impala")
    for sql in statements(Path(sys.argv[1]).read_text()):
        t0 = time.time()
        if sql.upper().startswith("SELECT"):
            rows = storage.query(sql)
            print(f"{time.time() - t0:6.1f}s  {len(rows):6d} rows  {' '.join(sql.split())[:90]}", flush=True)
            print("        " + " | ".join(rows.columns))
            for r in rows.head(SHOW_ROWS).itertuples(index=False):
                print("        " + " | ".join(str(v) for v in r))
        else:
            storage.execute(sql)
            print(f"{time.time() - t0:6.1f}s  {' '.join(sql.split())[:100]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
