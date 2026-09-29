"""sql/reports.sql runs only in Hue, so a renamed column would first show up in
front of an audience. Every identifier in a statement must be a keyword, an
alias the statement defines, or a column of a table it reads: columns come from
spend/schema.py (CAI outputs), spend/features.py (gold) and the CDE jobs' own
column lists (silver, dq_results)."""

import ast
import re

from spend import features
from spend.config import ROOT, settings
from spend.schema import OUTPUT_TABLES

SQL = (ROOT / "sql" / "reports.sql").read_text()
KEYWORDS = set("""select from where group by order desc asc limit with as case when then else end and or not
between in on join over partition distinct sum count max min avg round abs date_sub is null true false date
invalidate metadata refresh describe history""".split())


def _module_const(path: str, name: str):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == name)
    return ast.literal_eval(node.value)


def _columns() -> dict[str, set[str]]:
    dq = _module_const("cde/jobs/dq_check.py", "RESULT_SCHEMA")
    cols = {k: {c for c, _ in v} for k, v in OUTPUT_TABLES.items()}
    cols["spend_features"] = set(features.FEATURE_COLUMNS)
    cols["spend_cohort_weekly"] = set(features.COHORT_COLUMNS)
    cols["spend_mix_weekly"] = set(features.MIX_COLUMNS)
    cols["spend_event"] = set(_module_const("cde/jobs/build_silver.py", "EVENT_COLUMNS"))
    cols["dq_results"] = {part.split()[0] for part in dq.split(",")}
    return cols


def _statements() -> list[str]:
    body = re.sub(r"--[^\n]*", "", SQL)
    return [s.strip() for s in body.split(";") if s.strip()]


def test_tables_are_configured_names():
    configured = set(settings()["tables"].values())
    used = set(re.findall(r"\b(rsingh_spend_anl_\w+\.\w+)", SQL))
    assert used and used <= configured, used - configured


def test_every_identifier_is_a_column_of_a_table_it_reads():
    cols = _columns()
    name_of = {v: k for k, v in settings()["tables"].items()}
    checked = 0
    for stmt in _statements():
        text = re.sub(r"'[^']*'", "''", stmt.lower())
        tables = re.findall(r"\b(rsingh_spend_anl_\w+\.\w+)(?:\s+(?!where|group|order|limit|on|join)(\w+))?", text)
        known = set().union(*(cols[name_of[t]] for t, _ in tables))
        aliases = {a for _, a in tables if a}
        aliases |= set(re.findall(r"\bas\s+(\w+)", text))                  # output + CTE names
        aliases |= set(re.findall(r"\b(\w+)\s+as\s*\(", text))
        text = re.sub(r"\brsingh_spend_anl_\w+\.\w+", " ", text)
        for ident in set(re.findall(r"\b[a-z_][a-z0-9_]*\b", text)) - KEYWORDS - aliases:
            assert ident in known, f"{ident!r} is not a column of {[t for t, _ in tables]}:\n{stmt[:200]}"
            checked += 1
    assert checked > 50


def test_signal_and_mix_literals_exist():
    assert "WHERE dim = 'segment'" in SQL and "segment" in features.MIX_DIMENSIONS
    assert "is_home = 1" in SQL
