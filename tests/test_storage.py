from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from spend import storage
from spend.schema import OUTPUT_TABLES
from tests.conftest import ROOT


def _flow(run_date, n=2):
    return pd.DataFrame({"run_date": [run_date] * n, "run_id": ["r"] * n, "prev_run_date": [None] * n,
                         "from_persona_id": list(range(n)), "from_persona_name": ["a"] * n,
                         "to_persona_id": list(range(n)), "to_persona_name": ["a"] * n, "n_customers": [5] * n})


def test_parquet_replace_run_keeps_other_days(tmp_path):
    st = storage.ParquetStorage(tmp_path)
    st.replace_run("persona_flow", _flow(date(2026, 9, 24)), date(2026, 9, 24))
    st.replace_run("persona_flow", _flow(date(2026, 9, 25), 3), date(2026, 9, 25))
    st.replace_run("persona_flow", _flow(date(2026, 9, 25), 1), date(2026, 9, 25))
    df = st.read("persona_flow")
    assert sorted(pd.to_datetime(df["run_date"]).dt.date.value_counts().items()) == [
        (date(2026, 9, 24), 2), (date(2026, 9, 25), 1)]
    assert st.run_dates("persona_flow") == [date(2026, 9, 24), date(2026, 9, 25)]
    assert len(st.read("persona_flow", where_run_date=date(2026, 9, 24))) == 2


def test_parquet_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="run_cde_local"):
        storage.ParquetStorage(tmp_path).read("spend_features")


class FakeImpala(storage.ImpalaStorage):
    def __init__(self, results=None):
        super().__init__({"host": "fake", "insert_chunk": 2, "bulk_insert_chunk": 3})
        self.sql: list[str] = []
        self.results = results or {}

    def execute(self, sql):
        self.sql.append(sql)

    def query(self, sql):
        self.sql.append(sql)
        for key, df in self.results.items():
            if key in sql:
                return storage._numeric(df.copy())
        return pd.DataFrame()


def test_impala_replace_run_deletes_then_inserts_in_chunks():
    st = FakeImpala()
    st.replace_run("persona_flow", _flow(date(2026, 9, 25), 5), date(2026, 9, 25))
    create = [s for s in st.sql if s.startswith("CREATE TABLE")]
    assert "PARTITIONED BY SPEC (run_date)" in create[0] and "'format-version'='2'" in create[0]
    delete = st.sql.index(next(s for s in st.sql if s.startswith("DELETE")))
    inserts = [i for i, s in enumerate(st.sql) if s.startswith("INSERT")]
    assert len(inserts) == 3 and delete < inserts[0]
    assert "DATE '2026-09-25'" in st.sql[delete]


def test_impala_bulk_chunk_for_customer_persona():
    st = FakeImpala()
    cols = [c for c, _ in OUTPUT_TABLES["customer_persona"]]
    df = pd.DataFrame([{c: None for c in cols} | {"run_date": date(2026, 9, 25), "cif": f"C{i}"} for i in range(7)])
    st.append("customer_persona", df)
    assert len([s for s in st.sql if s.startswith("INSERT")]) == 3


def test_impala_numbers_stay_numeric():
    st = FakeImpala({"spend_model_run": pd.DataFrame({"wape_model": [Decimal("0.123")], "k": [7]})})
    df = st.read("spend_model_run")
    assert df["wape_model"].dtype == float and df["wape_model"].iloc[0] == pytest.approx(0.123)


def test_sql_literals():
    assert storage._sql_literal(None, "DOUBLE") == "NULL"
    assert storage._sql_literal(float("nan"), "DOUBLE") == "NULL"
    assert storage._sql_literal(pd.NA, "INT") == "NULL"
    assert storage._sql_literal("O'Brien\\x", "STRING") == "'O\\'Brien\\\\x'"
    assert storage._sql_literal(True, "BOOLEAN") == "true"
    assert storage._sql_literal(date(2026, 9, 25), "DATE") == "DATE '2026-09-25'"


def test_pinned_source():
    assert storage.ImpalaStorage.source("spend_features", "123").endswith("FOR SYSTEM_VERSION AS OF 123")
    assert "SYSTEM_VERSION" not in storage.ImpalaStorage.source("spend_features")


IMPALA_RESERVED = {line.strip() for line in (ROOT / "tests" / "data" / "impala_reserved_words.txt").read_text().splitlines()
                   if line.strip() and not line.startswith("#")}


def test_impala_sql_quotes_every_column():
    st = FakeImpala()
    cols = [c for c, _ in OUTPUT_TABLES["merchant_category_map"]]
    st.append("merchant_category_map", pd.DataFrame([{c: None for c in cols} | {"merchant_key": "k"}]))
    st.read("merchant_category_map", columns=["merchant_key", "category", "method"])
    create, insert, select = (next(s for s in st.sql if s.startswith(p)) for p in ("CREATE TABLE", "INSERT", "SELECT"))
    assert "`method` STRING" in create and "`method`" in insert and "`method`" in select
    assert all(f"`{c}`" in create for c in cols)


def test_reserved_column_names_are_quoted_in_the_report_sql():
    import re
    reserved = {c for cols in OUTPUT_TABLES.values() for c, _ in cols} & IMPALA_RESERVED
    assert reserved == {"method"}                       # a new one needs quoting in sql/reports.sql too
    sql = re.sub(r"--.*", "", (ROOT / "sql" / "reports.sql").read_text())
    for word in reserved:
        assert not re.search(rf"(?<![`\w]){word}(?![`\w])", sql), word
