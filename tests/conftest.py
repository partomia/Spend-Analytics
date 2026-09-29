import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


@pytest.fixture(scope="session")
def generator():
    return _load(ROOT / "cde" / "jobs" / "generate_spend_bronze.py")


@pytest.fixture(scope="session")
def silver_job():
    pytest.importorskip("pyspark")
    return _load(ROOT / "cde" / "jobs" / "build_silver.py")


@pytest.fixture(scope="session")
def gold_job():
    pytest.importorskip("pyspark")
    return _load(ROOT / "cde" / "jobs" / "build_gold_features.py")


@pytest.fixture(scope="session")
def dq_job():
    pytest.importorskip("pyspark")
    pytest.importorskip("great_expectations")
    return _load(ROOT / "cde" / "jobs" / "dq_check.py")


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    s = (SparkSession.builder.master("local[2]").appName("spend-tests")
         .config("spark.sql.warehouse.dir", str(tmp_path_factory.mktemp("spark-warehouse")))
         .config("spark.sql.shuffle.partitions", "2").config("spark.ui.enabled", "false")
         .config("spark.sql.session.timeZone", "UTC").getOrCreate())
    yield s
    s.stop()
