"""What the CAI runtime needs from every script it runs as a job, model or
application, checked statically. A Jupyter-kernel job runtime runs a script
without __file__ and with an extra -f <kernel.json> argument, and reports any
SystemExit, even sys.exit(0), as a failed run, which would stop the DAG."""

import ast

import pytest

from spend.config import ROOT

SCRIPTS = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "cai").rglob("*.py") if p.name != "__init__.py")
SCRIPTS.append("app/run.py")


def _tree(path: str) -> ast.Module:
    return ast.parse((ROOT / path).read_text())


def _main_block(tree: ast.Module):
    return next((n for n in tree.body if isinstance(n, ast.If) and "__main__" in ast.unparse(n.test)), None)


@pytest.mark.parametrize("path", SCRIPTS)
def test_no_file_at_module_level(path):
    tree = _tree(path)
    top = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef))
           and "__file__" in ast.unparse(n)]
    assert not top, f"{path}: __file__ outside a function"
    if "__file__" in ast.unparse(tree):
        assert "except NameError" in ast.unparse(tree), f"{path}: no fallback when __file__ is missing"


@pytest.mark.parametrize("path", SCRIPTS)
def test_unknown_arguments_are_ignored(path):
    src = ast.unparse(_tree(path))
    assert ".parse_args(" not in src, f"{path}: parse_args() fails on the kernel's -f argument"


@pytest.mark.parametrize("path", SCRIPTS)
def test_no_exit_on_success(path):
    tree = _tree(path)
    main = _main_block(tree)
    if main is None:
        return
    for node in ast.walk(main):
        if isinstance(node, ast.Call) and ast.unparse(node.func) in ("sys.exit", "exit", "SystemExit"):
            parents = [n for n in ast.walk(main) if isinstance(n, ast.If) and node in ast.walk(n) and n is not main]
            assert parents, f"{path}: {ast.unparse(node)} runs on success too"
            assert ast.unparse(node) not in ("sys.exit(0)", "sys.exit()"), path


def test_every_cai_job_script_is_covered():
    jobs = {p.stem for p in (ROOT / "cai" / "jobs").glob("*.py") if p.name != "__init__.py"}
    assert {"categorise", "feature_engineering", "train_models", "evaluate_models", "daily_run",
            "backfill_history"} <= jobs
