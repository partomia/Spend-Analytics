"""The Data Visualization dashboard is generated from one declaration, reads only the report
views, stays inside this project's names on the shared instance, and its export file carries
no credentials."""

import json
import re

from tests.conftest import ROOT, _load

BUILD = _load(ROOT / "dataviz" / "build_dashboard.py")
VIEWS_SQL = (ROOT / "sql" / "dataviz_views.sql").read_text()
EXPORT = json.loads((ROOT / "dataviz" / "spend_dashboards.json").read_text())
VISUALS = list(BUILD.visuals())
REPORT_DB = "rsingh_spend_anl_report"


def test_every_dataset_is_a_report_view():
    created = set(re.findall(rf"CREATE VIEW {REPORT_DB}\.(\w+) AS", VIEWS_SQL))
    assert BUILD.DB == REPORT_DB
    assert {view for _, view, _ in BUILD.DATASETS.values()} == created
    assert {d["fields"]["dataset_detail"] for d in EXPORT["datasets"]} == {f"{BUILD.DB}.{v}" for v in created}


def test_names_on_the_shared_instance_say_which_project_they_belong_to():
    assert BUILD.CONNECTION.startswith("rsingh-spend-anl") and BUILD.WORKSPACE == "rsingh-spend-anl"
    assert BUILD.DASHBOARD["title"].startswith("Spend Analytics - ")
    assert all(name.startswith("Spend - ") for name, _, _ in BUILD.DATASETS.values())


def test_the_export_file_matches_the_declaration():
    assert len(EXPORT["visuals"]) == len(VISUALS)
    (rec,) = EXPORT["dashboards"]
    body = json.loads(rec["fields"]["report_data"])
    assert [s["sheet_handle_title"] for s in body["dashboard_sheets"]] == [s for s, _ in BUILD.SHEETS]
    assert rec["fields"]["report_name"] == BUILD.DASHBOARD["title"] and rec["pk"] == BUILD.DASHBOARD_PK
    placed = {w["id"].rsplit("-", 1)[1] for s in body["dashboard_sheets"] for w in s["visual_widgets"]}
    assert placed == {str(v["pk"]) for v in EXPORT["visuals"]}
    uuids = [a["fields"]["uuid"] for k in ("datasets", "visuals", "dashboards") for a in EXPORT[k]]
    assert len(set(uuids)) == len(uuids)


def test_visuals_only_use_columns_of_their_dataset():
    for sheet, v in VISUALS:
        view = BUILD.DATASETS[v["ds"]][1]
        body = VIEWS_SQL.split(f"CREATE VIEW {REPORT_DB}.{view} AS", 1)[1].split(";", 1)[0]
        used = {c for c, _ in v.get("dims", []) + v.get("x", []) + v.get("color", [])}
        used |= set(re.findall(r"\[(\w+)\]", " ".join([e for e, _ in v["measures"]] + v.get("filters", []))))
        missing = {c for c in used if not re.search(rf"\b{c}\b", body)}
        assert not missing, (sheet, v["title"], missing)


def test_spend_mix_visuals_filter_one_dimension():
    for sheet, v in VISUALS:
        if v["ds"] == "mix":
            assert sum("[dim] =" in f for f in v.get("filters", [])) == 1, (sheet, v["title"])


def test_visuals_fit_the_64_column_grid_without_overlap():
    for sheet, items in BUILD.SHEETS:
        cells = set()
        for v in items:
            c, r, w, h = v["pos"]
            assert c >= 1 and c + w - 1 <= 64, (sheet, v["title"])
            box = {(x, y) for x in range(c, c + w) for y in range(r, r + h)}
            assert not cells & box, (sheet, v["title"])
            cells |= box


def test_only_this_projects_visuals_are_moved_into_its_workspace():
    viz = BUILD.DataViz.__new__(BUILD.DataViz)
    viz.url = "https://viz.example"
    datasets = [{"id": 37, "name": "Spend - Weekly mix"}, {"id": 26, "name": "Collections DLQ - Call list"}]
    visuals = [{"id": 203, "dataset_id": 37, "workspace_id": 2}, {"id": 204, "dataset_id": 37, "workspace_id": 6},
               {"id": 90, "dataset_id": 26, "workspace_id": 2}]
    for v in visuals:
        v.update(title="t", type="dashboard", description="d", data={})
    viz.get = lambda path, **kw: datasets if path.endswith("datasets") else visuals
    posted = []

    class Session:
        def post(self, url, data, timeout):
            posted.append((url, json.loads(data["data"])[0]["workspace_id"]))
            return type("R", (), {"status_code": 200, "text": ""})()

    viz.s = Session()
    viz.move_to_workspace(6)
    assert posted == [("https://viz.example/arc/adminapi/v1/visuals/203", 6)]


def test_no_credentials_in_the_export_and_nothing_in_the_pipeline_reads_the_views():
    text = json.dumps(EXPORT).lower()
    assert not any(w in text for w in ("password", "apikey", "bearer"))
    users = [p for p in [*ROOT.glob("cde/**/*.py"), *ROOT.glob("cai/**/*.py"), *ROOT.glob("spend/**/*.py"),
                         *ROOT.glob("app/**/*.py")] if REPORT_DB in p.read_text()]
    assert users == []
