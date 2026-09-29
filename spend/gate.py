"""KPI gate over the three models (config/policy.yaml `gate`).

  categoriser  accuracy >= min_category_accuracy on the seed holdout and on the
               hidden-MCC sample
  forecaster   WAPE(P50) <= max_wape_vs_naive x WAPE(seasonal naive), and P10-P90
               coverage inside coverage_band
  personas     silhouette >= min_silhouette; stability >= min_persona_stability
               against the previous published run (skipped on the first run)

A KPI that could not be measured fails its check (except stability on a first
run), so a broken stage cannot pass the gate by producing nothing.
"""

from __future__ import annotations


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def evaluate(kpis: dict, cfg: dict) -> tuple[bool, list[str]]:
    checks = []

    def at_least(name, value, floor):
        checks.append((value is not None and value >= floor, f"{name} {_fmt(value)} >= {floor}"))

    acc = float(cfg["min_category_accuracy"])
    at_least("category_accuracy_seed", kpis.get("category_accuracy_seed"), acc)
    at_least("category_accuracy_mcc", kpis.get("category_accuracy_mcc"), acc)

    ratio, cap = kpis.get("wape_ratio"), float(cfg["max_wape_vs_naive"])
    checks.append((ratio is not None and ratio <= cap,
                   f"wape_ratio {_fmt(ratio)} <= {cap} (model {_fmt(kpis.get('wape_model'))}, "
                   f"naive {_fmt(kpis.get('wape_naive'))})"))
    lo, hi = (float(x) for x in cfg["coverage_band"])
    cover = kpis.get("coverage")
    checks.append((cover is not None and lo <= cover <= hi, f"coverage {_fmt(cover)} in [{lo}, {hi}]"))

    at_least("silhouette", kpis.get("silhouette"), float(cfg["min_silhouette"]))
    stab = kpis.get("persona_stability")
    if stab is None:
        checks.append((True, "persona_stability skipped (no previous run)"))
    else:
        at_least("persona_stability", stab, float(cfg["min_persona_stability"]))

    passed = all(ok for ok, _ in checks)
    return passed, [f"{'PASS' if ok else 'FAIL'} {line}" for ok, line in checks]
