#!/usr/bin/env python3
"""
Phase 0 spike: time the two pretrained models on this machine before the
pipeline depends on them, and record the numbers in PLAN.md decisions.

  MiniLM     texts per second, embedding 10,000 noisy payee names
  Chronos-2  seconds for 224 weekly series x 12 weeks ahead with covariates

  python scripts/spike_models.py                  # cuda if available, else cpu
  python scripts/spike_models.py --device cpu
  python scripts/spike_models.py --device mps     # Apple silicon
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

WORDS = ["swiggy", "zomato", "kirana", "provisions", "medicals", "fuel", "petrol", "travels", "textiles",
         "mobiles", "tuition", "clinic", "hotel", "tiffin", "bakery", "electricals", "cabs", "pharmacy"]


def texts(n: int, rng: np.random.Generator) -> list[str]:
    out = []
    for _ in range(n):
        w = rng.choice(WORDS, size=rng.integers(1, 4))
        out.append(f"{' '.join(w).upper()}*{rng.integers(100, 9999)} {rng.choice(['BLR', 'MUM', 'DEL', ''])}")
    return out


def series(n_series: int, weeks: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2025-04-04", periods=weeks, freq="W-FRI")
    fut = pd.date_range(idx[-1] + pd.Timedelta(weeks=1), periods=12, freq="W-FRI")
    hist, future = [], []
    for s in range(n_series):
        base = rng.uniform(1e5, 1e7)
        season = 1 + 0.3 * np.sin(np.arange(weeks) * 2 * np.pi / 52)
        fest = (np.arange(weeks) % 52 == 28).astype(float)
        y = base * season * (1 + 0.8 * fest) * rng.lognormal(0, 0.08, weeks)
        hist.append(pd.DataFrame({"series_id": f"s{s}", "week_end": idx, "amount_inr": y, "festival_week": fest}))
        future.append(pd.DataFrame({"series_id": f"s{s}", "week_end": fut,
                                    "festival_week": (np.arange(weeks, weeks + 12) % 52 == 28).astype(float)}))
    return pd.concat(hist, ignore_index=True), pd.concat(future, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="")
    p.add_argument("--texts", type=int, default=10_000)
    p.add_argument("--series", type=int, default=224)
    p.add_argument("--weeks", type=int, default=77)
    args = p.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(42)
    print(f"torch {torch.__version__}, device {device}, threads {torch.get_num_threads()}")

    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    print(f"MiniLM loaded in {time.time() - t0:.1f}s")
    batch = texts(args.texts, rng)
    model.encode(batch[:256], batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    t0 = time.time()
    vecs = model.encode(batch, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    dt = time.time() - t0
    print(f"MiniLM: {len(batch):,} texts in {dt:.2f}s = {len(batch) / dt:,.0f} texts/s, dim {vecs.shape[1]}")

    from chronos import Chronos2Pipeline

    t0 = time.time()
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)
    print(f"Chronos-2 loaded in {time.time() - t0:.1f}s")
    hist, future = series(args.series, args.weeks, rng)
    keys = dict(id_column="series_id", timestamp_column="week_end", target="amount_inr")
    pipe.predict_df(hist[hist["series_id"] == "s0"], future_df=future[future["series_id"] == "s0"],
                    prediction_length=12, quantile_levels=[0.1, 0.5, 0.9], **keys)
    t0 = time.time()
    pred = pipe.predict_df(hist, future_df=future, prediction_length=12, quantile_levels=[0.1, 0.5, 0.9], **keys)
    dt = time.time() - t0
    print(f"Chronos-2: {args.series} series x 12 weeks ({args.weeks} weeks of context, 1 covariate) in {dt:.2f}s; "
          f"columns {list(pred.columns)}")


if __name__ == "__main__":
    main()
