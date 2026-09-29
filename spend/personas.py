"""Spend personas: KMeans on gold.spend_features for one snapshot.

Input: customers with at least personas.min_txn_13w transactions in 13 weeks,
the 24 share columns + log1p(spend_13w) + digital_share + home_share,
standardised and clipped at +-CLIP standard deviations (PLAN decision 14: a
few customers with one huge ticket otherwise pull clusters of their own).

k is chosen in k_min..k_max by silhouette on a sample; the previous run's k is
kept unless another k beats it by more than keep_k_within, so personas do not
split and merge on noise. IDs are stable: today's centroids are matched to the
previous run's (stored raw in persona_profile.centroid_json, re-scaled with
today's scaler) by the Hungarian algorithm, so persona 3 stays persona 3.
A cluster with no previous match gets a new ID.

Names are rules on the centroid, not on the model: the category the persona
over-indexes on most, the channel style (cash-first / digital) and the
dominant city tier, e.g. "Tier-1 digital foodie".
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from spend.features import (CAT_SHARES, CATEGORIES, CH_SHARES, CHANNELS, LOG_SPEND, PERSONA_FEATURES, TIER_SHARES,
                            TIERS)

logger = logging.getLogger(__name__)

CLIP = 3.0
SEED = 0
CATEGORY_WORD = {"GROCERY": "family grocer", "FOOD_DELIVERY": "foodie", "DINING": "diner", "FUEL": "commuter",
                 "TRAVEL": "traveller", "APPAREL": "shopper", "ELECTRONICS": "gadget buyer",
                 "UTILITIES": "bill payer", "TELECOM": "bill payer", "EMI_LOAN": "EMI borrower",
                 "EDUCATION": "school-fee family", "HEALTH": "health spender",
                 "ENTERTAINMENT": "entertainment seeker", "OTHER": "all-rounder"}


def persona_matrix(features: pd.DataFrame, min_txn: int) -> pd.DataFrame:
    """Profiled customers of one snapshot with the KMeans inputs (LOG_SPEND added)."""
    df = features[features["txn_13w"] >= min_txn].copy()
    df[LOG_SPEND] = np.log1p(df["spend_13w"].astype(float))
    return df.sort_values("cif").reset_index(drop=True)


@dataclass
class Scaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, raw: np.ndarray) -> "Scaler":
        scale = raw.std(axis=0)
        return cls(raw.mean(axis=0), np.where(scale > 0, scale, 1.0))

    def transform(self, raw: np.ndarray) -> np.ndarray:
        return np.clip((np.asarray(raw, dtype=float) - self.mean) / self.scale, -CLIP, CLIP)

    def to_json(self) -> str:
        return json.dumps({"features": PERSONA_FEATURES, "mean": self.mean.round(8).tolist(),
                           "scale": self.scale.round(8).tolist(), "clip": CLIP})

    @classmethod
    def from_json(cls, text: str) -> "Scaler":
        d = json.loads(text)
        return cls(np.array(d["mean"]), np.array(d["scale"]))


@dataclass
class PersonaFit:
    k: int
    labels: np.ndarray              # persona_id per row of the matrix
    distance: np.ndarray            # to the persona centroid, standardised space
    centroids_raw: dict[int, np.ndarray]
    names: dict[int, str]
    scaler: Scaler
    silhouette: float
    silhouette_by_k: dict[int, float]
    match_distance: dict[int, float | None] = field(default_factory=dict)


def _silhouette(x: np.ndarray, labels: np.ndarray, sample: np.ndarray) -> float:
    from sklearn.metrics import silhouette_score

    return float(silhouette_score(x[sample], labels[sample]))


def _sample(n: int, size: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return np.sort(rng.choice(n, size=min(n, size), replace=False))


def choose_k(x: np.ndarray, cfg: dict, prev_k: int | None) -> tuple[int, dict[int, float], dict]:
    from sklearn.cluster import KMeans

    sample = _sample(len(x), int(cfg["sample_rows"]))
    scores, models = {}, {}
    for k in range(int(cfg["k_min"]), int(cfg["k_max"]) + 1):
        if k >= len(x):
            break
        models[k] = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(x)
        scores[k] = round(_silhouette(x, models[k].labels_, sample), 6)
    if not scores:
        raise ValueError(f"only {len(x)} customers to profile, need more than k_min={cfg['k_min']}")
    best = max(scores, key=lambda k: (scores[k], -k))
    if prev_k in scores and scores[best] - scores[prev_k] <= float(cfg["keep_k_within"]):
        best = prev_k
    return best, scores, models


def match_ids(new_std: np.ndarray, prev_std: dict[int, np.ndarray]) -> tuple[dict[int, int], dict[int, float | None]]:
    """cluster index -> persona_id, and the matched distance (None for a new persona)."""
    from scipy.optimize import linear_sum_assignment

    if not prev_std:
        return {i: i for i in range(len(new_std))}, {i: None for i in range(len(new_std))}
    ids = sorted(prev_std)
    cost = np.linalg.norm(new_std[:, None, :] - np.stack([prev_std[i] for i in ids])[None], axis=2)
    rows, cols = linear_sum_assignment(cost)
    mapping = {int(r): ids[c] for r, c in zip(rows, cols)}
    dist = {int(r): float(cost[r, c]) for r, c in zip(rows, cols)}
    nxt = max(ids) + 1
    for i in range(len(new_std)):
        if i not in mapping:
            mapping[i], dist[i] = nxt, None
            nxt += 1
    return mapping, dist


def name_personas(centroids_raw: dict[int, np.ndarray], book_mean: np.ndarray) -> dict[int, str]:
    idx = {f: i for i, f in enumerate(PERSONA_FEATURES)}
    names = {}
    for pid, c in sorted(centroids_raw.items()):
        lift = {cat: c[idx[s]] / max(book_mean[idx[s]], 1e-9)
                for cat, s in zip(CATEGORIES, CAT_SHARES) if c[idx[s]] >= 0.05}
        word = CATEGORY_WORD[max(lift, key=lift.get)] if lift else "all-rounder"
        cash = c[idx["ch_atm_cash_share"]] / max(book_mean[idx["ch_atm_cash_share"]], 1e-9)
        style = "cash-first" if cash >= 1.5 else "digital" if c[idx["digital_share"]] >= 0.75 else ""
        tier = TIERS[int(np.argmax([c[idx[s]] for s in TIER_SHARES]))]
        names[pid] = " ".join(w for w in (f"Tier-{tier}", style, word) if w)
    seen: dict[str, int] = {}
    for pid in sorted(names):
        base = names[pid]
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            names[pid] = f"{base} {seen[base]}"
    return names


def fit_personas(matrix: pd.DataFrame, cfg: dict, prev_centroids: dict[int, np.ndarray] | None = None,
                 prev_k: int | None = None) -> PersonaFit:
    """prev_centroids: persona_id -> raw centroid (PERSONA_FEATURES order) of the previous run."""
    raw = matrix[PERSONA_FEATURES].to_numpy(dtype=float)
    scaler = Scaler.fit(raw)
    x = scaler.transform(raw)
    k, scores, models = choose_k(x, cfg, prev_k)
    km = models[k]
    prev_std = {pid: scaler.transform(c[None])[0] for pid, c in (prev_centroids or {}).items()}
    mapping, match_dist = match_ids(km.cluster_centers_, prev_std)
    labels = np.array([mapping[i] for i in km.labels_])
    dist = np.linalg.norm(x - km.cluster_centers_[km.labels_], axis=1)
    centroids_raw = {mapping[i]: raw[km.labels_ == i].mean(axis=0) for i in range(k)}
    names = name_personas(centroids_raw, raw.mean(axis=0))
    logger.info("personas: k=%d (silhouette by k %s), %s", k, scores,
                {names[p]: int((labels == p).sum()) for p in sorted(names)})
    return PersonaFit(k=k, labels=labels, distance=dist, centroids_raw=centroids_raw, names=names, scaler=scaler,
                      silhouette=scores[k], silhouette_by_k=scores,
                      match_distance={mapping[i]: d for i, d in match_dist.items()})


def assign(raw: np.ndarray, scaler: Scaler, centroids_raw: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Nearest persona for new rows (the endpoint): (persona_id, distance)."""
    ids = sorted(centroids_raw)
    c = scaler.transform(np.stack([centroids_raw[i] for i in ids]))
    d = np.linalg.norm(scaler.transform(raw)[:, None, :] - c[None], axis=2)
    return np.array(ids)[d.argmin(axis=1)], d.min(axis=1)


def stability(prev: pd.DataFrame | None, cur: pd.DataFrame) -> float | None:
    """Share of customers profiled in both runs who keep their persona_id."""
    if prev is None or prev.empty:
        return None
    m = cur[["cif", "persona_id"]].merge(prev[["cif", "persona_id"]], on="cif", suffixes=("", "_prev"))
    return float((m["persona_id"] == m["persona_id_prev"]).mean()) if len(m) else None


def ari_hidden(cur: pd.DataFrame, truth: pd.DataFrame | None) -> float | None:
    """Adjusted Rand index against the generator's hidden archetype (a demo diagnostic, not gated)."""
    if truth is None or truth.empty:
        return None
    from sklearn.metrics import adjusted_rand_score

    m = cur[["cif", "persona_id"]].merge(truth[["cif", "archetype"]], on="cif")
    return float(adjusted_rand_score(m["archetype"], m["persona_id"])) if len(m) else None


def centroids_from_profile(profile: pd.DataFrame | None) -> dict[int, np.ndarray]:
    if profile is None or profile.empty:
        return {}
    out = {}
    for pid, text in zip(profile["persona_id"], profile["centroid_json"]):
        d = json.loads(text)
        out[int(pid)] = np.array([d[f] for f in PERSONA_FEATURES], dtype=float)
    return out


def outputs(matrix: pd.DataFrame, fit: PersonaFit, prev_assign: pd.DataFrame | None, prev_run_date,
            prev_names: dict[int, str] | None = None) -> dict[str, pd.DataFrame]:
    """customer_persona, persona_profile, persona_flow rows (without run_date / run_id)."""
    cp = matrix[["snapshot_date", "cif", "home_city", "city_tier", "segment", "age_band", "spend_13w", "txn_13w",
                 "digital_share", "home_share"]].copy()
    cp["persona_id"] = fit.labels.astype(int)
    cp["persona_name"] = cp["persona_id"].map(fit.names)
    cp["distance"] = np.round(fit.distance, 4)
    cp["top_category"] = np.array(CATEGORIES)[matrix[CAT_SHARES].to_numpy().argmax(axis=1)]
    cp["top_channel"] = np.array(CHANNELS)[matrix[CH_SHARES].to_numpy().argmax(axis=1)]
    prev = prev_assign.set_index("cif")["persona_id"] if prev_assign is not None and len(prev_assign) else None
    cp["prev_persona_id"] = (cp["cif"].map(prev) if prev is not None else pd.Series(np.nan, index=cp.index)
                             ).astype("Int64")

    idx = {f: i for i, f in enumerate(PERSONA_FEATURES)}
    rows = []
    for pid in sorted(fit.names):
        c, members = fit.centroids_raw[pid], cp[cp["persona_id"] == pid]
        rows.append({
            "persona_id": pid, "persona_name": fit.names[pid], "n_customers": len(members),
            "customer_share": round(len(members) / len(cp), 6),
            "avg_spend_13w": round(float(members["spend_13w"].mean()), 2),
            "avg_txn_13w": round(float(members["txn_13w"].mean()), 2),
            "top_category": CATEGORIES[int(np.argmax([c[idx[s]] for s in CAT_SHARES]))],
            "top_channel": CHANNELS[int(np.argmax([c[idx[s]] for s in CH_SHARES]))],
            "dominant_tier": TIERS[int(np.argmax([c[idx[s]] for s in TIER_SHARES]))],
            "digital_share": round(float(c[idx["digital_share"]]), 6),
            "home_share": round(float(c[idx["home_share"]]), 6),
            "prev_match_distance": fit.match_distance.get(pid),
            "centroid_json": json.dumps({f: round(float(v), 8) for f, v in zip(PERSONA_FEATURES, c)}),
        })
    profile = pd.DataFrame(rows)

    flow = pd.DataFrame(columns=["prev_run_date", "from_persona_id", "from_persona_name", "to_persona_id",
                                 "to_persona_name", "n_customers"])
    if prev is not None:
        m = cp[cp["prev_persona_id"].notna()]
        flow = (m.groupby(["prev_persona_id", "persona_id"]).size().reset_index(name="n_customers")
                .rename(columns={"prev_persona_id": "from_persona_id", "persona_id": "to_persona_id"}))
        flow["from_persona_id"] = flow["from_persona_id"].astype(int)
        names = {**(prev_names or {})}
        flow["from_persona_name"] = flow["from_persona_id"].map(lambda p: names.get(p, fit.names.get(p)))
        flow["to_persona_name"] = flow["to_persona_id"].map(fit.names)
        flow["prev_run_date"] = prev_run_date
    return {"customer_persona": cp, "persona_profile": profile, "persona_flow": flow}
