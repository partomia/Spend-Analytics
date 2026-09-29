"""Merchant categorisation: MCC -> SEED -> EMBEDDING (MiniLM kNN) -> FALLBACK.

The cheapest reliable method goes first, and every merchant key records which
one decided it:

  MCC        card merchants whose (modal) MCC is in ref.mcc_category
  SEED       an exact normalised match with a name in ref.merchant_seed
  EMBEDDING  all-MiniLM-L6-v2 (+ character n-grams): the knn nearest seed names by
             cosine similarity, similarity-weighted vote, if the best similarity
             reaches policy.categorise.min_similarity
  FALLBACK   otherwise the nearest category description in ref.category_taxonomy,
             or OTHER if that is below the threshold too

normalise() defines the merchant key. cde/jobs/build_silver.py computes the
same key in Spark (merchant_key()); tests/test_cde_jobs.py checks they agree.

EMBEDDING blends MiniLM with a character n-gram TF-IDF (model.char_weight) and
adds the query's similarity to each category description as a prior
(policy.categorise.taxonomy_prior): MiniLM alone does not reach the accuracy
gate on payee names mangled by UPI apps (PLAN decision 11). The stub is the
character n-gram half alone (scikit-learn, no torch), with the same interface,
so CI and a laptop without torch run every code path.

Accuracy (part of the KPI gate, computed in the training stage):
  seed holdout  20% of the seed merchants' noisy variants are left out of the
                index and categorised with EMBEDDING only
  hidden MCC    card merchant keys with a mapped MCC, MCC hidden, through the
                path a UPI payee takes (SEED -> EMBEDDING -> FALLBACK)
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_SEPARATORS = re.compile(r"[*#/_.-]")
_REFERENCE = re.compile(r"\b\d{3,}\b")                 # reference numbers, terminal ids
_SUFFIX = re.compile(r"\b(pvt|ltd|private|limited|india|in)\b")
_SPACES = re.compile(r"\s+")
QUERY_CHUNK = 2048
STUB_MODEL_ID = "stub-tfidf-char-2-4"


def normalise(text: str | None) -> str | None:
    if text is None:
        return None
    t = text.lower()
    t = _SEPARATORS.sub(" ", t)
    t = _REFERENCE.sub(" ", t)
    t = _SUFFIX.sub(" ", t)
    t = _SPACES.sub(" ", t).strip()
    return t or None


# ---------------------------------------------------------------- embedders
#
# Every embedder has fit(corpus), encode(texts) and similarity(a, b) -> dense cosine matrix.

class CharEmbedder:
    """TF-IDF over character 2-4 grams (L2-normalised, sparse). On its own it is the CI / no-torch stub;
    inside HybridEmbedder it carries the spelling signal MiniLM's word pieces lose."""

    model_id = STUB_MODEL_ID
    device = "cpu"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=1)
        self._fitted = False

    def fit(self, corpus: list[str]) -> "CharEmbedder":
        self.vec.fit(list(corpus))
        self._fitted = True
        return self

    def encode(self, texts: list[str]):
        if not self._fitted:
            raise RuntimeError("CharEmbedder.fit() must see the seed corpus first")
        return self.vec.transform(list(texts))

    @staticmethod
    def similarity(a, b) -> np.ndarray:
        return (a @ b.T).toarray()


StubEmbedder = CharEmbedder


class MiniLMEmbedder:
    """sentence-transformers/all-MiniLM-L6-v2, 384-dim, unit-normalised."""

    def __init__(self, model_id: str | None = None, device: str | None = None, batch_size: int | None = None):
        from sentence_transformers import SentenceTransformer

        from spend.config import settings
        from spend.models import model_path, resolve_device

        cfg = settings()["model"]
        self.model_id = model_id or cfg["embedder"]
        self.device = resolve_device(device)
        self.batch_size = int(batch_size or cfg["embed_batch"])
        self.model = SentenceTransformer(model_path(self.model_id), device=self.device)

    def fit(self, corpus: list[str]) -> "MiniLMEmbedder":
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        return self.model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True,
                                 show_progress_bar=False, convert_to_numpy=True)

    @staticmethod
    def similarity(a, b) -> np.ndarray:
        return np.asarray(a @ b.T)


class HybridEmbedder:
    """Dense + sparse retrieval: similarity = (1 - w) x MiniLM cosine + w x character n-gram cosine, with
    w = model.char_weight. MiniLM brings meaning (a brand with no type word), the n-grams bring spelling
    (truncated, vowel-dropped and run-together payee names); PLAN decision 11."""

    def __init__(self, dense: MiniLMEmbedder, char_weight: float):
        self.dense, self.char, self.w = dense, CharEmbedder(), float(char_weight)
        self.model_id = f"{dense.model_id}+char-2-4@{self.w:g}"
        self.device = dense.device

    def fit(self, corpus: list[str]) -> "HybridEmbedder":
        self.char.fit(corpus)
        return self

    def encode(self, texts: list[str]):
        return self.dense.encode(texts), self.char.encode(texts)

    def similarity(self, a, b) -> np.ndarray:
        return (1 - self.w) * self.dense.similarity(a[0], b[0]) + self.w * self.char.similarity(a[1], b[1])


def load_embedder(stub: bool = False):
    if stub:
        return CharEmbedder()
    from spend.config import settings

    try:
        dense = MiniLMEmbedder()
    except ImportError:
        logger.warning("sentence-transformers not installed: using the character n-gram stand-in")
        return CharEmbedder()
    w = float(settings()["model"]["char_weight"])
    return HybridEmbedder(dense, w) if w > 0 else dense


def knn_vote(sims: np.ndarray, seed_cats: np.ndarray, k: int = 5, prior: np.ndarray | None = None,
             prior_cats: np.ndarray | None = None, prior_weight: float = 0.0) -> list[tuple[str, float, int]]:
    """For each query row of a similarity matrix: (voted category, best similarity, index of the nearest
    seed). The k nearest seeds vote with their similarity (averaged over k); prior adds prior_weight x the
    query's similarity to each category description."""
    k = min(k, sims.shape[1])
    idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    out = []
    for r, (row, ids) in enumerate(zip(sims, idx)):
        ids = ids[np.argsort(-row[ids])]
        votes: dict[str, float] = {}
        for i in ids:
            votes[seed_cats[i]] = votes.get(seed_cats[i], 0.0) + float(row[i]) / k
        if prior is not None and prior_weight:
            for c, v in zip(prior_cats, prior[r]):
                votes[c] = votes.get(c, 0.0) + prior_weight * float(v)
        out.append((max(votes, key=votes.get), float(row[ids[0]]), int(ids[0])))
    return out


# ---------------------------------------------------------------- the index

@dataclass
class SeedIndex:
    keys: np.ndarray            # normalised seed names
    cats: np.ndarray
    vecs: object
    exact: dict                 # key -> category (majority when a key maps to two)
    tax_cats: np.ndarray
    tax_vecs: object

    @classmethod
    def build(cls, seeds: pd.DataFrame, taxonomy: pd.DataFrame, embedder) -> "SeedIndex":
        s = seeds.assign(key=seeds["merchant_name"].map(normalise)).dropna(subset=["key"])
        s = (s.groupby(["key", "category"]).size().reset_index(name="n")
             .sort_values(["key", "n", "category"], ascending=[True, False, True]))
        exact = s.drop_duplicates("key").set_index("key")["category"].to_dict()
        tax_text = (taxonomy["category"].str.lower().str.replace("_", " ") + ": " + taxonomy["description"]).tolist()
        embedder.fit(list(s["key"]) + tax_text)
        return cls(keys=s["key"].to_numpy(), cats=s["category"].to_numpy(), vecs=embedder.encode(list(s["key"])),
                   exact=exact, tax_cats=taxonomy["category"].to_numpy(), tax_vecs=embedder.encode(tax_text))

    def embed_vote(self, keys: list[str], embedder, k: int, prior_weight: float = 0.0
                   ) -> list[tuple[str, float, str, str, float]]:
        """(voted category, best seed similarity, nearest seed, nearest category description, its similarity)."""
        out = []
        for lo in range(0, len(keys), QUERY_CHUNK):
            q = embedder.encode(keys[lo:lo + QUERY_CHUNK])
            sims = embedder.similarity(q, self.vecs)
            tax = embedder.similarity(q, self.tax_vecs)
            best_tax = tax.argmax(axis=1)
            for (cat, sim, i), b, trow in zip(knn_vote(sims, self.cats, k, tax, self.tax_cats, prior_weight),
                                              best_tax, tax):
                out.append((cat, sim, self.keys[i], self.tax_cats[b], float(trow[b])))
        return out


def assign(keys: list[str], index: SeedIndex, embedder, cfg: dict, mcc_category: dict[str, str] | None = None,
           use_seed: bool = True) -> pd.DataFrame:
    """Categorise merchant keys: columns merchant_key, category, method, similarity, nearest_seed."""
    mcc_category = mcc_category or {}
    min_sim = cfg["min_similarity"]
    rows: dict[str, tuple] = {}
    todo = []
    for key in keys:
        if key in mcc_category:
            rows[key] = (mcc_category[key], "MCC", 1.0, None)
        elif use_seed and key in index.exact:
            rows[key] = (index.exact[key], "SEED", 1.0, key)
        else:
            todo.append(key)
    for key, (cat, sim, nearest, tax_cat, tax_sim) in zip(
            todo, index.embed_vote(todo, embedder, cfg["knn"], cfg.get("taxonomy_prior", 0.0))):
        if sim >= min_sim:
            rows[key] = (cat, "EMBEDDING", sim, nearest)
        else:
            rows[key] = (tax_cat if tax_sim >= min_sim else "OTHER", "FALLBACK", tax_sim, nearest)
    return pd.DataFrame([(k, *rows[k]) for k in keys],
                        columns=["merchant_key", "category", "method", "similarity", "nearest_seed"])


def modal_mcc_category(candidates: pd.DataFrame, mcc_table: pd.DataFrame) -> dict[str, str]:
    """merchant_key -> category of its most frequent MCC, for keys whose modal MCC is mapped."""
    c = candidates.dropna(subset=["mcc"])
    if c.empty:
        return {}
    c = c.assign(mcc=c["mcc"].astype(int)).sort_values(["merchant_key", "n_events", "mcc"],
                                                       ascending=[True, False, True])
    top = c.drop_duplicates("merchant_key")
    cats = mcc_table.assign(mcc=mcc_table["mcc"].astype(int)).set_index("mcc")["category"]
    top = top[top["mcc"].isin(cats.index)]
    return dict(zip(top["merchant_key"], top["mcc"].map(cats)))


def categorise_new(candidates: pd.DataFrame, ref: dict[str, pd.DataFrame], embedder, cfg: dict, run_date: date,
                   run_id: str, index: SeedIndex | None = None) -> pd.DataFrame:
    """Map rows for merchant keys not yet in ref.merchant_category_map.

    candidates: merchant_key, mcc (NULL outside card), n_events (one row per key x MCC).
    ref: merchant_seed, mcc_category, category_taxonomy."""
    cols = ["merchant_key", "category", "method", "similarity", "nearest_seed", "model_id", "first_run_date",
            "run_id"]
    keys = sorted(candidates["merchant_key"].dropna().unique())
    if not keys:
        return pd.DataFrame(columns=cols)
    index = index or SeedIndex.build(ref["merchant_seed"], ref["category_taxonomy"], embedder)
    out = assign(keys, index, embedder, cfg, mcc_category=modal_mcc_category(candidates, ref["mcc_category"]))
    out["model_id"] = embedder.model_id
    out["first_run_date"] = run_date
    out["run_id"] = run_id
    logger.info("categorised %d new merchant keys: %s", len(out), out["method"].value_counts().to_dict())
    return out[cols]


# ---------------------------------------------------------------- accuracy (KPI gate)

def _bucket(text: str) -> float:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def seed_holdout_split(seeds: pd.DataFrame, share: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(index rows, held-out rows): a deterministic `share` of the VARIANT names leaves the index;
    canonical names always stay in."""
    held = (seeds["name_kind"] == "VARIANT") & (seeds["merchant_name"].map(lambda t: _bucket(f"h|{t}")) < share)
    keys = seeds["merchant_name"].map(normalise)
    index_keys = set(keys[~held])
    held &= ~keys.isin(index_keys)          # a variant that normalises to an indexed name is not unseen
    return seeds[~held].reset_index(drop=True), seeds[held].reset_index(drop=True)


def seed_holdout_predictions(seeds: pd.DataFrame, taxonomy: pd.DataFrame, embedder, cfg: dict) -> pd.DataFrame:
    """Held-out seed variants categorised with EMBEDDING only against an index that never saw them."""
    index_rows, held = seed_holdout_split(seeds, cfg["seed_holdout_share"])
    held = held.assign(merchant_key=held["merchant_name"].map(normalise)).dropna(subset=["merchant_key"])
    held = held.drop_duplicates("merchant_key")
    index = SeedIndex.build(index_rows, taxonomy, embedder)
    voted = index.embed_vote(list(held["merchant_key"]), embedder, cfg["knn"], cfg.get("taxonomy_prior", 0.0))
    return pd.DataFrame({"merchant_key": held["merchant_key"].to_numpy(), "truth": held["category"].to_numpy(),
                         "category": [v[0] for v in voted], "similarity": [v[1] for v in voted]})


def mcc_holdout_predictions(card_keys: pd.DataFrame, index: SeedIndex, embedder, cfg: dict) -> pd.DataFrame:
    """card_keys: merchant_key, truth (category of the modal MCC). The MCC is hidden and each key takes the
    path of a UPI payee: SEED -> EMBEDDING -> FALLBACK. A deterministic sample of mcc_holdout_rows keys."""
    n = int(cfg["mcc_holdout_rows"])
    s = card_keys.assign(_b=card_keys["merchant_key"].map(lambda t: _bucket(f"m|{t}"))).sort_values("_b").head(n)
    got = assign(list(s["merchant_key"]), index, embedder, cfg)
    return got.assign(truth=s["truth"].to_numpy())


def accuracy(pred: pd.DataFrame) -> float | None:
    return float((pred["category"] == pred["truth"]).mean()) if len(pred) else None


# ---------------------------------------------------------------- local Spark stage

SPARK_TYPES = {"STRING": "string", "DOUBLE": "double", "DATE": "date", "INT": "int", "BOOLEAN": "boolean",
               "TIMESTAMP": "timestamp", "BIGINT": "bigint"}


def categorise_spark(spark, db_prefix: str, stub: bool = False, run_date: date | None = None) -> pd.DataFrame:
    """scripts/run_cde_local.py categorise: the CAI job's code path, reading and writing the local Iceberg
    catalog through Spark instead of Impala."""
    from datetime import datetime, timezone

    from pyspark.sql import functions as F

    from spend.config import policy
    from spend.schema import OUTPUT_TABLES

    ref_db, silver = f"{db_prefix}_ref", f"{db_prefix}_silver"
    target = f"{ref_db}.merchant_category_map"
    events = spark.table(f"{silver}.spend_event").where(F.col("merchant_key").isNotNull())
    if spark.catalog.tableExists(target):
        events = events.join(spark.table(target).select("merchant_key"), "merchant_key", "left_anti")
    candidates = events.groupBy("merchant_key", "mcc").agg(F.count("*").alias("n_events")).toPandas()
    ref = {t: spark.table(f"{ref_db}.{t}").toPandas() for t in ("merchant_seed", "mcc_category", "category_taxonomy")}
    run_date = run_date or spark.table(f"{db_prefix}_bronze.crm_customer").agg(F.max("batch_as_of")).first()[0]
    run_id = f"categorise-{run_date:%Y%m%d}-{datetime.now(timezone.utc):%H%M%S}"
    out = categorise_new(candidates, ref, load_embedder(stub), policy()["categorise"], run_date, run_id)
    if out.empty:
        logger.info("no new merchant keys")
        return out
    schema = ", ".join(f"{c} {SPARK_TYPES[t]}" for c, t in OUTPUT_TABLES["merchant_category_map"])
    df = spark.createDataFrame(out.astype(object).where(out.notna(), None), schema)
    if spark.catalog.tableExists(target):
        df.writeTo(target).append()
    else:
        df.writeTo(target).using("iceberg").tableProperty("format-version", "2").create()
    total = spark.table(target).count()
    logger.info("%s: +%d keys, %d total", target, len(out), total)
    return out
