from datetime import date

import numpy as np
import pandas as pd
import pytest

from cai.jobs.categorise import run_categorise
from spend import categorise as C
from spend import storage
from tests.test_storage import FakeImpala

CFG = {"min_similarity": 0.45, "knn": 3, "taxonomy_prior": 1.0, "seed_holdout_share": 0.5, "mcc_holdout_rows": 100}
RUN = date(2026, 9, 25)


def _seeds():
    rows = []
    for mid, cat, names in [
        ("M1", "GROCERY", ["Ramesh Kirana Store", "Sri Balaji Kirana", "Fresh Mart Supermarket", "Daily Kirana"]),
        ("M2", "HEALTH", ["Apollo Pharmacy", "Sai Medicals", "City Clinic", "Care Diagnostics"]),
        ("M3", "FUEL", ["Indian Oil Petrol Pump", "HP Fuel Station", "Bharat Petroleum"]),
    ]:
        for i, n in enumerate(names):
            rows.append({"merchant_id": f"{mid}{i}", "merchant_name": n, "category": cat,
                         "name_kind": "CANONICAL" if i == 0 else "VARIANT"})
    return pd.DataFrame(rows)


def _taxonomy():
    return pd.DataFrame({"category": ["GROCERY", "HEALTH", "FUEL", "OTHER"],
                         "description": ["grocery store, kirana, supermarket", "pharmacy, chemist, clinic, hospital",
                                         "petrol pump, fuel station, diesel", "anything else"]})


def _mcc():
    return pd.DataFrame({"mcc": [5411, 5912, 5541], "description": ["grocery", "drug stores", "service stations"],
                         "category": ["GROCERY", "HEALTH", "FUEL"]})


def _ref():
    return {"merchant_seed": _seeds(), "mcc_category": _mcc(), "category_taxonomy": _taxonomy()}


def test_knn_vote_is_similarity_weighted_and_the_prior_can_flip_it():
    sims = np.array([[0.9, 0.5, 0.5, 0.1]])
    cats = np.array(["A", "B", "B", "C"])
    cat, best, nearest = C.knn_vote(sims, cats, k=3)[0]
    assert (cat, nearest) == ("B", 0) and best == pytest.approx(0.9)
    cat, _, _ = C.knn_vote(np.array([[0.9, 0.3, 0.3, 0.1]]), cats, k=3)[0]
    assert cat == "A"
    prior = np.array([[0.0, 0.4]])
    cat, _, _ = C.knn_vote(np.array([[0.9, 0.3, 0.3, 0.1]]), cats, k=3, prior=prior,
                           prior_cats=np.array(["A", "B"]), prior_weight=1.0)[0]
    assert cat == "B"


def test_knn_vote_with_fewer_seeds_than_k():
    assert C.knn_vote(np.array([[0.7, 0.2]]), np.array(["A", "B"]), k=5)[0][0] == "A"


def test_assign_order_mcc_then_seed_then_embedding_then_fallback():
    emb = C.CharEmbedder()
    index = C.SeedIndex.build(_seeds(), _taxonomy(), emb)
    keys = ["sai medicals", "apollo pharmacy", "sri balaji kirana stor", "zzqx"]
    out = C.assign(keys, index, emb, CFG, mcc_category={"sai medicals": "FUEL"}).set_index("merchant_key")
    assert out.loc["sai medicals", ["category", "method"]].tolist() == ["FUEL", "MCC"]
    assert out.loc["apollo pharmacy", ["category", "method", "similarity"]].tolist() == ["HEALTH", "SEED", 1.0]
    assert out.loc["sri balaji kirana stor", ["category", "method"]].tolist() == ["GROCERY", "EMBEDDING"]
    assert out.loc["sri balaji kirana stor", "nearest_seed"] == "sri balaji kirana"
    assert out.loc["zzqx", ["category", "method"]].tolist() == ["OTHER", "FALLBACK"]


def test_char_embedder_must_be_fitted():
    with pytest.raises(RuntimeError, match="fit"):
        C.CharEmbedder().encode(["x"])


class _FakeDense:
    model_id = "fake-dense"
    device = "cpu"

    def encode(self, texts):
        return np.array([[1.0, 0.0] if "a" in t else [0.0, 1.0] for t in texts])

    @staticmethod
    def similarity(a, b):
        return a @ b.T


def test_hybrid_similarity_is_the_weighted_blend():
    h = C.HybridEmbedder(_FakeDense(), char_weight=0.25).fit(["abc", "xyz"])
    a, b = h.encode(["abc"]), h.encode(["abc", "xyz"])
    got = h.similarity(a, b)
    char = C.CharEmbedder().fit(["abc", "xyz"])
    want = 0.75 * np.array([[1.0, 0.0]]) + 0.25 * (char.encode(["abc"]) @ char.encode(["abc", "xyz"]).T).toarray()
    np.testing.assert_allclose(got, want)
    assert h.model_id == "fake-dense+char-2-4@0.25"


def test_modal_mcc_category_takes_the_most_frequent_mapped_mcc():
    cand = pd.DataFrame({"merchant_key": ["a", "a", "b", "c", "d"], "mcc": [5411, 5912, 9999, 5541, None],
                         "n_events": [5, 2, 3, 1, 4]})
    assert C.modal_mcc_category(cand, _mcc()) == {"a": "GROCERY", "c": "FUEL"}


def test_categorise_new_rows_and_lineage():
    cand = pd.DataFrame({"merchant_key": ["ramesh kirana", "hp fuel stn", "hp fuel stn", "x shop"],
                         "mcc": [None, 5541, 5411, None], "n_events": [3, 4, 1, 2]})
    out = C.categorise_new(cand, _ref(), C.CharEmbedder(), CFG, RUN, "run-1")
    assert list(out["merchant_key"]) == ["hp fuel stn", "ramesh kirana", "x shop"]
    assert out.set_index("merchant_key").loc["hp fuel stn", "method"] == "MCC"
    assert (out["model_id"] == C.STUB_MODEL_ID).all() and (out["first_run_date"] == RUN).all()
    assert (out["run_id"] == "run-1").all()
    assert C.categorise_new(cand.iloc[0:0], _ref(), C.CharEmbedder(), CFG, RUN, "r").empty


def test_seed_holdout_split_is_deterministic_and_never_holds_canonical_names():
    seeds = _seeds()
    idx1, held1 = C.seed_holdout_split(seeds, 0.5)
    idx2, held2 = C.seed_holdout_split(seeds, 0.5)
    pd.testing.assert_frame_equal(held1, held2)
    assert (held1["name_kind"] == "VARIANT").all() and len(idx1) + len(held1) == len(seeds)
    assert not set(held1["merchant_name"].map(C.normalise)) & set(idx1["merchant_name"].map(C.normalise))
    assert C.seed_holdout_split(seeds, 0.0)[1].empty


def test_holdout_predictions_and_accuracy():
    pred = C.seed_holdout_predictions(_seeds(), _taxonomy(), C.CharEmbedder(), CFG)
    assert set(pred.columns) >= {"merchant_key", "truth", "category", "similarity"}
    emb = C.CharEmbedder()
    index = C.SeedIndex.build(_seeds(), _taxonomy(), emb)
    card = pd.DataFrame({"merchant_key": ["sai medicals", "hp petrol pump"], "truth": ["HEALTH", "FUEL"]})
    got = C.mcc_holdout_predictions(card, index, emb, CFG)
    assert C.accuracy(got) == 1.0 and C.accuracy(got.iloc[0:0]) is None


def _events(keys_mcc):
    return pd.DataFrame({"merchant_key": [k for k, _ in keys_mcc], "mcc": [m for _, m in keys_mcc]})


def test_job_appends_new_keys_once_on_parquet(tmp_path):
    st = storage.ParquetStorage(tmp_path)
    for key, df in _ref().items():
        df.to_parquet(tmp_path / f"{key}.parquet", index=False)
    _events([("ramesh kirana", None), ("hp fuel stn", 5541), (None, None)]).to_parquet(
        tmp_path / "spend_event.parquet", index=False)

    dry = run_categorise(st, RUN, stub=True, write=False)
    assert dry["new_keys"] == 2 and not st.exists("merchant_category_map")

    first = run_categorise(st, RUN, stub=True)
    assert first["new_keys"] == 2 and first["unmapped_after"] == 0
    assert first["methods"] == {"MCC": 1, "EMBEDDING": 1}

    _events([("ramesh kirana", None), ("hp fuel stn", 5541), ("city clinic", None)]).to_parquet(
        tmp_path / "spend_event.parquet", index=False)
    st.refresh("spend_event")
    second = run_categorise(st, date(2026, 10, 2), stub=True)
    assert second["new_keys"] == 1 and second["methods"] == {"SEED": 1}
    m = st.read("merchant_category_map")
    assert len(m) == 3 and m.set_index("merchant_key").loc["ramesh kirana", "first_run_date"] == RUN
    assert run_categorise(st, RUN, stub=True)["new_keys"] == 0


def test_impala_new_keys_anti_joins_the_map_only_when_it_exists():
    st = FakeImpala()
    st.new_merchant_keys()
    assert "LEFT ANTI JOIN" not in st.sql[-1] and "GROUP BY e.merchant_key, e.mcc" in st.sql[-1]
    st = FakeImpala({"SHOW TABLES": pd.DataFrame({"name": ["merchant_category_map"]})})
    st.new_merchant_keys()
    assert "LEFT ANTI JOIN rsingh_spend_anl_ref.merchant_category_map m" in st.sql[-1]
