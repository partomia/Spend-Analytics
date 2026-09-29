"""The bronze generator's pure-Python simulator: determinism, prefix stability, keys,
volumes and the planted demo customers. Spark is not needed."""

import re
from collections import Counter
from datetime import date

import pytest

AS_OF = date(2026, 9, 25)
EARLIER = date(2026, 6, 30)
TS = {4: 2, 5: 6, 6: 4, 7: 3}          # timestamp position in card_txn, upi_txn, nb_txn, atm_txn rows
N = 400


@pytest.fixture(scope="module")
def book(generator):
    return [generator.unit_rows(generator.simulate_unit(i, generator.DEFAULT_SEED, AS_OF), AS_OF) for i in range(N)]


def _category_of(generator):
    lookup = {}
    for m in generator.merchant_universe(generator.DEFAULT_SEED):
        for v in [m["name"], *m["variants"]]:
            lookup[v] = m["category"]
            lookup[m["name"].upper()[:24]] = m["category"]
    return lookup


def test_deterministic(generator):
    a = generator.unit_rows(generator.simulate_unit(7, 11, AS_OF), AS_OF)
    b = generator.unit_rows(generator.simulate_unit(7, 11, AS_OF), AS_OF)
    assert a == b
    c = generator.unit_rows(generator.simulate_unit(7, 12, AS_OF), AS_OF)
    assert a != c


@pytest.mark.parametrize("idx", [0, 1, 2, 5, 42, 399, 150_000])
def test_prefix_stable(generator, idx):
    """A later as-of adds rows and never changes an earlier one."""
    early = generator.unit_rows(generator.simulate_unit(idx, generator.DEFAULT_SEED, EARLIER), EARLIER)
    late = generator.unit_rows(generator.simulate_unit(idx, generator.DEFAULT_SEED, AS_OF), AS_OF)
    for k, pos in TS.items():
        assert [r for r in late[k] if r[pos].date() <= EARLIER] == early[k]
    for k in (0, 1, 2, 3, 8):
        assert set(early[k]) <= set(late[k])


def test_master_tables_do_not_need_the_simulation(generator):
    full = generator.unit_rows(generator.simulate_unit(9, generator.DEFAULT_SEED, AS_OF), AS_OF)
    light = generator.unit_rows(generator.simulate_unit(9, generator.DEFAULT_SEED, AS_OF, events=False), AS_OF)
    assert full[:4] == light[:4] and full[8] == light[8]
    assert all(not light[k] for k in TS)


def test_keys_unique_across_customers(book):
    for k, key in [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)]:
        ids = [r[key] for rows in book for r in rows[k]]
        assert len(ids) == len(set(ids)), k


def test_every_event_resolves_to_a_customer(book):
    for rows in book:
        vpas, cards = {v[0] for v in rows[3]}, {c[0] for c in rows[2]}
        accounts = {a[0] for a in rows[1]}
        assert all(t[1] in vpas for t in rows[5])
        assert all(t[1] in cards for t in rows[4]) and all(t[1] in cards for t in rows[7])
        assert all(v[1] in accounts for v in rows[3]) and all(c[1] in accounts for c in rows[2])
        if rows[0]:
            assert all(t[1] == rows[0][0][0] for t in rows[6])


def test_volume_and_channel_mix(book):
    n_cust = sum(1 for rows in book if rows[0])
    counts = Counter({"card": sum(len(r[4]) for r in book), "upi": sum(len(r[5]) for r in book),
                      "nb": sum(len(r[6]) for r in book), "atm": sum(len(r[7]) for r in book)})
    total = sum(counts.values())
    assert 125 <= total / n_cust <= 175            # ~25-30M events at 200k customers
    assert 0.55 <= counts["upi"] / total <= 0.65   # ~60% UPI by design
    p2p = sum(1 for r in book for t in r[5] if t[5] == "P2P")
    assert 0.15 <= p2p / counts["upi"] <= 0.35


def test_amounts_and_formats(book, generator):
    atm_ids = {a[0] for a in generator.ref_rows(generator.DEFAULT_SEED)["atm_location"]}
    pins = {p[0] for p in generator.ref_rows(generator.DEFAULT_SEED)["pincode_geo"]}
    for rows in book:
        assert all(t[3] > 0 for t in rows[4]) and all(t[7] > 0 for t in rows[5]) and all(t[5] > 0 for t in rows[6])
        assert all(t[4] > 0 and t[4] % 100 == 0 and t[2] in atm_ids for t in rows[7])
        assert all(c[2] in pins for c in rows[0])
        assert all(t[4] is None or t[4] in pins for t in rows[5])
        assert all(t[5] in ("P2P", "P2M") for t in rows[5])
        assert all(t[7] in ("CHIP", "CONTACTLESS", "MAGSTRIPE", "ECOM") for t in rows[4])
        assert all(re.fullmatch(r"[0-9a-f]{32}", c[0]) for c in rows[2])


def test_no_raw_card_number_in_text(book):
    raw = re.compile(r"\d{16}")
    for rows in book:
        for t in rows[4]:
            assert not raw.search(t[5] or "")
        for t in rows[5]:
            assert not any(raw.search(x or "") for x in (t[1], t[2], t[3], t[8]))


def test_archetypes_all_present(book):
    arch = Counter(rows[8][0][1] for rows in book if rows[8])
    assert len(arch) == 7 and min(arch.values()) >= 15


def test_merchant_universe(generator):
    mu = generator.merchant_universe(generator.DEFAULT_SEED)
    assert 7_500 <= len(mu) <= 8_500
    assert len({m["merchant_id"] for m in mu}) == len(mu)
    assert all(2 <= len([v for v in m["variants"] if v != m["name"]]) <= 5 for m in mu)
    assert {m["category"] for m in mu} == set(generator.CATEGORIES)
    assert 0.25 <= sum(m["seed"] for m in mu) / len(mu) <= 0.40


def test_ref_tables(generator):
    ref = generator.ref_rows(generator.DEFAULT_SEED)
    assert set(ref) == {"pincode_geo", "mcc_category", "category_taxonomy", "atm_location", "merchant_seed",
                        "calendar_weekly"}
    assert [r[1] for r in ref["category_taxonomy"]] == generator.CATEGORIES
    assert {r[3] for r in ref["pincode_geo"]} == {1, 2, 3, 4}
    assert {r[2] for r in ref["mcc_category"]} <= set(generator.CATEGORIES)
    assert not {m for m, _, _ in generator.MCC_CATEGORY} & set(generator.MCC_UNMAPPED)
    assert {r[3] for r in ref["merchant_seed"]} == {"CANONICAL", "VARIANT"}
    for name, rows in ref.items():
        assert len(generator.REF_SCHEMAS[name].split(",")) == len(rows[0]), name


def test_calendar_weeks(generator):
    cal = generator.calendar_rows()
    ends = [r[0] for r in cal]
    assert all(d.weekday() == 4 for d in ends)                            # Friday week_end
    assert all((b - a).days == 7 for a, b in zip(ends, ends[1:]))
    assert all((r[0] - r[1]).days == 6 for r in cal)                      # Saturday start
    assert cal[0][1] <= generator.HISTORY_START <= cal[0][0] and ends[-1] >= date(2027, 6, 30)
    diwali = {r[0] for r in cal if r[3] == "Diwali"}
    assert date(2025, 10, 24) in diwali and date(2025, 10, 17) in diwali  # the week of and before


def test_demo_travel_surge(generator):
    rows = generator.unit_rows(generator.simulate_unit(0, generator.DEFAULT_SEED, AS_OF), AS_OF)
    cat = _category_of(generator)

    def travel(lo, hi):
        card = sum(t[3] for t in rows[4] if lo <= t[2].date() <= hi and cat.get(t[5]) == "TRAVEL")
        upi = sum(t[7] for t in rows[5] if lo <= t[6].date() <= hi and cat.get(t[3]) == "TRAVEL")
        return card + upi
    recent = travel(date(2026, 8, 29), AS_OF)
    norm_weekly = travel(date(2026, 6, 27), date(2026, 8, 28)) / 9
    assert recent >= 5_000 and recent >= 3 * 4 * norm_weekly
    assert generator.unit_truth(generator.simulate_unit(0, generator.DEFAULT_SEED))["trips"][0][2] == "GOA"


def test_demo_relocation(generator):
    u = generator.simulate_unit(1, generator.DEFAULT_SEED, AS_OF)
    rows = generator.unit_rows(u, AS_OF)
    assert u.customer["home_city"] == "HYD" and u.customer["relocate"] == (date(2026, 8, 24), "NAG")
    assert rows[0][0][2].startswith("500")                               # CRM keeps the Hyderabad pincode
    after = [t[4][:3] for t in rows[5] if t[6].date() >= date(2026, 8, 24) and t[4]]
    before = [t[4][:3] for t in rows[5] if date(2026, 6, 1) <= t[6].date() < date(2026, 8, 24) and t[4]]
    assert after and all(p == "440" for p in after)
    assert not before or Counter(before).most_common(1)[0][0] == "500"
    assert rows[8][0][3:] == ("NAG", date(2026, 8, 24))


def test_demo_festive_electronics(generator):
    as_of = date(2025, 10, 24)
    rows = generator.unit_rows(generator.simulate_unit(2, generator.DEFAULT_SEED, as_of), as_of)
    cat = _category_of(generator)
    lo = date(2025, 9, 27)
    card = sum(t[3] for t in rows[4] if t[2].date() >= lo and cat.get(t[5]) == "ELECTRONICS")
    upi = sum(t[7] for t in rows[5] if t[6].date() >= lo and cat.get(t[3]) == "ELECTRONICS")
    assert card + upi >= 50_000


def test_resent_rate(generator):
    keys = [f"U{i:012d}" for i in range(200_000)]
    rate = sum(generator.is_resent(k) for k in keys) / len(keys)
    assert 0.0015 <= rate <= 0.0025


def test_arrow_partition_emits_duplicates(generator):
    pa = pytest.importorskip("pyarrow")
    out, table, dupe_key, cols = next(o for o in generator.OUTPUTS if o[1] == "bronze.upi_txn")
    fn = generator._arrow_partition(generator.DEFAULT_SEED, AS_OF, out, cols, dupe_key)
    batches = list(fn([pa.RecordBatch.from_pydict({"id": list(range(60))})]))
    tbl = pa.Table.from_batches(batches)
    assert tbl.column_names == [n for n, _ in cols] + ["resent"]
    assert str(tbl.schema.field("txn_ts").type) == "timestamp[us, tz=UTC]"
    resent = tbl.filter(pa.compute.equal(tbl["resent"], 1))["txn_id"].to_pylist()
    originals = set(tbl.filter(pa.compute.equal(tbl["resent"], 0))["txn_id"].to_pylist())
    assert resent and set(resent) <= originals


def test_arrow_timestamps_keep_wall_clock(generator):
    pa = pytest.importorskip("pyarrow")
    out, _, dupe_key, cols = next(o for o in generator.OUTPUTS if o[1] == "bronze.nb_txn")
    tbl = pa.Table.from_batches(list(generator._arrow_partition(
        generator.DEFAULT_SEED, AS_OF, out, cols, dupe_key)([pa.RecordBatch.from_pydict({"id": [3]})])))
    rows = generator.unit_rows(generator.simulate_unit(3, generator.DEFAULT_SEED, AS_OF), AS_OF)[6]
    got = [t.replace(tzinfo=None) for t in tbl.filter(pa.compute.equal(tbl["resent"], 0))["txn_ts"].to_pylist()]
    assert got == [r[4] for r in rows]


def test_customer_cap(generator):
    with pytest.raises(SystemExit):
        generator.parse_args(["--customers", str(generator.MAX_CUSTOMERS + 1)])
    args = generator.parse_args(["--as-of", "2026-09-25", "--unknown-flag"])
    assert args.customers == generator.DEFAULT_CUSTOMERS and args.db_prefix == "rsingh_spend_anl"
