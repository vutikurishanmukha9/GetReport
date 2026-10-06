"""Row intelligence: which rows may the pipeline touch, and which must stay untouched."""
import random

import polars as pl
import pytest

from app.services.data_processing import clean_data
from app.services.row_intelligence import RowTriageConfig, RowVerdict, sparse_row_mask, triage_rows


def _base_rows(n=60, seed=1):
    random.seed(seed)
    return [
        {
            "order_id": f"O{i}",
            "customer": f"Cust {i % 9}",
            "region": random.choice(["N", "S", "E"]),
            "amount": float(100 + random.randint(0, 40)),
            "qty": random.randint(1, 5),
            "email": f"c{i}@x.com",
            "age": random.randint(20, 60),
        }
        for i in range(n)
    ]


@pytest.fixture
def messy_df():
    rows = _base_rows()
    rows[5]["amount"] = None                 # imputable null
    rows[6]["region"] = "N/A"                # masked placeholder
    rows[7]["amount"] = 99999.0              # legitimate-looking outlier
    rows[8]["email"] = "not-an-email"
    rows[9]["age"] = 150
    rows.insert(20, {k: None for k in rows[0]})                                  # blank row
    rows.insert(30, {k: k for k in rows[0]})                                     # repeated header
    rows.append({"order_id": "Total", "customer": None, "region": None, "amount": 6000.0,
                 "qty": 150, "email": None, "age": None})                        # total row
    rows.append({"order_id": "Source: ERP export generated 2026-09-01", "customer": None,
                 "region": None, "amount": None, "qty": None, "email": None, "age": None})  # footer
    rows.append(dict(rows[0]))                                                   # exact duplicate
    return pl.DataFrame(rows, strict=False)


def _verdicts(result):
    return {int(i): v for i, v in zip(result.audit_frame()["row_id"], result.audit_frame()["verdict"])}


def test_each_planted_row_gets_the_right_verdict(messy_df):
    r = triage_rows(messy_df)
    v = _verdicts(r)
    assert v[5] == "FIX" and v[6] == "FIX"
    assert v[7] == "REVIEW" and v[8] == "REVIEW" and v[9] == "REVIEW"
    assert v[20] == "EXCLUDE"   # blank
    assert v[30] == "EXCLUDE"   # repeated header
    assert v[62] == "EXCLUDE"   # total
    assert v[63] == "EXCLUDE"   # footer
    assert v[64] == "EXCLUDE"   # duplicate
    s = r.summary()
    assert s["verdicts"]["PROTECT"] == s["rows_scanned"] - len(v)


def test_every_row_has_exactly_one_verdict(messy_df):
    r = triage_rows(messy_df)
    assert sum(r.summary()["verdicts"].values()) == messy_df.height


def test_protect_rows_are_bit_identical_after_cleaning(messy_df):
    r = triage_rows(messy_df)
    before = messy_df.filter(pl.Series(r.protect_mask))     # by row position, not by key
    cleaned, report, _ = clean_data(messy_df, None, None, "t")
    # protected rows keep their order, so they appear in `cleaned` as the rows that were never flagged
    flagged_keys = set(messy_df.filter(pl.Series(~r.protect_mask))["order_id"].to_list())
    after = cleaned.filter(~pl.col("order_id").is_in(list(flagged_keys)))
    before = before.filter(~pl.col("order_id").is_in(list(flagged_keys)))
    assert before.height == after.height > 0
    for col in before.columns:
        assert before[col].to_list() == after[col].to_list(), f"protected rows changed in {col}"
    assert report.row_triage["verdicts"]["PROTECT"] == int(r.protect_mask.sum())


def test_non_data_rows_are_removed_and_review_rows_survive_unchanged(messy_df):
    cleaned, report, dag = clean_data(messy_df, None, None, "t")
    keys = set(cleaned["order_id"].to_list())
    assert "Total" not in keys and "order_id" not in keys
    assert not any(str(k).startswith("Source:") for k in keys)
    assert report.non_data_rows_dropped == 4        # blank, header, total, footer
    assert report.duplicate_rows_removed == 1       # still accounted by the dedupe step
    # outlier row is flagged for review, never altered
    row = cleaned.filter(pl.col("order_id") == "O7")
    assert float(row["amount"].item()) == 99999.0   # the mid-file header makes columns text; value is intact
    assert any(n.operation == "drop_non_data_rows" for n in dag.nodes.values()) if hasattr(dag, "nodes") else True


def test_fix_rows_change_only_the_flagged_cells(messy_df):
    cleaned, _, _ = clean_data(messy_df, None, None, "t")
    before = messy_df.filter(pl.col("order_id") == "O5").row(0, named=True)
    after = cleaned.filter(pl.col("order_id") == "O5").row(0, named=True)
    assert before["amount"] is None and after["amount"] is not None   # the flagged cell
    for col in before:
        if col != "amount":
            assert before[col] == after[col]


def test_sparse_row_is_not_imputed():
    rows = _base_rows(40)
    rows[10] = {**{k: None for k in rows[10]}, "order_id": "O10"}      # only the id is filled
    df = pl.DataFrame(rows, strict=False)
    assert sparse_row_mask(df)[10]
    cleaned, _, _ = clean_data(df, None, None, "t")
    row = cleaned.filter(pl.col("order_id") == "O10").row(0, named=True)
    assert row["amount"] is None and row["customer"] is None and row["age"] is None


def test_mostly_missing_column_is_never_imputed():
    rows = _base_rows(50)
    for i, r in enumerate(rows):
        if i % 2 == 0:
            r["amount"] = None                                         # 50% missing
    df = pl.DataFrame(rows, strict=False)
    r = triage_rows(df)
    assert "amount" in r.non_imputable_columns
    cleaned, report, _ = clean_data(df, None, None, "t")
    assert cleaned["amount"].null_count() == 25


def test_total_label_used_as_a_real_category_is_not_a_summary_row():
    rows = _base_rows(60)
    for i in range(0, 60, 3):
        rows[i]["customer"] = "Total Solutions Inc"                    # real company, many rows
    df = pl.DataFrame(rows, strict=False)
    r = triage_rows(df)
    assert r.cell_mask("summary_row").sum() == 0


def test_id_and_protected_columns_are_never_imputed_or_flagged():
    rows = _base_rows(40)
    rows[3]["qty"] = None
    df = pl.DataFrame(rows, strict=False)
    cfg = RowTriageConfig(protected_columns=frozenset({"qty"}))
    r = triage_rows(df, cfg)
    assert r.cell_mask("impute", "qty").sum() == 0
    assert r.cell_mask("impute", "order_id").sum() == 0


def test_clean_dataset_is_fully_protected():
    df = pl.DataFrame(_base_rows(50), strict=False)
    r = triage_rows(df)
    assert r.summary()["untouched_pct"] == 100.0
    cleaned, report, _ = clean_data(df, None, None, "t")
    assert cleaned.equals(df)
    assert report.non_data_rows_dropped == 0


def test_row_intelligence_can_be_switched_off(messy_df):
    cleaned, report, _ = clean_data(messy_df, None, None, "t", row_intelligence=False)
    assert report.row_triage == {}
    assert report.non_data_rows_dropped == 0


# ─── Learned layer: cross-column consistency ─────────────────────────────────
import sys

import numpy as np


def _people(n=3000, seed=1001, plant=True):
    rng = np.random.default_rng(seed)
    age = rng.integers(22, 60, n)
    exp = np.clip(age - 22 - rng.integers(0, 4, n), 0, None)
    dept = rng.choice(["eng", "sales", "ops"], n, p=[0.5, 0.3, 0.2])
    salary = 30000 + exp * 2500 + rng.normal(0, 2500, n) + np.where(dept == "eng", 8000, 0)
    df = pl.DataFrame({"emp_id": [f"E{i}" for i in range(n)], "age": age, "experience": exp, "salary": salary, "dept": dept})
    bad = []
    if plant:
        bad = sorted(rng.choice(n, 14, replace=False).tolist())
        a, b = bad[:7], bad[7:]
        pos = pl.int_range(pl.len())
        df = df.with_columns(
            pl.when(pos.is_in(a)).then(pl.lit(23)).otherwise(pl.col("age")).alias("age"),
            pl.when(pos.is_in(a)).then(pl.lit(20)).when(pos.is_in(b)).then(pl.lit(1)).otherwise(pl.col("experience")).alias("experience"),
            pl.when(pos.is_in(b)).then(pl.lit(95000.0)).otherwise(pl.col("salary")).alias("salary"),
        )
    return df, bad


def test_learned_layer_catches_combinations_no_single_column_rule_can_see():
    df, bad = _people()
    r = triage_rows(df, RowTriageConfig(learned_enabled=False))
    assert not (set(bad) & set(np.flatnonzero(r.severity > 0)))     # rules alone see nothing
    r = triage_rows(df)
    hit = set(np.flatnonzero(r.cell_mask("relationship_violation")))
    assert len(hit & set(bad)) >= 12                                # recall >= 12/14
    assert len(hit - set(bad)) <= 2                                 # almost no extras
    assert all(r.severity[p] == 1 for p in hit)                     # REVIEW only, never FIX or EXCLUDE


def test_learned_flags_explain_which_column_and_what_was_expected():
    df, bad = _people()
    r = triage_rows(df)
    audit = r.audit_frame().filter(pl.col("reasons").list.eval(pl.element().str.starts_with("relationship_violation")).list.any())
    assert audit.height >= 12 and audit["why"].null_count() == 0
    sample = audit["why"][0]
    print(audit["why"].to_list()[:3])
    assert "expected about" in sample and "given the rest of the row" in sample


def test_clean_structured_data_gets_no_learned_flags():
    for seed in (2001, 2002, 2003):
        df, _ = _people(seed=seed, plant=False)
        assert triage_rows(df).cell_mask("relationship_violation").sum() == 0


def test_bad_rows_do_not_teach_the_model_that_bad_is_normal():
    # planted rows share salary 95000; normal rows near that salary must not be flagged because of them
    df, bad = _people(seed=1001)
    r = triage_rows(df)
    hit = set(np.flatnonzero(r.cell_mask("relationship_violation")))
    near = set(df.with_row_index("i").filter((pl.col("salary") > 93000) & (pl.col("salary") < 97000))["i"].to_list()) - set(bad)
    assert not (hit & near)


def test_independent_columns_are_skipped_not_flagged():
    rng = np.random.default_rng(5)
    df = pl.DataFrame({"a": rng.normal(0, 1, 1000), "b": rng.normal(5, 2, 1000), "c": rng.integers(0, 100, 1000)})
    r = triage_rows(df)
    assert r.cell_mask("relationship_violation").sum() == 0
    assert set(r.learned["skipped_columns"]) == {"a", "b", "c"}


def test_small_datasets_skip_the_learned_layer_and_say_why():
    r = triage_rows(pl.DataFrame(_base_rows(100), strict=False))
    assert r.learned["ran"] is False and "too_few_rows" in r.learned["reason"]


def test_learned_layer_degrades_gracefully_without_scikit_learn(monkeypatch):
    df, _ = _people(n=600)
    monkeypatch.setitem(sys.modules, "sklearn", None)
    r = triage_rows(df)
    assert r.learned["ran"] is False and "scikit-learn" in r.learned["reason"]
    assert r.cell_mask("relationship_violation").sum() == 0           # rule layer still works


def test_learned_flags_never_change_or_drop_data():
    df, bad = _people(n=1500)
    cleaned, report, _ = clean_data(df, None, None, "t")
    assert cleaned.height == df.height                                  # nothing removed
    flagged_ids = df["emp_id"].filter(pl.Series(triage_rows(df).cell_mask("relationship_violation")))
    assert len(flagged_ids) > 0
    for col in ("age", "experience", "salary", "dept"):
        assert cleaned.filter(pl.col("emp_id").is_in(flagged_ids.implode()))[col].to_list() == df.filter(pl.col("emp_id").is_in(flagged_ids.implode()))[col].to_list()
    assert report.row_triage["learned"]["ran"] is True


def test_learned_layer_is_deterministic():
    df, _ = _people(n=1500)
    a = triage_rows(df).cell_mask("relationship_violation")
    b = triage_rows(df).cell_mask("relationship_violation")
    assert (a == b).all()


def test_learned_layer_can_be_disabled():
    df, _ = _people(n=1500)
    r = triage_rows(df, RowTriageConfig(learned_enabled=False))
    assert r.learned["ran"] is False and r.cell_mask("relationship_violation").sum() == 0


def test_missing_predictors_do_not_cause_a_flood_of_false_flags():
    rng = np.random.default_rng(7)
    n = 3000
    x = rng.normal(50, 10, n); y = 2 * x + rng.normal(0, 3, n); w = x + rng.normal(0, 2, n)
    xm, wm = x.copy(), w.copy()
    xm[rng.random(n) < 0.15] = np.nan
    wm[rng.random(n) < 0.15] = np.nan
    bad = rng.choice(n, 10, replace=False)
    y[bad] += 60
    r = triage_rows(pl.DataFrame({"x": xm, "y": y, "w": wm}))
    hit = set(np.flatnonzero(r.cell_mask("relationship_violation")))
    assert len(hit & set(bad.tolist())) >= 8
    assert len(hit - set(bad.tolist())) <= 6
