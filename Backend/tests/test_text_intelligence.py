"""Text intelligence: category conflicts, format violations and spelling variants in non-numeric data."""
import sys

import numpy as np
import polars as pl
import pytest

from app.services.data_processing import clean_data
from app.services.row_intelligence import RowTriageConfig, triage_rows

PAIRS = [("Hyderabad", "Telangana"), ("Bengaluru", "Karnataka"), ("Chennai", "Tamil Nadu"),
         ("Rajahmundry", "Andhra Pradesh"), ("Mumbai", "Maharashtra")]
TITLES = {"Engineering": "Developer", "Sales": "Account Exec", "Support": "Agent"}


def _staff(seed=1, n=2000):
    rng = np.random.default_rng(5000 + seed)
    idx = rng.integers(0, 5, n)
    dept = rng.choice(list(TITLES), n)
    return pl.DataFrame({
        "city": [PAIRS[i][0] for i in idx],
        "state": [PAIRS[i][1] for i in idx],
        "dept": dept,
        "title": [TITLES[d] for d in dept],
        "phone": [f"9{x}" for x in rng.integers(100000000, 999999999, n)],
    })


def _set(df, col, row, value):
    vals = df[col].to_list()
    vals[row] = value
    return df.with_columns(pl.Series(col, vals))


@pytest.fixture
def dirty():
    df = _staff()
    for i in (10, 200, 400, 800, 1200):
        df = _set(df, "state", i, "Kerala")                                    # city/state contradiction
    for i in (50, 600, 1000):
        df = _set(df, "title", i, "Developer" if df["dept"][i] != "Engineering" else "Agent")
    df = _set(df, "city", 30, "Hyd")
    df = _set(df, "city", 31, "hyderabad")
    df = _set(df, "city", 32, "Hyderabad ")
    df = _set(df, "city", 700, "Chenai")
    df = _set(df, "phone", 90, "12345")
    df = _set(df, "phone", 91, "98-765-abc")
    return df


def _codes(result, row):
    return {f.label for f in result.flags if f.mask[row]}


def test_all_planted_text_problems_are_caught_with_the_right_reason(dirty):
    r = triage_rows(dirty)
    for row in (10, 200, 400, 800, 1200):
        assert "category_conflict:state" in _codes(r, row)
    for row in (50, 600, 1000):
        assert any(c.startswith("category_conflict:") for c in _codes(r, row))
    for row in (30, 31, 32, 700):
        assert _codes(r, row) == {"variant_value:city"}                        # one cell, one reason
    for row in (90, 91):
        assert _codes(r, row) == {"format_violation:phone"}


def test_variant_flags_carry_a_suggested_value(dirty):
    r = triage_rows(dirty)
    why = {int(i): w for i, w in zip(r.audit_frame()["row_id"], r.audit_frame()["why"])}
    assert "suggested value: Hyderabad" in why[31]
    assert "suggested value: Hyderabad" in why[30] and "abbreviation" in why[30]
    assert "suggested value: Chennai" in why[700]
    assert "usual format" in why[90]


def test_clean_structured_text_gets_no_flags():
    for seed in (1, 2, 3):
        r = triage_rows(_staff(seed))
        assert sum(f.count for f in r.flags if f.code in ("category_conflict", "format_violation", "variant_value")) == 0


def test_a_dependency_that_is_only_mostly_true_is_not_enforced():
    df = _staff(1)
    rng = np.random.default_rng(3)
    alt = list(TITLES.values())
    titles = [t if rng.random() > 0.10 else str(rng.choice(alt)) for t in df["title"].to_list()]
    r = triage_rows(df.with_columns(pl.Series("title", titles)))
    assert r.cell_mask("category_conflict").sum() == 0
    assert any("not_determined_by_other_columns" in v for v in r.learned["text"]["conflicts_skipped_columns"].values())


def test_legitimate_rare_values_that_are_internally_consistent_are_not_flagged():
    df = _staff(2)
    dept, title, city, state = (df[c].to_list() for c in ("dept", "title", "city", "state"))
    for i in range(0, 2000, 333):
        dept[i], title[i] = "HR", "Recruiter"
    for i in range(1, 2000, 400):
        city[i], state[i] = "Vizag", "Andhra Pradesh"
    df = pl.DataFrame({"city": city, "state": state, "dept": dept, "title": title, "phone": df["phone"]})
    r = triage_rows(df)
    assert r.cell_mask("category_conflict").sum() == 0
    assert r.cell_mask("format_violation").sum() == 0
    assert r.cell_mask("variant_value").sum() == 0


def test_common_values_are_not_excused_by_the_rare_value_rule(dirty):
    # regression: "Developer" is common, so a wrong Developer must still be caught
    r = triage_rows(dirty)
    assert sum(any(c.startswith("category_conflict:") for c in _codes(r, row)) for row in (50, 600, 1000)) == 3


def test_names_free_text_emails_and_mixed_date_formats_are_left_alone():
    n = 2000
    rng = np.random.default_rng(9)
    names = ["Ravi", "Anjali", "Mary Ann", "Sai Kiran", "O'Neil", "Priya", "Karthik", "Lakshmi Devi", "John", "Fatima"]
    words = ["the", "order", "was", "late", "and", "customer", "called", "about", "refund", "please", "check"]
    note = [" ".join(rng.choice(words, rng.integers(8, 20))) for _ in range(n)]
    d1 = [f"2025-{rng.integers(1, 13):02d}-{rng.integers(1, 28):02d}" for _ in range(n)]
    d2 = [f"{rng.integers(1, 28):02d}/{rng.integers(1, 13):02d}/2025" for _ in range(n)]
    df = pl.DataFrame({
        "name": rng.choice(names, n), "note": note,
        "email": [f"user{i}@example.com" for i in range(n)],               # variable-length numbering
        "joined": [a if rng.random() < 0.5 else b for a, b in zip(d1, d2)],
    })
    r = triage_rows(df)
    assert sum(f.count for f in r.flags if f.code in ("format_violation", "variant_value", "category_conflict")) == 0
    assert r.learned["text"]["skipped"]["note"] == "free_text"


def test_a_stray_minority_format_is_flagged_but_a_fifty_fifty_split_is_not():
    n = 2000
    rng = np.random.default_rng(11)
    d1 = [f"2025-{rng.integers(1, 13):02d}-{rng.integers(1, 28):02d}" for _ in range(n)]
    d2 = [f"{rng.integers(1, 28):02d}/{rng.integers(1, 13):02d}/2025" for _ in range(n)]
    stray = [a if rng.random() > 0.03 else b for a, b in zip(d1, d2)]
    r = triage_rows(pl.DataFrame({"joined": stray, "k": rng.integers(0, 50, n).astype(str)}))
    assert 30 <= r.cell_mask("format_violation", "joined").sum() <= 90


def test_low_cardinality_columns_are_never_format_checked():
    df = _staff(1)
    r = triage_rows(df)
    assert r.learned["text"]["skipped"]["dept"] == "low_cardinality_category"
    # a legitimate fourth department used once is therefore not a "format violation"
    r2 = triage_rows(_set(df, "dept", 5, "HR"))
    assert r2.cell_mask("format_violation", "dept").sum() == 0


def test_text_flags_never_change_or_drop_data(dirty):
    cleaned, report, _ = clean_data(dirty, None, None, "t")
    assert cleaned.height == dirty.height - report.duplicate_rows_removed
    assert report.row_triage["learned"]["text"]["ran"] is True
    for row, col, value in ((31, "city", "hyderabad"), (90, "phone", "12345"), (10, "state", "Kerala")):
        key = dirty["phone"][row]
        assert cleaned.filter(pl.col("phone") == key)[col].to_list()[0] == value


def test_text_layer_can_be_disabled(dirty):
    r = triage_rows(dirty, RowTriageConfig(text_enabled=False))
    assert "text" not in r.learned
    assert r.cell_mask("variant_value").sum() == r.cell_mask("format_violation").sum() == 0


def test_a_failing_technique_is_isolated_and_reported(dirty, monkeypatch):
    import app.services.text_intelligence as ti

    def boom(*a, **k):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(ti, "_variant_layer", boom)
    r = triage_rows(dirty)
    assert any("variants: RuntimeError" in e for e in r.learned["text"]["errors"])
    assert r.cell_mask("format_violation").sum() > 0                            # other techniques still ran


def test_without_scikit_learn_only_the_conflict_check_is_skipped(dirty, monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn", None)
    r = triage_rows(dirty)
    assert "scikit-learn" in r.learned["text"]["conflicts_skipped"]
    assert r.cell_mask("variant_value").sum() > 0 and r.cell_mask("format_violation").sum() > 0


def test_small_datasets_skip_the_text_layer_quietly():
    r = triage_rows(_staff(1, n=60))
    assert r.cell_mask("category_conflict").sum() == 0
    assert r.cell_mask("variant_value").sum() == 0


def test_text_layer_is_deterministic(dirty):
    a, b = triage_rows(dirty), triage_rows(dirty)
    for code in ("category_conflict", "variant_value", "format_violation"):
        assert (a.cell_mask(code) == b.cell_mask(code)).all()
