"""
GetReport report-output regression suite.

Each test = one finding from the review of getreport_messy_master_100k_Report.pdf.
Ground truth comes from the generator's anomaly_manifest.json and the source CSV, so a test
fails when GetReport's PDF disagrees with what we KNOW was injected.

Test groups
  GT_*   ground-truth self-checks: prove the dataset still contains what the manifest claims
         (must always pass; if they fail, regenerate the data, not the product).
  CTL_*  positive controls: things GetReport already does right (guard against regressions).
  R##    regression tests for defects / missing features. These FAIL today and should turn green as fixed.

Run:
  pip install pytest pandas numpy
  GETREPORT_PDF=/path/to/Report.pdf pytest test_report_regression.py -v
  pytest -k "GT_ or CTL_"      # only the always-green tests
Requires poppler's `pdftotext` on PATH.
"""
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATA_DIR = Path(os.environ.get("GETREPORT_DATA_DIR", "/mnt/user-data/outputs"))
CSV = Path(os.environ.get("GETREPORT_CSV", DATA_DIR / "getreport_messy_master_100k.csv"))
MANIFEST = Path(os.environ.get("GETREPORT_MANIFEST", DATA_DIR / "anomaly_manifest.json"))
PDF = Path(os.environ.get("GETREPORT_PDF", "/mnt/user-data/uploads/getreport_messy_master_100k_Report.pdf"))

MAX_ACCEPTABLE_GRADE = os.environ.get("GETREPORT_MAX_GRADE", "C")  # best grade a dataset this dirty may receive
BENFORD_CRIT_001 = 20.09
NULL_TOKENS = {"", "N/A", "n/a", "NA", "NULL", "null", "None", "-", "--", "?", "-999", "-9999", "#NULL!", "#N/A", "nan",
               "NaN", "missing", "unknown", "TBD", "(blank)", "<null>", "nil", "undefined", "#REF!", "#DIV/0!"}
NON_DATA_ROW_KEYS = ["fully_blank_rows", "whitespace_only_rows", "repeated_mid_file_header_rows",
                     "subtotal_summary_rows", "grand_total_rows", "sparse_footer_rows"]


# ----------------------------------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def manifest():
    if not MANIFEST.exists():
        pytest.skip(f"manifest file not found: {MANIFEST}")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def df():
    if not CSV.exists():
        pytest.skip(f"CSV file not found: {CSV}")
    return pd.read_csv(CSV, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def _pdftotext(*flags):
    if not PDF.exists():
        pytest.skip(f"report PDF not found: {PDF}")
    import shutil
    if shutil.which("pdftotext"):
        return subprocess.run(["pdftotext", *flags, str(PDF), "-"], capture_output=True, text=True, check=True).stdout
    try:
        import pymupdf
        doc = pymupdf.open(str(PDF))
        if "-layout" in flags:
            all_pages = []
            for page in doc:
                words = page.get_text("words")
                words.sort(key=lambda w: (round(w[1] / 3) * 3, w[0]))
                lines = []
                current_y = None
                current_line = []
                for w in words:
                    y_bucket = round(w[1] / 3) * 3
                    if current_y is None or abs(y_bucket - current_y) <= 3:
                        current_y = y_bucket
                        current_line.append(w)
                    else:
                        line_str = ""
                        for i, word in enumerate(current_line):
                            if i == 0:
                                indent = " " * max(4, int(word[0] / 8))
                                line_str = indent + word[4]
                            else:
                                prev = current_line[i - 1]
                                gap = word[0] - prev[2]
                                sep = "   " if gap > 12 else " "
                                line_str += sep + word[4]
                        lines.append(line_str)
                        current_y = y_bucket
                        current_line = [w]
                if current_line:
                    line_str = ""
                    for i, word in enumerate(current_line):
                        if i == 0:
                            indent = " " * max(4, int(word[0] / 8))
                            line_str = indent + word[4]
                        else:
                            prev = current_line[i - 1]
                            gap = word[0] - prev[2]
                            sep = "   " if gap > 12 else " "
                            line_str += sep + word[4]
                    lines.append(line_str)
                all_pages.append("\n".join(lines))
            return "\n\n".join(all_pages)
        return "\n\n".join(page.get_text("text") for page in doc)
    except Exception as e:
        pytest.skip(f"Could not extract PDF text (pdftotext/pymupdf error: {e})")


@pytest.fixture(scope="session")
def text():
    """Layout-preserving text: table rows stay on one line."""
    return _pdftotext("-layout")


@pytest.fixture(scope="session")
def raw():
    """Reading-order text: wrapped table cells stay contiguous."""
    return _pdftotext()


def num(s):
    return pd.to_numeric(s.str.replace(r"[^0-9.eE+\-]", "", regex=True), errors="coerce")


def luhn_ok(digits: str) -> bool:
    d = [int(c) for c in reversed(digits)]
    return (sum(d[0::2]) + sum(sum(divmod(x * 2, 10)) for x in d[1::2])) % 10 == 0


def first_digits(series):
    x = num(series).abs()
    x = x[(x > 0) & np.isfinite(x)]
    fd = x.map(lambda v: f"{v:.10e}"[0]).astype(int)
    return fd


def benford_chi2(series):
    fd = first_digits(series)
    obs = fd.value_counts().reindex(range(1, 10), fill_value=0).values
    exp = np.log10(1 + 1 / np.arange(1, 10)) * obs.sum()
    return float(((obs - exp) ** 2 / exp).sum())


def expected_clean_rows(m):
    master = next(iter(m["files"].values()))
    # near-duplicates are NOT exact duplicates and must survive; all-null-token rows must not.
    return master["data_rows_requested"] + m["structural_anomalies_master"]["near_duplicate_rows"]


def parse_total_rows(text):
    mm = re.search(r"Total Rows\s+([\d,]+)", text)
    assert mm, "report has no 'Total Rows' line"
    return int(mm.group(1).replace(",", ""))


def grade_table(text):
    rows = re.findall(r"^\s+([A-Za-z_][\w]*)\s+([A-F])\s+(\d+)%\s+(\d+)%\s+(\d+)%\s+(\d+)%\s*$", text, re.M)
    return {r[0]: dict(grade=r[1], completeness=int(r[2]), consistency=int(r[3]), validity=int(r[4]),
                       stability=int(r[5])) for r in rows}


# ----------------------------------------------------------------------------------------------
# GT_  ground-truth self-checks (always green)
# ----------------------------------------------------------------------------------------------
def test_GT01_shape_matches_manifest(df, manifest):
    master = next(iter(manifest["files"].values()))
    assert len(df) == master["rows_including_structural_junk"]
    assert df.shape[1] == master["columns"] == 44


def test_GT02_exact_duplicates_present(df, manifest):
    nonblank = (df.apply(lambda c: c.str.strip() != "")).any(axis=1)
    dups = int(df[nonblank].duplicated().sum())
    n = manifest["structural_anomalies_master"]["exact_duplicate_rows"]
    # +/- a handful of repeated mid-file header rows that are identical to one another
    assert n <= dups <= n + manifest["structural_anomalies_master"]["repeated_mid_file_header_rows"]


def test_GT03_benford_signal(df):
    assert benford_chi2(df["unit_price"]) < BENFORD_CRIT_001, "natural column should conform"
    assert benford_chi2(df["bonus_payout"]) > BENFORD_CRIT_001 * 100, "fabricated column should fail hard"


def test_GT04_weight_perfect_collinearity(df):
    kg, lb = num(df["weight_kg"]), num(df["weight_lb"])
    ok = (kg > 0) & (lb > 0) & (lb < 500)
    assert np.corrcoef(kg[ok], lb[ok])[0, 1] > 0.9999


def test_GT05_total_equals_subtotal_plus_tax(df, manifest):
    plain = r"-?\d+\.\d{2}"  # ignore the ~10% rows deliberately written as $1,234.56 / 1.234,56 / (12.00) / 1e3
    mask = df["Subtotal"].str.fullmatch(plain) & df["Tax"].str.fullmatch(plain) & df["Total"].str.fullmatch(plain)
    sub, tax, tot = (df.loc[mask, c].astype(float) for c in ("Subtotal", "Tax", "Total"))
    good = (np.abs(sub + tax - tot) < 0.02).mean()
    assert mask.sum() > 50_000 and good > 0.99  # remainder = injected Total:violates_subtotal_plus_tax


def test_GT06_manifest_expected_row_count_reconciles(manifest):
    s = manifest["structural_anomalies_master"]
    master = next(iter(manifest["files"].values()))
    total = master["data_rows_requested"] + s["exact_duplicate_rows"] + s["near_duplicate_rows"] + s["all_null_token_rows"] \
        + sum(s[k] for k in NON_DATA_ROW_KEYS)
    assert total == master["rows_including_structural_junk"]


# ----------------------------------------------------------------------------------------------
# CTL_  positive controls (GetReport already gets these right)
# ----------------------------------------------------------------------------------------------
def test_CTL01_exact_duplicates_removed_count(text, manifest):
    mm = re.search(r"Duplicate Rows Removed\s+([\d,]+)", text)
    assert mm, "no 'Duplicate Rows Removed' line"
    n = manifest["structural_anomalies_master"]["exact_duplicate_rows"]
    assert abs(int(mm.group(1).replace(",", "")) - n) <= 12


def test_CTL02_headers_normalized(text):
    assert "column_44" in text and "notes_1" in text  # empty header + duplicate header resolved
    assert "e_mail" in text and "row_id" in text


def test_CTL03_impossible_ages_detected(text):
    assert re.search(r"age values out of human range", text)


# ----------------------------------------------------------------------------------------------
# R01-R03  Privacy / secret leakage  (CRITICAL)
# ----------------------------------------------------------------------------------------------
def test_R01_no_ssn_values_printed(text, df):
    leaked = set(re.findall(r"\b\d{3}-\d{2}-\d{4}\b", text)) & set(df["ssn"])
    assert not leaked, f"SSNs printed in report: {sorted(leaked)[:5]}"


def test_R02_no_api_keys_or_secrets_printed(text):
    pat = r"(sk_live_[A-Za-z0-9]{8,}|AKIA[A-Z2-7]{12,}|ghp_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_\-]{20,})"
    assert not re.findall(pat, text), "secret-pattern strings printed in report"


def test_R03_no_real_card_numbers_printed(text, df):
    cards = set(df["credit_card"].str.replace(r"\D", "", regex=True)) - {""}
    found = {m for m in re.findall(r"\b(\d{15,16})(?:\.\d+)?\b", text) if luhn_ok(m) and m in cards}
    assert not found, f"{len(found)} card numbers printed (descriptive stats on a PAN column?)"


def test_R04_pii_findings_section_exists(text):
    assert re.search(r"\b(PII|personally identifiable|sensitive data)\b", text, re.I), "no PII findings section"
    for kind in ("SSN", "credit card", "API key"):
        assert re.search(kind, text, re.I)


# ----------------------------------------------------------------------------------------------
# R05-R08  Sentinel / null-token handling
# ----------------------------------------------------------------------------------------------
def test_R05_no_sentinel_minimums(text):
    hits = re.findall(r"^\s*Min\s+-(?:9999|999)(?:\.0+)?\s*$", text, re.M)
    assert not hits, f"{len(hits)} descriptive-stat tables still contain a -9999/-999 sentinel as Min"


def test_R06_null_tokens_not_reported_as_categories(text):
    pat = r"^\s*(<null>|#null!|--|-999|-9999|\(blank\)|tbd|unknown|nil|undefined)\s+\d+\s+[\d.]+%\s*$"
    hits = re.findall(pat, text, re.M | re.I)
    assert not hits, f"{len(hits)} null-token spellings counted as real categories"


def test_R07_all_null_token_rows_excluded(text, manifest):
    assert parse_total_rows(text) <= expected_clean_rows(manifest), \
        "rows consisting only of null tokens must be EXCLUDEd before profiling"


def test_R08_row_triage_disclosed(text):
    for word in ("blank", "header", "subtotal|total row", "footer"):
        assert re.search(word, text, re.I), f"cleaning summary does not disclose removal of '{word}' rows"


# ----------------------------------------------------------------------------------------------
# R09-R11  Scoring honesty
# ----------------------------------------------------------------------------------------------
def test_R09_overall_grade_reflects_dirty_data(text):
    mm = re.search(r"Overall Data Quality: Grade ([A-F])", text)
    assert mm
    assert mm.group(1) >= MAX_ACCEPTABLE_GRADE, f"dataset graded {mm.group(1)}; expected {MAX_ACCEPTABLE_GRADE} or worse"


def test_R10_summary_matches_column_table(text):
    mm = re.search(r"(\d+) high-confidence columns", text)
    assert mm
    grades = grade_table(text)
    a_cols = sum(1 for g in grades.values() if g["grade"] == "A")
    assert int(mm.group(1)) <= a_cols, f"summary claims {mm.group(1)} high-confidence columns but only {a_cols} are graded A"


def test_R11_consistency_scores_calibrated(text, manifest):
    grades = grade_table(text)
    assert "country" in grades
    cell = manifest["cell_level_anomalies_master"]
    n = next(iter(manifest["files"].values()))["data_rows_requested"]
    bad = sum(v for k, v in cell.items() if k.startswith("Country:") and not k.endswith("null_token")) / n
    assert grades["country"]["consistency"] <= round(100 * (1 - bad)) + 5, \
        f"country consistency {grades['country']['consistency']}% but ~{bad:.0%} of cells are variants"


def test_R12_column_grade_table_complete(text, manifest):
    n = next(iter(manifest["files"].values()))["columns"]
    assert len(grade_table(text)) == n, f"grade table lists {len(grade_table(text))} of {n} columns"


# ----------------------------------------------------------------------------------------------
# R13-R15  Numeric robustness & rendering
# ----------------------------------------------------------------------------------------------
def test_R13_no_nonfinite_stat_values(text):
    assert not re.findall(r"^\s*(?:Mean|Std|Skewness|Kurtosis|Min|Max)\s+(?:nan|-?inf)\s*$", text, re.M | re.I), \
        "nan/inf printed as statistics: treat IEEE-754 specials as null"


def test_R14_no_absurd_digit_runs(text):
    assert not re.search(r"\d{30,}", text), "300-digit number printed; use scientific notation or clip"


def test_R15_no_missing_glyph_boxes(text):
    assert "\u25a0" not in text, "font fallback missing: non-Latin names and 'Tōkyō' rendered as ■"


# ----------------------------------------------------------------------------------------------
# R16-R19  Type inference: identifiers are not numbers
# ----------------------------------------------------------------------------------------------
def test_R16_zip_not_reported_missing_from_type_coercion(text, df):
    z = df["zip_code"].str.strip()
    true_missing = 100 * z.isin(NULL_TOKENS).mean()
    mm = re.search(r"^\s*zip_code\s+[A-F]\s+(\d+)%", text, re.M)
    assert mm, "zip_code absent from grade table"
    reported_missing = 100 - int(mm.group(1))
    assert reported_missing <= true_missing + 3, \
        f"zip_code reported {reported_missing}% missing but only {true_missing:.1f}% are null tokens (alphanumeric postcodes coerced away)"


def test_R17_zip_not_given_numeric_outlier_treatment(text):
    assert not re.search(r"zip_code\s+[\d,]+ outliers", text), "zip_code given IQR outlier/clip treatment"
    assert not re.search(r"^\s*zip_code\b.*Fill with median", text, re.M), "zip_code median-imputed"


def test_R18_identifier_like_columns_not_numeric_stats(text):
    # credit_card (PAN), zip_code and ssn are identifiers, not measures
    assert not re.search(r"^\s*credit_card\s+[\d,]+\s+[\d.]+%\s+[\d.e+]+\s+[\d.e+]+\s*$", text, re.M), \
        "credit_card appears in the numeric outlier table"
    assert not re.search(r"^\s*Mean\s+4\d{15}", text, re.M), "credit_card has descriptive statistics"


def test_R19_sensitive_columns_not_typed_or_encoded_as_features(text):
    assert not re.search(r"ssn\s+String\s+phone", text), "ssn mistyped as phone"
    assert not re.search(r"^\s*ssn\s+Hash", text, re.M), "ssn recommended as an ML feature"
    assert not re.search(r"^\s*credit_card\s+Robust", text, re.M), "credit_card recommended for numeric scaling"


# ----------------------------------------------------------------------------------------------
# R20-R23  Missing analyses (forensics, relationships, time series, target)
# ----------------------------------------------------------------------------------------------
def test_R20_benford_audit_reports_bonus_payout(text):
    i = text.lower().find("benford")
    assert i >= 0, "no Benford's-law section"
    assert "bonus_payout" in text[i:i + 2500], "fabricated-looking bonus_payout not flagged"


def test_R21_weight_collinearity_detected(text):
    pat = r"weight_kg.{0,200}weight_lb|weight_lb.{0,200}weight_kg"
    hit = re.search(pat, text, re.S | re.I)
    assert hit and re.search(r"collinear|redundan|r\s*=\s*1\.0|1\.00|VIF", hit.group(0), re.I), \
        "weight_kg/weight_lb perfect collinearity not reported"


def test_R22_symbolic_equation_total_equals_subtotal_plus_tax(text):
    assert re.search(r"total\s*=\s*subtotal\s*\+\s*tax|subtotal\s*\+\s*tax\s*=\s*total", text, re.I), \
        "hidden accounting equation not discovered"


def test_R23_time_series_not_skipped_and_trend_reported(text):
    assert not re.search(r"Time-Series Analysis\s+SKIPPED", text), "order_date column not recognised (mixed formats)"
    assert re.search(r"Mann-Kendall|Sen'?s slope|STL", text, re.I)


def test_R24_churn_target_detected_and_not_clipped(text):
    assert not re.search(r"churned\s+[\d,]+ outliers", text), "binary target flagged as outliers (clip to [0,0])"
    assert re.search(r"AutoML|Key Driver|target.*churned|churned.*target", text, re.I), "no supervised section for churned"


def test_R25_near_duplicates_reported(text, manifest):
    assert re.search(r"near[- ]duplicate|fuzzy duplicate|similar rows", text, re.I), \
        f"{manifest['structural_anomalies_master']['near_duplicate_rows']} near-duplicate rows not surfaced"


def test_R26_mixed_date_formats_flagged(text):
    assert re.search(r"mixed[- ](?:date[- ])?format|multiple date formats|inconsistent date", text, re.I), \
        "10+ date formats injected; none flagged"


# ----------------------------------------------------------------------------------------------
# R27-R30  Categories, constants, column accounting
# ----------------------------------------------------------------------------------------------
def test_R27_category_variants_consolidated(text):
    limits = {"product_category": 20, "status": 12, "is_active": 8}  # truth: 8 / 5 / 2 plus a few junk/null values
    found = dict((a, int(b)) for a, b in re.findall(r"^\s*(\w+)\s+.*?cardinality \((\d+) categories\)", text, re.M))
    for col, lim in limits.items():
        assert col in found, f"{col} not in encoding table"
        assert found[col] <= lim, f"{col}: {found[col]} categories (variants/typos not merged; expected <= {lim})"


def test_R28_constant_columns_flagged_not_charted(text):
    assert re.search(r"constant column|zero[- ]variance|near[- ]constant", text, re.I), "constant_col / zero_var_num never flagged"
    assert "Bar Chart: constant_col" not in text, "constant column charted as a dimension"


def test_R29_every_input_column_accounted_for(text, manifest):
    mm = re.search(r"Total Columns\s+(\d+)", text)
    assert mm and int(mm.group(1)) == next(iter(manifest["files"].values()))["columns"], "column count mismatch"
    body = re.sub(r"Columns Renamed.*?Issue Ledger", "", text, flags=re.S).lower()
    missing = [c for c in ("subtotal", "latitude", "longitude", "zero_var_num") if c not in body]
    assert not missing, f"columns never analysed or explicitly excluded: {missing}"


def test_R30_geo_validity_reported(text):
    assert re.search(r"(latitude|longitude)[^\n]*(out of range|swap|invalid|null island)", text, re.I), \
        "swapped / null-island / out-of-range coordinates not reported"


# ----------------------------------------------------------------------------------------------
# R31-R35  Recommendations & claims must be sane
# ----------------------------------------------------------------------------------------------
def test_R31_no_imputation_suggested_for_sparse_or_identifier_columns(text):
    bad = re.findall(r"^\s*(credit_card|api_key|column_44|ssn)\b.*Fill with", text, re.M)
    assert not bad, f"imputation suggested for {sorted(set(bad))} (>50% missing / sensitive)"


def test_R32_no_nonsensical_recommended_pairs(text):
    assert not re.search(r"(credit_card|ssn|api_key)\s*↔|↔\s*(credit_card|ssn|api_key)", text), \
        "recommended analysis pairs a sensitive column with identifiers"
    assert not re.search(r"->\s*row_id\s*\(foreign_key\)", text), "false foreign-key relationships"


def test_R33_key_conflicts_not_raised_on_legit_columns(raw):
    flagged = set(re.findall(r"duplicate IDs in\s*'\s*(\w+)", raw))
    assert not (flagged & {"zip_code", "customer_id"}), f"false primary-key conflicts on {flagged & {'zip_code', 'customer_id'}}"


def test_R34_improvement_label_matches_direction(text):
    mm = re.search(r"Completeness Score\s+([\d.]+)%\s+([\d.]+)%", text)
    assert mm
    before, after = float(mm.group(1)), float(mm.group(2))
    if after < before:
        assert "Top Recovered Columns" not in text, "completeness fell yet report advertises 'recovered' columns"


def test_R35_cluster_sample_is_representative(text):
    rows = parse_total_rows(text)
    mm = re.search(r"\(([\d,]+) records\)", text)
    if mm:
        assert int(mm.group(1).replace(",", "")) >= 0.5 * rows, "clustering ran on a tiny complete-case subset"


# ----------------------------------------------------------------------------------------------
# R36-R37  Security findings about the DATA + report hygiene
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("label,pattern", [
    ("formula injection", r"formula injection|CWE-1236"),
    ("XSS / script payloads", r"XSS|script injection|<script"),
    ("SQL injection strings", r"SQL injection|sqli"),
    ("prompt-injection phrases", r"prompt injection"),
    ("mojibake / encoding damage", r"mojibake|encoding (damage|issue|error)"),
])
def test_R36_text_hazards_reported(text, manifest, label, pattern):
    assert re.search(pattern, text, re.I), f"report never mentions {label}; manifest injected many cells"


def test_R37_no_truncated_table_cells(text):
    assert not re.search(r"(Analys|\()\s+(RAN|SKIPPED)\b", text), "methodology table cells truncated"
    assert not re.search(r"\d+ cate\s*$", text, re.M), "'cate' truncation in methodology table"


def test_R38_counts_reconcile_across_sections(text):
    cats = {int(x) for x in re.findall(r"(\d+) categorical", text)}
    assert len(cats) <= 1, f"categorical column count differs by section: {sorted(cats)} (explain subsets or reconcile)"
