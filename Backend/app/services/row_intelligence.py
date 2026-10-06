"""
Row Intelligence
================
Decides, for every row of an uploaded dataset, whether the pipeline may touch it.

Each row gets exactly one verdict:

    PROTECT  passes every check, must stay bit-identical through cleaning
    REVIEW   suspicious but possibly legitimate (outliers, rule violations); flag, never change
    FIX      has a deterministic, safe automatic fix (masked null placeholder, imputable null)
    EXCLUDE  not real data (blank row, repeated header, total row, footer note, exact duplicate)

Design rules:
    1. Nothing is mutated here. This module only produces decisions and reasons.
    2. Structural checks run first, so junk rows cannot distort column statistics.
    3. Statistics (null rates, quartiles) are computed on valid rows only.
    4. When unsure, the verdict is REVIEW, never FIX or EXCLUDE.
    5. ID-like and user-protected columns are never imputed or outlier-flagged.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

ROW_ID = "__row_id__"


class RowVerdict(str, Enum):
    PROTECT = "PROTECT"
    REVIEW = "REVIEW"
    FIX = "FIX"
    EXCLUDE = "EXCLUDE"


_SEVERITY: dict[RowVerdict, int] = {
    RowVerdict.PROTECT: 0,
    RowVerdict.REVIEW: 1,
    RowVerdict.FIX: 2,
    RowVerdict.EXCLUDE: 3,
}
_BY_SEVERITY: dict[int, RowVerdict] = {v: k for k, v in _SEVERITY.items()}

_ID_TOKENS = frozenset({"id", "uuid", "guid", "code", "sku", "zip", "zipcode", "pin", "phone", "mobile", "key"})
_NUMERIC_JUNK = r"[,\s$\u20ac\u00a3\u20b9%]"
_SUMMARY_LABEL = re.compile(
    r"^\s*(grand\s+total|sub\s*-?\s*total|total|sum|average|avg|mean|median|summary|count|overall)\b",
    re.IGNORECASE,
)
_NOTE_LABEL = re.compile(
    r"^\s*(source|sources|note|notes|generated|printed|report|page|confidential|disclaimer)\b",
    re.IGNORECASE,
)


# ─── Config ──────────────────────────────────────────────────────────────────
@dataclass
class RowTriageConfig:
    protected_columns: frozenset[str] = frozenset()
    placeholders: frozenset[str] = frozenset({"n/a", "na", "-999", "null", "?", "-", "missing", "unknown", "none"})
    sparse_fill_ratio: float = 0.10       # row with <= 10% of cells filled is "sparse"
    sparse_min_columns: int = 5
    footer_scan_rows: int = 5
    type_majority: float = 0.90           # share of values that must parse for a column to count as numeric
    max_impute_null_rate: float = 0.30    # never impute a column that is missing more than this
    outlier_iqr_k: float = 3.0
    min_rows_for_stats: int = 20
    summary_column_scan: int = 5
    summary_max_share: float = 0.05       # a "Total" label on >5% of rows is a category, not a total row
    dedupe_exact: bool = True
    # learned layer: each numeric column is predicted from the others (self-supervised, per dataset)
    learned_enabled: bool = True
    learned_min_rows: int = 300
    learned_z_threshold: float = 6.0      # robust residual z-score needed to flag a cell
    learned_min_r2: float = 0.30          # a column is only judged if the others explain at least this much of it
    learned_folds: int = 3
    learned_fit_sample: int = 50_000
    learned_max_targets: int = 15
    learned_max_predictors: int = 30
    learned_max_categories: int = 50
    learned_max_flag_share: float = 0.01  # if more than 1% of a column trips the test, keep only the worst 1%
    learned_tail_multiple: float = 2.5    # threshold is also at least this multiple of the column's own 99th percentile z
    learned_min_group: int = 50           # smallest missing-data pattern that gets its own error scale
    # text layer (see text_intelligence.py)
    text_enabled: bool = True
    text_min_rows: int = 100
    text_max_classes: int = 20            # categorical targets with more classes are not modelled
    text_max_targets: int = 10
    text_min_accuracy: float = 0.97       # a dependency must hold for ~all rows before violations are flagged
    text_min_lift: float = 0.15           # and must beat always guessing the most common value by this much
    text_p_low: float = 0.05              # actual value must be this unlikely given the rest of the row
    text_confident: float = 0.90          # while the model is at least this sure of another value
    text_shape_coverage: float = 0.95     # established formats must cover this share of a column
    text_shape_max_established: int = 3   # more than this many formats means the column has no fixed format
    text_shape_established: float = 0.05  # a format used by fewer than this share of values is a stray format
    text_fixed_width_share: float = 0.90  # if one exact shape holds this share, the column is fixed width
    text_coherent_group: float = 0.80     # rare values whose rows all point to the same context are real categories
    text_rare_class_share: float = 0.02   # a value used by fewer than this share of rows counts as rare
    text_free_text_len: float = 60.0      # columns with longer average text are free text and never judged
    text_max_distinct: int = 5000
    text_variant_similarity: float = 0.84
    random_state: int = 0


# ─── Flag / Result ───────────────────────────────────────────────────────────
@dataclass
class RowFlag:
    verdict: RowVerdict
    code: str
    column: str | None
    mask: np.ndarray  # bool, length n_rows

    @property
    def label(self) -> str:
        return f"{self.code}:{self.column}" if self.column else self.code

    @property
    def count(self) -> int:
        return int(self.mask.sum())


@dataclass
class RowTriageResult:
    row_ids: np.ndarray
    flags: list[RowFlag]
    severity: np.ndarray                      # int8 per row
    non_imputable_columns: dict[str, float]   # column -> null rate among valid rows
    column_roles: dict[str, str]              # column -> "id" | "protected" | "numeric" | "text" | "other"
    explanations: dict[int, list[dict[str, Any]]] = field(default_factory=dict)  # row position -> drivers
    learned: dict[str, Any] = field(default_factory=lambda: {"ran": False, "reason": "not_run"})

    @property
    def n_rows(self) -> int:
        return int(self.row_ids.shape[0])

    # ---- masks -------------------------------------------------------------
    def mask(self, verdict: RowVerdict) -> np.ndarray:
        return self.severity == _SEVERITY[verdict]

    @property
    def exclude_mask(self) -> np.ndarray:
        return self.mask(RowVerdict.EXCLUDE)

    def exclude_mask_without(self, *codes: str) -> np.ndarray:
        """Rows excluded for any reason other than ``codes`` (used to leave dedupe to its own step)."""
        out = np.zeros(self.n_rows, dtype=bool)
        for f in self.flags:
            if f.verdict is RowVerdict.EXCLUDE and f.code not in codes:
                out |= f.mask
        return out

    @property
    def protect_mask(self) -> np.ndarray:
        return self.mask(RowVerdict.PROTECT)

    def cell_mask(self, code: str, column: str | None = None) -> np.ndarray:
        out = np.zeros(self.n_rows, dtype=bool)
        for f in self.flags:
            if f.code == code and (column is None or f.column == column):
                out |= f.mask
        return out

    def ids_with(self, code: str) -> np.ndarray:
        return self.row_ids[self.cell_mask(code)]

    # ---- reporting ---------------------------------------------------------
    def reasons_by_row(self) -> dict[int, list[str]]:
        reasons: dict[int, list[str]] = {}
        for f in self.flags:
            for pos in np.flatnonzero(f.mask):
                reasons.setdefault(int(pos), []).append(f.label)
        return reasons

    def audit_frame(self, include_protected: bool = False) -> pl.DataFrame:
        reasons = self.reasons_by_row()
        positions = np.arange(self.n_rows) if include_protected else np.array(sorted(reasons), dtype=int)
        return pl.DataFrame(
            {
                "row_id": [int(self.row_ids[p]) for p in positions],
                "verdict": [_BY_SEVERITY[int(self.severity[p])].value for p in positions],
                "reasons": [reasons.get(int(p), []) for p in positions],
                "why": [self.explain(int(p)) for p in positions],
            },
            schema={"row_id": pl.Int64, "verdict": pl.Utf8, "reasons": pl.List(pl.Utf8), "why": pl.Utf8},
        )

    def explain(self, position: int) -> str | None:
        drivers = self.explanations.get(position)
        if not drivers:
            return None
        parts = [
            f"{d['column']}={d['value']} ({d['note']})" if "note" in d
            else f"{d['column']}={d['value']} (expected about {d['expected']} given the rest of the row)"
            for d in drivers
        ]
        return "; ".join(parts)

    def summary(self) -> dict[str, Any]:
        n = self.n_rows
        counts = {v.value: int((self.severity == s).sum()) for v, s in _SEVERITY.items()}
        by_reason: dict[str, int] = {}
        for f in self.flags:
            by_reason[f.code] = by_reason.get(f.code, 0) + f.count
        return {
            "rows_scanned": n,
            "verdicts": counts,
            "untouched_pct": round(counts["PROTECT"] / n * 100, 2) if n else 100.0,
            "by_reason": dict(sorted(by_reason.items(), key=lambda kv: -kv[1])),
            "non_imputable_columns": {k: round(v, 4) for k, v in self.non_imputable_columns.items()},
            "learned": self.learned,
        }

    def to_dict(self, max_examples: int = 25) -> dict[str, Any]:
        audit = self.audit_frame()
        examples = audit.head(max_examples).to_dicts()
        out = self.summary()
        out["examples"] = examples
        return out


# ─── Engine ──────────────────────────────────────────────────────────────────
def _tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", name.lower()) if t}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _is_id_column(name: str) -> bool:
    return bool(_tokens(name) & _ID_TOKENS)


def _is_text(dtype: pl.DataType) -> bool:
    return dtype in (pl.Utf8, pl.String)


def _blank_expr(col: str, dtype: pl.DataType) -> pl.Expr:
    c = pl.col(col)
    if _is_text(dtype):
        return c.is_null() | (c.str.strip_chars() == "")
    if dtype in (pl.Float32, pl.Float64):
        return c.is_null() | c.is_nan()
    return c.is_null()


# ─── Learned layer ───────────────────────────────────────────────────────────
def _model_matrix(
    data: pl.DataFrame, roles: dict[str, str], valid: np.ndarray, cfg: RowTriageConfig
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Returns (numeric_features, categorical_codes). NaN marks missing; the model handles NaN natively."""
    numeric: dict[str, np.ndarray] = {}
    categorical: dict[str, np.ndarray] = {}
    for c in data.columns:
        role = roles.get(c)
        s = data[c]
        if role in ("id", "protected"):
            continue
        if role == "numeric":
            numeric[c] = s.cast(pl.Float64, strict=False).to_numpy().astype(float)
        elif role == "numeric_text":
            numeric[c] = s.str.replace_all(_NUMERIC_JUNK, "").cast(pl.Float64, strict=False).to_numpy().astype(float)
        elif s.dtype in (pl.Date, pl.Datetime):
            numeric[c] = s.cast(pl.Datetime("us")).cast(pl.Int64).cast(pl.Float64).to_numpy().astype(float) / 86_400_000_000.0
        elif role == "text":
            sub = s.filter(pl.Series(valid)).drop_nulls()
            if 2 <= sub.n_unique() <= cfg.learned_max_categories:
                cats = sub.unique().sort().to_list()
                lookup = {v: i for i, v in enumerate(cats)}
                categorical[c] = np.array([lookup.get(v, -1) if v is not None else -1 for v in s.to_list()], dtype=float)
    return numeric, categorical


def _oof_predictions(
    F: np.ndarray, y: np.ndarray, cat_mask: np.ndarray, valid_pos: np.ndarray, cfg: RowTriageConfig,
    exclude_from_training: np.ndarray | None = None,
):
    """Out-of-fold predictions so a row is never judged by a model that memorised it."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    # SeedSequence with a salt: fold labels can never coincide with a user's own seeded data
    rng = np.random.default_rng([cfg.random_state, 0x5EED])
    known = valid_pos[~np.isnan(y[valid_pos])]
    folds = rng.integers(0, cfg.learned_folds, known.size)
    pred = np.full(y.shape[0], np.nan)
    for k in range(cfg.learned_folds):
        train = known[folds != k]
        if exclude_from_training is not None:
            train = train[~exclude_from_training[train]]
        test = known[folds == k]
        if train.size > cfg.learned_fit_sample:
            train = rng.choice(train, cfg.learned_fit_sample, replace=False)
        if train.size < 30 or test.size == 0:
            continue
        model = HistGradientBoostingRegressor(
            max_iter=120, learning_rate=0.1, max_leaf_nodes=15, categorical_features=cat_mask if cat_mask.any() else None,
            random_state=cfg.random_state,
        )
        model.fit(F[train], y[train])
        pred[test] = model.predict(F[test])
    return pred, known


def _robust_sigma(r: np.ndarray) -> float:
    if r.size == 0:
        return 0.0
    sig = float(np.median(np.abs(r - np.median(r)))) * 1.4826
    return sig if sig > 0 else float(np.std(r))


def _scaled_residuals(resid: np.ndarray, F_rows: np.ndarray, base_sigma: float, cfg: RowTriageConfig) -> np.ndarray:
    """|residual| / error scale, where the scale depends on which predictors were missing for that row."""
    nan = np.isnan(F_rows)
    cols = np.flatnonzero(nan.any(axis=0))
    if cols.size == 0:
        return np.abs(resid) / base_sigma
    cols = cols[np.argsort(-nan[:, cols].sum(axis=0))][:20]
    pattern = (nan[:, cols].astype(np.int64) * (1 << np.arange(cols.size, dtype=np.int64))).sum(axis=1)
    sigma = np.full(resid.shape[0], base_sigma)
    for pid in np.unique(pattern):
        m = pattern == pid
        if m.sum() >= cfg.learned_min_group:
            g = _robust_sigma(resid[m])
            if g > 0:
                sigma[m] = g
    return np.abs(resid) / sigma


def _learned_layer(
    data: pl.DataFrame,
    roles: dict[str, str],
    valid: np.ndarray,
    cfg: RowTriageConfig,
    flags: list[RowFlag],
    explanations: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    """
    Cross-column consistency. Each numeric column is predicted from the others; a cell is flagged when its
    actual value is far from what the rest of the row implies. Adds REVIEW flags only, never FIX or EXCLUDE.
    Columns the others cannot explain (R2 below the gate) are skipped, so ordinary spread is not treated as error.
    """
    n_valid = int(valid.sum())
    if n_valid < cfg.learned_min_rows:
        return {"ran": False, "reason": f"too_few_rows ({n_valid} < {cfg.learned_min_rows})"}
    try:
        import sklearn  # noqa: F401
    except ImportError:
        logger.warning("scikit-learn not installed; learned row layer skipped")
        return {"ran": False, "reason": "scikit-learn_not_installed"}

    numeric, categorical = _model_matrix(data, roles, valid, cfg)
    if not numeric or len(numeric) + len(categorical) < 2:
        return {"ran": False, "reason": "too_few_usable_columns"}

    names = list(numeric)[: cfg.learned_max_predictors] + list(categorical)[: max(0, cfg.learned_max_predictors - len(numeric))]
    full = np.column_stack([numeric[c] if c in numeric else categorical[c] for c in names])
    is_cat = np.array([c in categorical for c in names])
    valid_pos = np.flatnonzero(valid)

    # judge columns with the fewest gaps first
    targets = sorted(
        (c for c in numeric if c in names),
        key=lambda c: float(np.isnan(numeric[c][valid_pos]).mean()),
    )[: cfg.learned_max_targets]

    judged: dict[str, dict[str, float]] = {}
    skipped: dict[str, str] = {}
    total_mask = np.zeros(data.height, dtype=bool)
    for c in targets:
        j = names.index(c)
        keep_cols = [i for i in range(len(names)) if i != j]
        F = full[:, keep_cols]
        y = numeric[c]
        pred, known = _oof_predictions(F, y, is_cat[keep_cols], valid_pos, cfg)
        ok = known[~np.isnan(pred[known])]
        if ok.size < cfg.learned_min_rows:
            skipped[c] = "too_few_known_values"
            continue
        resid = y[ok] - pred[ok]
        var_y = float(np.var(y[ok]))
        r2 = 1.0 - float(np.var(resid)) / var_y if var_y > 0 else 0.0
        if r2 < cfg.learned_min_r2:
            skipped[c] = f"not_explained_by_other_columns (r2={r2:.2f})"
            continue
        sigma = float(np.median(np.abs(resid - np.median(resid)))) * 1.4826
        if sigma <= 0:
            sigma = float(np.std(resid))
        if sigma <= 0:
            skipped[c] = "zero_residual_spread"
            continue

        # Refit once without the rows that look wrong, so bad rows cannot teach the model that bad is normal.
        first = np.zeros(data.height, dtype=bool)
        first[ok[np.abs(resid) / sigma >= cfg.learned_z_threshold]] = True
        if first.any():
            pred2, _ = _oof_predictions(F, y, is_cat[keep_cols], valid_pos, cfg, exclude_from_training=first)
            ok2 = known[~np.isnan(pred2[known])]
            if ok2.size >= cfg.learned_min_rows:
                pred, ok = pred2, ok2
                resid = y[ok] - pred[ok]
                clean_resid = resid[~first[ok]]
                sig2 = float(np.median(np.abs(clean_resid - np.median(clean_resid)))) * 1.4826 if clean_resid.size else 0.0
                sigma = sig2 if sig2 > 0 else sigma

        z = _scaled_residuals(y[ok] - pred[ok], F[ok], sigma, cfg)
        threshold = max(cfg.learned_z_threshold, cfg.learned_tail_multiple * float(np.percentile(z, 99)))
        hit = z >= threshold
        cap = max(1, int(cfg.learned_max_flag_share * ok.size))
        if hit.sum() > cap:
            hit = z >= max(np.sort(z)[-cap], threshold)
        judged[c] = {"r2": round(r2, 3), "flagged": int(hit.sum()), "threshold": round(float(threshold), 1)}
        if not hit.any():
            continue
        mask = np.zeros(data.height, dtype=bool)
        mask[ok[hit]] = True
        flags.append(RowFlag(RowVerdict.REVIEW, "relationship_violation", c, mask))
        total_mask |= mask
        for pos, zz in zip(ok[hit], z[hit]):
            exp = pred[pos]
            exp_txt = int(round(exp)) if float(exp).is_integer() or abs(exp) >= 100 else round(float(exp), 2)
            explanations.setdefault(int(pos), []).append(
                {"column": c, "value": data[c][int(pos)], "expected": exp_txt, "z": round(float(zz), 1)}
            )

    for pos in explanations:
        explanations[pos].sort(key=lambda d: -d["z"])
        del explanations[pos][3:]

    logger.info("Learned layer: judged %d columns, %d rows flagged", len(judged), int(total_mask.sum()))
    return {
        "ran": True,
        "reason": "ok",
        "method": "cross_column_prediction",
        "judged_columns": judged,
        "skipped_columns": skipped,
        "rows_flagged": int(total_mask.sum()),
    }


def _sparse_limit(ncols: int, cfg: "RowTriageConfig") -> int:
    """Max filled cells for a row to count as sparse. Never below 1, so an id-only row is always sparse."""
    return max(1, int(cfg.sparse_fill_ratio * ncols))


def sparse_row_mask(df: pl.DataFrame, config: RowTriageConfig | None = None) -> np.ndarray:
    """Rows with almost nothing filled. The pipeline must never impute these (it would invent a record)."""
    cfg = config or RowTriageConfig()
    cols = [c for c in df.columns if c != ROW_ID]
    if df.height == 0 or len(cols) < cfg.sparse_min_columns:
        return np.zeros(df.height, dtype=bool)
    filled = np.zeros(df.height, dtype=np.int32)
    for c in cols:
        filled += (~df.select(_blank_expr(c, df[c].dtype)).to_series().to_numpy().astype(bool)).astype(np.int32)
    return (filled > 0) & (filled <= _sparse_limit(len(cols), cfg))


def triage_rows(df: pl.DataFrame, config: RowTriageConfig | None = None) -> RowTriageResult:
    """Classify every row of ``df``. ``df`` may carry a ROW_ID column to keep ids stable across steps."""
    cfg = config or RowTriageConfig()
    n = df.height
    ids = df[ROW_ID].to_numpy().astype(np.int64) if ROW_ID in df.columns else np.arange(n, dtype=np.int64)
    data = df.drop(ROW_ID) if ROW_ID in df.columns else df
    cols = data.columns
    flags: list[RowFlag] = []
    roles: dict[str, str] = {}

    def add(verdict: RowVerdict, code: str, column: str | None, mask: np.ndarray) -> None:
        if mask.any():
            flags.append(RowFlag(verdict, code, column, mask.astype(bool)))

    if n == 0 or not cols:
        return RowTriageResult(ids, [], np.zeros(n, dtype=np.int8), {}, {})

    text_cols = [c for c in cols if _is_text(data[c].dtype)]
    for c in cols:
        if c in cfg.protected_columns:
            roles[c] = "protected"
        elif _is_id_column(c):
            roles[c] = "id"
        elif data[c].dtype.is_numeric():
            roles[c] = "numeric"
        elif _is_text(data[c].dtype):
            roles[c] = "text"
        else:
            roles[c] = "other"

    # blank matrix, one bool array per column
    blanks: dict[str, np.ndarray] = {
        c: data.select(_blank_expr(c, data[c].dtype)).to_series().to_numpy().astype(bool) for c in cols
    }
    filled = np.zeros(n, dtype=np.int32)
    for c in cols:
        filled += (~blanks[c]).astype(np.int32)
    ncols = len(cols)

    # numeric view of every text column (a stray header row can turn numeric columns into text)
    parsed: dict[str, pl.Series] = {}
    parsed_ok: dict[str, np.ndarray] = {}
    placeholder_np: dict[str, np.ndarray] = {}
    for c in text_cols:
        p = data[c].str.replace_all(_NUMERIC_JUNK, "").cast(pl.Float64, strict=False)
        parsed[c] = p
        parsed_ok[c] = p.is_not_null().to_numpy().astype(bool)
        placeholder_np[c] = (
            data[c].str.strip_chars().str.to_lowercase().is_in(list(cfg.placeholders)).fill_null(False).to_numpy().astype(bool)
            if cfg.placeholders else np.zeros(n, dtype=bool)
        )

    # 1. Structural: blank rows
    add(RowVerdict.EXCLUDE, "blank_row", None, filled == 0)

    # 2. Structural: footer notes (single text cell, near the end, note-like)
    if ncols >= 3:
        tail = np.zeros(n, dtype=bool)
        tail[max(0, n - cfg.footer_scan_rows):] = True
        single = filled == 1
        for c in text_cols:
            s = data[c]
            looks_like_note = (
                s.str.len_chars().fill_null(0).gt(25)
                | s.str.contains(r"[:\*]").fill_null(False)
                | s.str.contains("(?i)" + _NOTE_LABEL.pattern).fill_null(False)
            ).to_numpy()
            add(RowVerdict.EXCLUDE, "footer_note", c, single & tail & ~blanks[c] & looks_like_note)

    # 3. Structural: repeated header rows
    if len(text_cols) >= 2:
        needed = max(2, math.ceil(len(text_cols) / 2))
        hits = np.zeros(n, dtype=np.int32)
        for c in text_cols:
            target = _norm(c)
            if not target:
                continue
            m = (
                data[c].str.to_lowercase().str.replace_all(r"[^a-z0-9]+", "").fill_null("") == target
            ).to_numpy()
            hits += m.astype(np.int32)
        add(RowVerdict.EXCLUDE, "repeated_header", None, hits >= needed)

    # 4. Structural: summary / total rows
    for c in text_cols[: cfg.summary_column_scan]:
        s = data[c]
        label = s.str.contains("(?i)" + _SUMMARY_LABEL.pattern).fill_null(False)
        label = label & s.str.len_chars().fill_null(0).le(40)
        label_np = label.to_numpy().astype(bool)
        if not label_np.any():
            continue
        if n >= cfg.min_rows_for_stats and label_np.sum() > cfg.summary_max_share * n:
            continue  # frequent label: it is a category value, not a total row
        other_text_filled = np.zeros(n, dtype=bool)
        for o in text_cols:
            if o != c:
                other_text_filled |= ~blanks[o] & ~parsed_ok[o]
        others_filled = filled - (~blanks[c]).astype(np.int32)
        add(RowVerdict.EXCLUDE, "summary_row", c, label_np & ~other_text_filled & (others_filled >= 1))

    # 5. Structural: exact duplicates (later copies only)
    if cfg.dedupe_exact:
        try:
            dup = ~data.select(pl.struct(pl.all()).is_first_distinct()).to_series().to_numpy().astype(bool)
            add(RowVerdict.EXCLUDE, "exact_duplicate", None, dup & (filled > 0))
        except Exception as exc:  # unsupported dtype inside struct
            logger.warning("Duplicate check skipped: %s", exc)

    # 6. Sparse rows (REVIEW, and later barred from imputation)
    if ncols >= cfg.sparse_min_columns:
        already_footer = np.zeros(n, dtype=bool)
        for f in flags:
            if f.code == "footer_note":
                already_footer |= f.mask
        add(RowVerdict.REVIEW, "sparse_row", None, (filled > 0) & (filled <= _sparse_limit(ncols, cfg)) & ~already_footer)

    # valid rows = not structurally excluded; all statistics use them
    excluded_so_far = np.zeros(n, dtype=bool)
    for f in flags:
        if f.verdict is RowVerdict.EXCLUDE:
            excluded_so_far |= f.mask
    valid = ~excluded_so_far
    n_valid = int(valid.sum())
    non_imputable: dict[str, float] = {}

    # 7. Column-level checks
    for c in cols:
        role = roles[c]
        s = data[c]
        numeric_text = False

        if role == "text":
            cand = valid & ~blanks[c] & ~placeholder_np[c]
            total = int(cand.sum())
            if total >= 5:
                rate = float((parsed_ok[c] & cand).sum()) / total
                if rate >= cfg.type_majority:
                    numeric_text = True
                    roles[c] = "numeric_text"
                    if rate < 1.0:
                        add(RowVerdict.REVIEW, "type_violation", c, cand & ~parsed_ok[c])

        is_num = role == "numeric" or numeric_text
        vals = s.cast(pl.Float64, strict=False) if role == "numeric" else (parsed[c] if numeric_text else None)

        # 7a. masked null placeholders in text columns
        if role == "text" and cfg.placeholders:
            add(RowVerdict.FIX, "masked_null", c, placeholder_np[c] & valid)

        # 7b. missing values: safe to impute only when the column is mostly present
        if (is_num or role == "text") and n_valid:
            missing = blanks[c] | placeholder_np[c] if role == "text" else blanks[c]
            null_rate = float((missing & valid).sum()) / n_valid
            if null_rate > 0:
                if null_rate <= cfg.max_impute_null_rate:
                    add(RowVerdict.FIX, "impute", c, blanks[c] & valid)
                else:
                    non_imputable[c] = null_rate

        # 7c. outliers (flag only, never changed here)
        if is_num and vals is not None and n_valid >= cfg.min_rows_for_stats:
            valid_vals = vals.filter(pl.Series(valid)).drop_nulls().drop_nans()
            if valid_vals.len() >= cfg.min_rows_for_stats:
                q1, q3 = valid_vals.quantile(0.25), valid_vals.quantile(0.75)
                if q1 is not None and q3 is not None and q3 > q1:
                    iqr = q3 - q1
                    lo, hi = q1 - cfg.outlier_iqr_k * iqr, q3 + cfg.outlier_iqr_k * iqr
                    out = ((vals < lo) | (vals > hi)).fill_null(False).to_numpy().astype(bool) & valid
                    add(RowVerdict.REVIEW, "outlier", c, out)

        # 7d. business rules (flag only)
        toks = _tokens(c)
        if role == "text" and not numeric_text and "email" in toks:
            bad = (~s.str.contains(r"^[^@\s]+@[^@\s]+\.[^@\s]+$").fill_null(True)).to_numpy().astype(bool)
            add(RowVerdict.REVIEW, "invalid_email", c, bad & ~blanks[c] & valid)
        if is_num and vals is not None and "age" in toks:
            bad = ((vals < 0) | (vals > 120)).fill_null(False).to_numpy().astype(bool)
            add(RowVerdict.REVIEW, "age_out_of_range", c, bad & valid)

    # 8. Cross-column rule: end date earlier than start date
    date_cols = [c for c in cols if data[c].dtype in (pl.Date, pl.Datetime)]
    if len(date_cols) >= 2:
        start = next((c for c in date_cols if _tokens(c) & {"start", "order", "create", "created", "begin", "open"}), None)
        end = next((c for c in date_cols if _tokens(c) & {"end", "ship", "shipped", "close", "closed", "finish", "deliver", "delivered", "expire"}), None)
        if start and end and start != end:
            bad = (data[end] < data[start]).fill_null(False).to_numpy().astype(bool)
            add(RowVerdict.REVIEW, "chronology_violation", end, bad & valid)

    explanations: dict[int, list[dict[str, Any]]] = {}
    learned_info: dict[str, Any] = {"ran": False, "reason": "disabled"}
    if cfg.learned_enabled:
        learned_info = _learned_layer(data, roles, valid, cfg, flags, explanations)

    if cfg.text_enabled:
        from app.services.text_intelligence import run_text_layer  # local import: text layer imports this module

        learned_info["text"] = run_text_layer(data, roles, valid, cfg, flags, explanations)

    severity = np.zeros(n, dtype=np.int8)
    for f in flags:
        np.maximum(severity, np.where(f.mask, _SEVERITY[f.verdict], 0).astype(np.int8), out=severity)

    result = RowTriageResult(ids, flags, severity, non_imputable, roles, explanations, learned_info)
    logger.info("Row triage: %s", result.summary()["verdicts"])
    return result
