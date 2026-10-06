"""
Text Intelligence
=================
Row-level checks for non-numeric (text / categorical) data. Companion to ``row_intelligence``.

Three techniques, all learned from the uploaded dataset itself and all REVIEW-only (nothing is edited here):

    1. category_conflict   a categorical value contradicts the rest of the row
                           (state=Kerala while city=Hyderabad), found by predicting each
                           categorical column from the others and flagging confident disagreements.
    2. format_violation    a value breaks the column's learned shape
                           (phone "12345" in a column of 10-digit numbers).
    3. variant_value       a rare value is a spelling, case or whitespace variant of a common one
                           ("hyderabad", "Hyderabad ", "Chenai" vs "Chennai"). Carries a suggestion.

Free-text columns (long, high-variety strings) are never judged. A column that cannot be modelled is skipped
with a reason, never guessed at.
"""
from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import polars as pl

from app.services.row_intelligence import RowFlag, RowTriageConfig, RowVerdict, _model_matrix

logger = logging.getLogger(__name__)

_SHAPE_STEPS = (
    (r"\p{Nd}", "9"),
    (r"[\p{Lo}\p{Lm}\p{Lt}]", "a"),
    (r"\p{Lu}", "A"),
    (r"\p{Ll}", "a"),
)
_MAX_DRIVERS = 3


# ─── helpers ─────────────────────────────────────────────────────────────────
def _is_text(dtype: pl.DataType) -> bool:
    return dtype in (pl.Utf8, pl.String)


def _blank_and_placeholder(s: pl.Series, cfg: RowTriageConfig) -> tuple[np.ndarray, np.ndarray]:
    stripped = s.str.strip_chars()
    blank = (s.is_null() | (stripped == "")).to_numpy().astype(bool)
    if cfg.placeholders:
        ph = stripped.str.to_lowercase().is_in(list(cfg.placeholders)).fill_null(False).to_numpy().astype(bool)
    else:
        ph = np.zeros(s.len(), dtype=bool)
    return blank, ph


def _claimed(info: dict[str, Any], column: str, n: int) -> np.ndarray:
    return info["_claimed"].get(column, np.zeros(n, dtype=bool))


def _claim(info: dict[str, Any], column: str, mask: np.ndarray) -> None:
    info["_claimed"][column] = info["_claimed"].get(column, np.zeros(mask.shape[0], dtype=bool)) | mask


def _add_driver(explanations: dict[int, list[dict[str, Any]]], pos: int, driver: dict[str, Any]) -> None:
    drivers = explanations.setdefault(pos, [])
    if len(drivers) < _MAX_DRIVERS:
        drivers.append(driver)


# ─── 2. format shape ─────────────────────────────────────────────────────────
def _shapes(s: pl.Series) -> tuple[pl.Series, pl.Series]:
    exact = s
    for pattern, repl in _SHAPE_STEPS:
        exact = exact.str.replace_all(pattern, repl)
    coarse = exact
    for ch in ("9", "a", "A"):
        coarse = coarse.str.replace_all(f"{ch}+", ch)
    return exact, coarse


def _established_shapes(shapes: pl.Series, cand: np.ndarray, cfg: RowTriageConfig) -> list[str] | None:
    sub = shapes.filter(pl.Series(cand))
    total = sub.len()
    if total < cfg.text_min_rows:
        return None
    vc = sub.value_counts().sort("count", descending=True)
    names = vc[vc.columns[0]].to_list()
    counts = vc["count"].to_list()
    floor = max(cfg.text_shape_established, 3.0 / total)
    est = [nm for nm, ct in zip(names, counts) if ct / total >= floor]
    coverage = sum(ct for nm, ct in zip(names, counts) if nm in set(est)) / total
    if not est or len(est) > cfg.text_shape_max_established or coverage < cfg.text_shape_coverage:
        return None
    return est


def _format_layer(
    data: pl.DataFrame, roles: dict[str, str], valid: np.ndarray, cfg: RowTriageConfig,
    flags: list[RowFlag], explanations: dict[int, list[dict[str, Any]]], info: dict[str, Any],
) -> None:
    for c in data.columns:
        s = data[c]
        if roles.get(c) not in ("text", "id") or not _is_text(s.dtype):
            continue
        blank, ph = _blank_and_placeholder(s, cfg)
        cand = valid & ~blank & ~ph & ~_claimed(info, c, data.height)
        if int(cand.sum()) < cfg.text_min_rows:
            continue
        if s.filter(pl.Series(cand)).n_unique() <= cfg.text_max_classes:
            info["skipped"][c] = "low_cardinality_category"   # a shape means nothing for a handful of fixed labels
            continue
        if float(s.filter(pl.Series(cand)).str.len_chars().mean() or 0) > cfg.text_free_text_len:
            info["skipped"][c] = "free_text"
            continue
        exact, coarse = _shapes(s)
        top_exact_share = float(exact.filter(pl.Series(cand)).value_counts().sort("count", descending=True)["count"][0]) / int(cand.sum())
        order = (("exact", exact), ("coarse", coarse)) if top_exact_share >= cfg.text_fixed_width_share else (("coarse", coarse),)
        for label, shapes in order:
            est = _established_shapes(shapes, cand, cfg)
            if est is None:
                continue
            bad = cand & ~shapes.is_in(est).fill_null(False).to_numpy().astype(bool)
            if bad.any():
                top = est[0]
                example = s.filter(pl.Series(cand) & (shapes == top)).head(1).to_list()
                example = example[0] if example else ""
                flags.append(RowFlag(RowVerdict.REVIEW, "format_violation", c, bad))
                for pos in np.flatnonzero(bad)[:1000]:
                    _add_driver(explanations, int(pos), {
                        "column": c, "value": s[int(pos)],
                        "note": f"does not match the usual format of this column, for example {example}",
                    })
            info["format_checked"][c] = {"granularity": label, "shapes": est, "flagged": int(bad.sum())}
            break
        else:
            info["skipped"].setdefault(c, "no_consistent_format")


# ─── 3. variants ─────────────────────────────────────────────────────────────
def _norm(v: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", v.lower())).strip()


def _letter_ratio(v: str) -> float:
    return sum(ch.isalpha() for ch in v) / len(v) if v else 0.0


def _variant_layer(
    data: pl.DataFrame, roles: dict[str, str], valid: np.ndarray, cfg: RowTriageConfig,
    flags: list[RowFlag], explanations: dict[int, list[dict[str, Any]]], info: dict[str, Any],
) -> None:
    n_valid = int(valid.sum())
    if n_valid < cfg.text_min_rows:
        return
    for c in data.columns:
        s = data[c]
        if roles.get(c) != "text" or not _is_text(s.dtype):
            continue
        blank, ph = _blank_and_placeholder(s, cfg)
        cand = valid & ~blank & ~ph
        vc = s.filter(pl.Series(cand)).value_counts()
        distinct = vc.height
        if distinct < 2 or distinct > cfg.text_max_distinct or distinct / max(1, int(cand.sum())) > 0.5:
            continue
        counts: dict[str, int] = dict(zip(vc[vc.columns[0]].to_list(), vc["count"].to_list()))

        suggestion: dict[str, tuple[str, str]] = {}   # raw value -> (canonical raw value, reason)

        # A. same text after normalising case, spacing and punctuation
        groups: dict[str, list[str]] = {}
        for raw in counts:
            groups.setdefault(_norm(raw), []).append(raw)
        for key, raws in groups.items():
            if len(raws) > 1 and key:
                canon = max(raws, key=lambda r: (counts[r], r == r.strip(), -len(r)))
                for raw in raws:
                    if raw != canon:
                        suggestion[raw] = (canon, "differs only in case, spacing or punctuation")

        # B. rare value very close to one common value
        key_counts = {k: sum(counts[r] for r in raws) for k, raws in groups.items()}
        key_canon = {k: max(raws, key=lambda r: counts[r]) for k, raws in groups.items()}
        rare_max = max(2, int(0.002 * int(cand.sum())))
        frequent = sorted((k for k, v in key_counts.items() if v >= 10 and k), key=lambda k: -key_counts[k])[:300]
        for key, cnt in key_counts.items():
            if cnt > rare_max or len(key) < 3 or _letter_ratio(key) < 0.7 or key in frequent:
                continue
            if all(raw in suggestion for raw in groups[key]):
                continue
            pool = [f for f in frequent if key_counts[f] >= 10 * cnt and abs(len(f) - len(key)) <= 3 and f[0] == key[0]]
            scored = sorted(((SequenceMatcher(None, key, f).ratio(), f) for f in pool), reverse=True)
            target, reason = None, ""
            if len(key) >= 4 and scored and scored[0][0] >= cfg.text_variant_similarity and (len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.05):
                target, reason = scored[0][1], "looks like a misspelling of a common value"
            else:
                pref = [f for f in frequent if f.startswith(key) and len(f) > len(key) and key_counts[f] >= 10 * cnt]
                if len(key) >= 3 and len(pref) == 1:
                    target, reason = pref[0], "may be an abbreviation of a common value"
            if target:
                for raw in groups[key]:
                    suggestion.setdefault(raw, (key_canon[target], reason))

        if not suggestion:
            continue
        values = list(suggestion)
        mask = cand & s.is_in(values).fill_null(False).to_numpy().astype(bool)
        if not mask.any():
            continue
        flags.append(RowFlag(RowVerdict.REVIEW, "variant_value", c, mask))
        _claim(info, c, mask)
        for pos in np.flatnonzero(mask)[:1000]:
            raw = s[int(pos)]
            canon, reason = suggestion[raw]
            _add_driver(explanations, int(pos), {
                "column": c, "value": repr(raw) if raw != raw.strip() else raw,
                "note": f"{reason}; suggested value: {canon} (appears {counts.get(canon, 0)} times)",
            })
        info["variants"][c] = int(mask.sum())


# ─── 1. category conflicts ───────────────────────────────────────────────────
def _oof_class_proba(
    F: np.ndarray, y: np.ndarray, cat_mask: np.ndarray, known: np.ndarray, cfg: RowTriageConfig,
    exclude: np.ndarray | None = None,
):
    """Out-of-fold class probabilities. Returns (p_actual, confidence, predicted_code) aligned to ``known``."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng([cfg.random_state, 0x7E47])
    folds = rng.integers(0, cfg.learned_folds, known.size)
    p_actual = np.full(known.size, np.nan)
    conf = np.full(known.size, np.nan)
    pred = np.full(known.size, -1, dtype=np.int64)
    for k in range(cfg.learned_folds):
        tr = known[folds != k]
        if exclude is not None:
            tr = tr[~exclude[tr]]
        te_idx = np.flatnonzero(folds == k)
        te = known[te_idx]
        if tr.size > cfg.learned_fit_sample:
            tr = rng.choice(tr, cfg.learned_fit_sample, replace=False)
        classes_in_train = np.unique(y[tr])
        if tr.size < 30 or classes_in_train.size < 2 or te.size == 0:
            continue
        model = HistGradientBoostingClassifier(
            max_iter=60, learning_rate=0.15, max_leaf_nodes=15,
            categorical_features=cat_mask if cat_mask.any() else None, random_state=cfg.random_state,
        )
        model.fit(F[tr], y[tr].astype(int))
        proba = model.predict_proba(F[te])
        classes = model.classes_
        col_of = {int(cl): i for i, cl in enumerate(classes)}
        actual = y[te].astype(int)
        p_actual[te_idx] = [proba[i, col_of[a]] if a in col_of else 0.0 for i, a in enumerate(actual)]
        conf[te_idx] = proba.max(axis=1)
        pred[te_idx] = classes[proba.argmax(axis=1)]
    return p_actual, conf, pred


def _conflict_layer(
    data: pl.DataFrame, roles: dict[str, str], valid: np.ndarray, cfg: RowTriageConfig,
    flags: list[RowFlag], explanations: dict[int, list[dict[str, Any]]], info: dict[str, Any],
) -> None:
    if int(valid.sum()) < cfg.learned_min_rows:
        info["conflicts_skipped"] = f"too_few_rows ({int(valid.sum())} < {cfg.learned_min_rows})"
        return
    try:
        import sklearn  # noqa: F401
    except ImportError:
        info["conflicts_skipped"] = "scikit-learn_not_installed"
        return

    numeric, categorical = _model_matrix(data, roles, valid, cfg)
    targets = [c for c in categorical if 2 <= int(np.unique(categorical[c][valid & (categorical[c] >= 0)]).size) <= cfg.text_max_classes]
    targets = sorted(targets, key=lambda c: int(np.unique(categorical[c][valid]).size))[: cfg.text_max_targets]
    if not targets or len(numeric) + len(categorical) < 2:
        info["conflicts_skipped"] = "too_few_usable_columns"
        return

    valid_pos = np.flatnonzero(valid)
    for c in targets:
        others = [k for k in list(numeric) + list(categorical) if k != c][: cfg.learned_max_predictors]
        if not others:
            continue
        F = np.column_stack([numeric[k] if k in numeric else categorical[k] for k in others])
        cat_mask = np.array([k in categorical for k in others])
        y = categorical[c]
        known = valid_pos[y[valid_pos] >= 0]
        if known.size < cfg.learned_min_rows:
            continue
        labels = data[c].filter(pl.Series(valid)).drop_nulls().unique().sort().to_list()

        p, conf, pred = _oof_class_proba(F, y, cat_mask, known, cfg)
        ok = ~np.isnan(p)
        if ok.sum() < cfg.learned_min_rows:
            info["conflicts_skipped_columns"][c] = "too_few_known_values"
            continue
        actual = y[known].astype(np.int64)
        acc = float((pred[ok] == actual[ok]).mean())
        majority = float(np.bincount(actual[ok]).max() / ok.sum())
        if acc < cfg.text_min_accuracy or acc - majority < cfg.text_min_lift:
            info["conflicts_skipped_columns"][c] = f"not_determined_by_other_columns (accuracy={acc:.2f}, baseline={majority:.2f})"
            continue

        claimed_known = _claimed(info, c, data.height)[known]

        def disagreements(p_, conf_, pred_):
            return (~np.isnan(p_)) & (p_ < cfg.text_p_low) & (conf_ >= cfg.text_confident) & (pred_ != actual) & ~claimed_known

        first = disagreements(p, conf, pred)
        if first.any():
            ex = np.zeros(data.height, dtype=bool)
            ex[known[first]] = True
            p2, conf2, pred2 = _oof_class_proba(F, y, cat_mask, known, cfg, exclude=ex)
            if (~np.isnan(p2)).sum() >= cfg.learned_min_rows:
                p, conf, pred = p2, conf2, pred2
        hit = disagreements(p, conf, pred)

        # A rare value whose rows all point to the same context is a consistent category, not an error:
        # the model simply cannot learn a class that small. Singletons and pairs cannot be judged this way.
        valid_pred = ~np.isnan(p)
        for k in np.unique(actual[hit]):
            rows_k = valid_pred & (actual == k)
            if cfg.text_rare_class_share * known.size > rows_k.sum() >= 3:
                top_share = np.bincount(pred[rows_k]).max() / rows_k.sum()
                if top_share >= cfg.text_coherent_group:
                    hit &= ~(actual == k)
        cap = max(1, int(cfg.learned_max_flag_share * known.size))
        if hit.sum() > cap:
            order = np.argsort(np.where(hit, p, np.inf))[:cap]
            hit = np.zeros_like(hit)
            hit[order] = True
        info["conflicts"][c] = {"accuracy": round(acc, 3), "baseline": round(majority, 3), "flagged": int(hit.sum())}
        if not hit.any():
            continue
        mask = np.zeros(data.height, dtype=bool)
        mask[known[hit]] = True
        flags.append(RowFlag(RowVerdict.REVIEW, "category_conflict", c, mask))
        for pos, pr, cf in zip(known[hit], pred[hit], conf[hit]):
            _add_driver(explanations, int(pos), {
                "column": c, "value": data[c][int(pos)],
                "note": f"conflicts with the rest of the row, which points to {labels[int(pr)]} ({cf:.0%} confidence)",
            })


# ─── entry point ─────────────────────────────────────────────────────────────
def run_text_layer(
    data: pl.DataFrame,
    roles: dict[str, str],
    valid: np.ndarray,
    cfg: RowTriageConfig,
    flags: list[RowFlag],
    explanations: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Runs the three text techniques. Each is isolated so a failure in one never breaks triage or cleaning."""
    info: dict[str, Any] = {
        "ran": True, "format_checked": {}, "variants": {}, "conflicts": {},
        "conflicts_skipped_columns": {}, "skipped": {}, "errors": [], "_claimed": {},
    }
    for name, fn in (("variants", _variant_layer), ("format", _format_layer), ("conflicts", _conflict_layer)):
        try:
            fn(data, roles, valid, cfg, flags, explanations, info)
        except Exception as exc:  # never let a heuristic layer take the pipeline down
            logger.warning("Text layer '%s' failed: %s", name, exc)
            info["errors"].append(f"{name}: {type(exc).__name__}: {exc}")
    info.pop("_claimed", None)
    return info
