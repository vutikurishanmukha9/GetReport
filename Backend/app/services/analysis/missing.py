""" Missingness Diagnostic Engine """
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

try:  # SciPy is optional; fallbacks keep the engine dependency-light.
    from scipy import stats as _sps
except Exception:  # pragma: no cover
    _sps = None

logger = logging.getLogger(__name__)
SCHEMA_VERSION = "2.0"


# --------------------------------------------------------------------------- config
@dataclass(frozen=True)
class MissingnessConfig:
    alpha: float = 0.05                 # FDR-adjusted significance level
    max_rows_sample: int = 50_000       # stride sample for heavy tests (counts stay exact)
    max_target_cols: int = 20           # columns with missing analysed in depth
    max_predictors: int = 25            # candidate explanatory columns per target
    min_group: int = 8                  # min rows in missing / observed group
    min_rank_biserial: float = 0.20     # effect gate (AUC >= 0.60 or <= 0.40)
    min_cramers_v: float = 0.10
    phi_threshold: float = 0.30
    block_phi: float = 0.50             # co-missing block grouping threshold
    max_cat_levels: int = 30
    mnar_rate: float = 0.30             # unexplained missing rate that raises MNAR suspicion
    drop_rate: float = 0.60             # column too empty to impute responsibly
    littles_max_cols: int = 12
    littles_max_rows: int = 20_000
    littles_max_patterns: int = 64
    top_k: int = 10
    treat_nan_as_missing: bool = True
    sentinel_strings: tuple[str, ...] = (
        "", "na", "n/a", "nan", "null", "none", "nil", "-", "--", "?",
        "unknown", "missing", "undefined", "#n/a",
    )
    sentinel_numbers: tuple[float, ...] = (-999, -9999, -99999, 9999, 99999, 999999)


# --------------------------------------------------------------------------- stats helpers
def _norm_p2(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def _chi2_sf(x: float, dof: float) -> float:
    if dof <= 0 or not np.isfinite(x):
        return float("nan")
    if _sps is not None:
        return float(_sps.chi2.sf(x, dof))
    z = ((x / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))  # Wilson-Hilferty
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _rank_avg(x: np.ndarray) -> np.ndarray:
    if _sps is not None:
        return _sps.rankdata(x, method="average")
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2.0)[inv.ravel()]


def _bh_adjust(p: np.ndarray) -> np.ndarray:
    p = np.nan_to_num(np.asarray(p, dtype=float), nan=1.0)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    scaled = p[order] * n / np.arange(1, n + 1)
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(scaled, 0.0, 1.0)
    return out


def _auc_mw(x_miss: np.ndarray, x_obs: np.ndarray) -> tuple[float, float]:
    """AUC of a feature for separating missing from observed rows, with a normal-approx p."""
    n1, n0 = len(x_miss), len(x_obs)
    ranks = _rank_avg(np.concatenate([x_miss, x_obs]))
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    sigma = math.sqrt(n1 * n0 * (n1 + n0 + 1) / 12.0)
    z = (u - n1 * n0 / 2.0) / sigma if sigma > 0 else 0.0
    return float(u / (n1 * n0)), _norm_p2(z)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    pooled = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def _cat_assoc(codes: np.ndarray, m: np.ndarray, min_group: int) -> tuple[float, float, int] | None:
    """Chi-square test and Cramer's V between category codes (-1 = null) and missingness."""
    valid = codes >= 0
    c, mm = codes[valid], m[valid]
    if c.size < 30:
        return None
    levels, inv = np.unique(c, return_inverse=True)
    table = np.zeros((len(levels), 2))
    np.add.at(table, (inv.ravel(), mm.astype(int)), 1)
    table = table[table.sum(1) >= 5]
    if table.shape[0] < 2 or (table.sum(0) < min_group).any():
        return None
    exp = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / table.sum()
    chi2 = float(((table - exp) ** 2 / np.where(exp > 0, exp, 1)).sum())
    n = table.sum()
    return chi2, float(math.sqrt(chi2 / n)), table.shape[0] - 1


# --------------------------------------------------------------------------- dtype helpers
def _is_cat(t: pl.DataType) -> bool:
    return t in (pl.String, pl.Categorical, pl.Boolean) or str(t).startswith("Enum")


def _severity(rate: float) -> str:
    return "critical" if rate > 0.5 else "high" if rate > 0.2 else "medium" if rate > 0.05 else "low"


def _py(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


# --------------------------------------------------------------------------- Little's MCAR
def _littles_mcar(X: np.ndarray, alpha: float, max_patterns: int,
                  max_iter: int = 200, tol: float = 1e-6) -> dict[str, Any]:
    """Little (1988) chi-square test. EM estimates mu and Sigma under multivariate normality."""
    n, p = X.shape
    obs = ~np.isnan(X)
    keys, inv = np.unique(obs, axis=0, return_inverse=True)
    inv = inv.ravel()
    patterns = []
    for k, key in enumerate(keys):
        o = np.where(key)[0]
        if o.size:
            patterns.append((np.where(inv == k)[0], o, np.where(~key)[0]))
    if len(patterns) < 2:
        return {"status": "skipped", "reason": "single_pattern"}
    if len(patterns) > max_patterns:
        return {"status": "skipped", "reason": "too_many_patterns", "n_patterns": len(patterns)}

    mu, sigma, eye, converged = np.nanmean(X, axis=0), np.eye(p), np.eye(p), False
    for _ in range(max_iter):
        sx, sxx, cnt = np.zeros(p), np.zeros((p, p)), 0
        for rows, o, mi in patterns:
            xo, nj = X[np.ix_(rows, o)], len(rows)
            xf = np.empty((nj, p))
            xf[:, o] = xo
            cov_add = np.zeros((p, p))
            if mi.size:
                soo, smo = sigma[np.ix_(o, o)], sigma[np.ix_(mi, o)]
                b = np.linalg.solve(soo, smo.T).T
                xf[:, mi] = mu[mi] + (xo - mu[o]) @ b.T
                cov_add[np.ix_(mi, mi)] = sigma[np.ix_(mi, mi)] - b @ smo.T
            sx += xf.sum(0)
            sxx += xf.T @ xf + nj * cov_add
            cnt += nj
        mu_n = sx / cnt
        sig_n = sxx / cnt - np.outer(mu_n, mu_n)
        sig_n = (sig_n + sig_n.T) / 2 + 1e-6 * eye
        delta = max(np.abs(mu_n - mu).max(), np.abs(sig_n - sigma).max())
        mu, sigma = mu_n, sig_n
        if delta < tol:
            converged = True
            break

    d2, dof = 0.0, -p
    for rows, o, _mi in patterns:
        diff = X[np.ix_(rows, o)].mean(0) - mu[o]
        d2 += len(rows) * float(diff @ np.linalg.solve(sigma[np.ix_(o, o)], diff))
        dof += o.size
    if dof <= 0:
        return {"status": "skipped", "reason": "non_positive_df"}
    pval = _chi2_sf(d2, dof)
    return {
        "status": "ok", "statistic": d2, "df": dof, "p_value": pval,
        "n_patterns": len(patterns), "em_converged": converged,
        "rejects_mcar": bool(pval < alpha),
        "caveat": "Assumes multivariate normality; use as a screen, not proof.",
    }


# --------------------------------------------------------------------------- disguised missing
def _disguised_missing(df: pl.DataFrame, cfg: MissingnessConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    floor = max(3, int(0.001 * df.height))
    for c, t in df.schema.items():
        try:
            if t in (pl.String, pl.Categorical):
                n = df.select(
                    pl.col(c).cast(pl.String).str.strip_chars().str.to_lowercase().is_in(list(cfg.sentinel_strings)).sum()
                ).item()
                if n >= floor:
                    out[c] = {"kind": "string_placeholder", "count": int(n),
                              "percentage": round(n / df.height * 100, 2)}
            elif t.is_numeric():
                n = df.select(pl.col(c).cast(pl.Float64).is_in([float(v) for v in cfg.sentinel_numbers]).sum()).item()
                if n >= floor:
                    out[c] = {"kind": "numeric_sentinel", "count": int(n),
                              "percentage": round(n / df.height * 100, 2)}
        except Exception:  # pragma: no cover
            continue
    return out


# --------------------------------------------------------------------------- recommendations
def _recommend(dtype: pl.DataType, rate: float, mech: str, predictors: list[str]) -> tuple[str, bool, str]:
    if rate >= 0.6:
        return "drop_column_or_source_more_data", True, "Too sparse to impute without inventing most of the column."
    if mech == "MNAR_suspected":
        return ("flag_and_impute_with_sensitivity_analysis", True,
                "Missingness is unexplained by observed data; keep an indicator and test conclusions under shifted imputations.")
    if mech == "MAR":
        if dtype.is_numeric():
            s = "iterative_imputation_mice" if len(predictors) >= 2 else "knn_imputation"
        elif dtype.is_temporal():
            s = "interpolate_by_ordering"
        else:
            s = "group_conditional_mode"
        return s, rate > 0.2, "Missingness depends on observed columns; condition on them: " + ", ".join(predictors[:3]) + "."
    if dtype.is_numeric():
        s = "median_imputation" if rate <= 0.25 else "flag_and_impute"
    elif dtype == pl.String:
        s = "mode_or_missing_category"
    elif dtype.is_temporal():
        s = "forward_fill_or_median_date"
    else:
        s = "mode_imputation"
    return s, rate > 0.25, "No dependence on observed data detected; simple imputation is defensible."


# --------------------------------------------------------------------------- main entry
def analyze_missing_patterns(df: pl.DataFrame, config: MissingnessConfig | None = None) -> dict[str, Any]:
    cfg = config or MissingnessConfig()
    total_rows = df.height
    if total_rows == 0 or df.width == 0:
        return {"schema_version": SCHEMA_VERSION, "has_missing": False, "message": "Empty dataset"}

    if cfg.treat_nan_as_missing:
        nan_fix = [pl.col(c).fill_nan(None) for c, t in df.schema.items() if t in (pl.Float32, pl.Float64)]
        if nan_fix:
            df = df.with_columns(nan_fix)

    disguised = _disguised_missing(df, cfg)
    null_counts = df.null_count().row(0, named=True)
    cols_missing = sorted((c for c in df.columns if null_counts[c] > 0),
                          key=lambda c: null_counts[c], reverse=True)

    if not cols_missing:
        return {
            "schema_version": SCHEMA_VERSION, "has_missing": False,
            "message": "No missing values detected",
            "complete_cases_pct": 100.0, "data_loss_risk": "None",
            "disguised_missing": disguised,
            "disguised_missing_note": "Placeholders that look like data were found; they are not counted as missing."
            if disguised else None,
        }

    schema = df.schema
    column_details: dict[str, dict[str, Any]] = {}
    for c in cols_missing:
        rate = null_counts[c] / total_rows
        column_details[c] = {
            "count": null_counts[c], "percentage": round(rate * 100, 2),
            "severity": _severity(rate), "dtype": str(schema[c]),
        }

    # ---- order-preserving stride sample for the heavy tests
    step = max(1, math.ceil(total_rows / cfg.max_rows_sample))
    work = df.gather_every(step) if step > 1 else df
    n_w = work.height
    targets = cols_missing[: cfg.max_target_cols]
    M = work.select([pl.col(c).is_null().alias(c) for c in targets]).to_numpy().astype(bool)

    num_pred: dict[str, np.ndarray] = {}
    cat_pred: dict[str, np.ndarray] = {}
    for c, t in schema.items():
        if t.is_numeric() and len(num_pred) < cfg.max_predictors:
            arr = work[c].cast(pl.Float64).to_numpy()
            if np.nanstd(arr) > 0:
                num_pred[c] = arr
        elif _is_cat(t) and len(cat_pred) < cfg.max_predictors:
            codes = work[c].cast(pl.String).cast(pl.Categorical).to_physical().fill_null(-1).to_numpy().astype(np.int64)
            if 1 < len(np.unique(codes[codes >= 0])) <= cfg.max_cat_levels:
                cat_pred[c] = codes
    row_order = np.arange(n_w, dtype=float)

    # ---- MAR screening: every target against every predictor, then FDR over all tests
    # Adapt min_group for smaller datasets or slices while preserving cfg.min_group for larger datasets
    eff_min_group = max(3, min(cfg.min_group, n_w // 4))
    tests: list[dict[str, Any]] = []
    for ti, target in enumerate(targets):
        m = M[:, ti]
        n_miss, n_obs = int(m.sum()), int((~m).sum())
        if n_miss < eff_min_group or n_obs < eff_min_group:
            continue
        for pname, x in num_pred.items():
            if pname == target:
                continue
            xm, xo = x[m], x[~m]
            xm, xo = xm[~np.isnan(xm)], xo[~np.isnan(xo)]
            if len(xm) < eff_min_group or len(xo) < eff_min_group:
                continue
            auc, p = _auc_mw(xm, xo)
            tests.append({"target": target, "predictor": pname, "kind": "numeric", "auc": auc,
                          "effect": abs(2 * auc - 1), "measure": "rank_biserial", "p": p,
                          "cohens_d": _cohens_d(xm, xo), "gate": cfg.min_rank_biserial})
        for pname, codes in cat_pred.items():
            if pname == target:
                continue
            res = _cat_assoc(codes, m, eff_min_group)
            if res:
                chi2, v, dof = res
                tests.append({"target": target, "predictor": pname, "kind": "categorical",
                              "effect": v, "measure": "cramers_v", "p": _chi2_sf(chi2, dof),
                              "gate": cfg.min_cramers_v})
        auc, p = _auc_mw(row_order[m], row_order[~m])
        tests.append({"target": target, "predictor": "__row_order__", "kind": "position", "auc": auc,
                      "effect": abs(2 * auc - 1), "measure": "rank_biserial", "p": p,
                      "gate": cfg.min_rank_biserial})

    if tests:
        for t, padj in zip(tests, _bh_adjust(np.array([t["p"] for t in tests]))):
            t["p_adj"] = float(padj)
            t["significant"] = bool(padj < cfg.alpha and t["effect"] >= t["gate"])

    # ---- Phi matrix between missingness indicators (vectorised)
    inter: list[dict[str, Any]] = []
    blocks: list[list[str]] = []
    if len(targets) >= 2:
        Xf = M.astype(float)
        sd = Xf.std(0)
        Z = np.where(sd > 0, (Xf - Xf.mean(0)) / np.where(sd > 0, sd, 1), 0.0)
        phi = Z.T @ Z / n_w
        iu = np.triu_indices(len(targets), 1)
        pv = np.array([_chi2_sf(n_w * phi[i, j] ** 2, 1) for i, j in zip(*iu)])
        padj = _bh_adjust(pv)
        parent = list(range(len(targets)))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for (i, j), p_raw, pa in zip(zip(*iu), pv, padj):
            val = float(phi[i, j])
            if abs(val) >= cfg.phi_threshold and pa < cfg.alpha:
                inter.append({"column_a": targets[i], "column_b": targets[j],
                              "phi_coefficient": round(val, 4), "p_adj": float(pa),
                              "co_occurrence": "simultaneous" if val > 0 else "alternating"})
            if val >= cfg.block_phi and pa < cfg.alpha:
                parent[find(i)] = find(j)
        groups: dict[int, list[str]] = {}
        for i, c in enumerate(targets):
            groups.setdefault(find(i), []).append(c)
        blocks = [g for g in groups.values() if len(g) > 1]
        inter.sort(key=lambda d: -abs(d["phi_coefficient"]))

    # ---- Little's MCAR (numeric columns, EM)
    mcar_test: dict[str, Any] = {"status": "skipped", "reason": "no_numeric_columns_with_missing"}
    try:
        num_missing = [c for c in cols_missing if schema[c].is_numeric()]
        num_complete = [c for c in df.columns if schema[c].is_numeric() and null_counts[c] == 0]
        if num_missing:
            lstep = max(1, math.ceil(total_rows / cfg.littles_max_rows))
            lw = df.gather_every(lstep) if lstep > 1 else df
            cols = (num_missing[: cfg.littles_max_cols] + num_complete)[: cfg.littles_max_cols]
            while len(cols) >= 2:
                X = lw.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy()
                keep = np.nanstd(X, axis=0) > 0
                X = X[:, keep]
                kept = [c for c, k in zip(cols, keep) if k]
                X = X[~np.isnan(X).all(1)]
                X = (X - np.nanmean(X, 0)) / np.nanstd(X, 0)
                res = _littles_mcar(X, cfg.alpha, cfg.littles_max_patterns)
                if res.get("reason") == "too_many_patterns" and len(cols) > 2:
                    cols = [c for c in cols if c in kept][:-1]
                    continue
                res["columns_used"] = kept
                mcar_test = res
                break
    except Exception as exc:  # pragma: no cover
        mcar_test = {"status": "error", "reason": str(exc)}

    # ---- per-column mechanism, evidence, recommendation
    sig_by_target: dict[str, list[dict[str, Any]]] = {}
    n_tests_by_target: dict[str, int] = {}
    for t in tests:
        n_tests_by_target[t["target"]] = n_tests_by_target.get(t["target"], 0) + 1
        if t.get("significant"):
            sig_by_target.setdefault(t["target"], []).append(t)
    in_block = {c for b in blocks for c in b}

    for c in cols_missing:
        d = column_details[c]
        rate = d["percentage"] / 100
        if c not in targets or n_tests_by_target.get(c, 0) == 0:
            mech, conf, evidence, preds = "undetermined", "none", ["Not enough rows or columns for testing."], []
        else:
            sig = sorted(sig_by_target.get(c, []), key=lambda t: -t["effect"])
            preds = [t["predictor"] for t in sig if t["kind"] != "position"]
            evidence = [f"{t['predictor']}: {t['measure']}={t['effect']:.2f}, p_adj={t['p_adj']:.1e}" for t in sig[:5]]
            if sig:
                mech = "MAR"
                conf = "high" if sig[0]["effect"] >= 2 * sig[0]["gate"] else "medium"
                if c in in_block:
                    evidence.append("Co-missing with other columns (shared mechanism).")
            elif c in in_block:
                mech, conf = "MAR", "low"
                evidence = ["Missing together with other columns (structural or shared cause)."]
            elif rate >= cfg.mnar_rate:
                mech, conf = "MNAR_suspected", "low"
                evidence = [f"{rate:.0%} missing with no observed explainer. The cause may be the value itself or an unmeasured variable."]
            else:
                mech = "MCAR_consistent"
                conf = "medium" if n_tests_by_target[c] >= 10 and d["count"] >= 30 else "low"
                evidence = [f"No significant dependence across {n_tests_by_target[c]} tests."]
        strat, flag, why = _recommend(schema[c], rate, mech, preds)
        d.update({"mechanism": mech, "confidence": conf, "evidence": evidence,
                  "conditioning_predictors": preds[:5], "recommended_imputation": strat,
                  "add_missing_indicator": flag, "rationale": why})

    # ---- v1-compatible correlation list
    corr_out = []
    for t in sorted((t for t in tests if t.get("significant") and t["kind"] != "position"),
                    key=lambda t: -t["effect"])[:10]:
        corr_out.append({
            "missing_column": t["target"], "correlated_with": t["predictor"],
            "correlation": round(t["effect"], 4), "measure": t["measure"], "p_adj": t["p_adj"],
            "interpretation": f"Missingness in '{t['target']}' depends on '{t['predictor']}' "
                              f"({t['measure']}={t['effect']:.2f}, FDR-adjusted p={t['p_adj']:.1e})",
        })
    drift = [{"column": t["target"], "rank_biserial": round(t["effect"], 4),
              "direction": "later_rows" if t["auc"] > 0.5 else "earlier_rows"}
             for t in tests if t.get("significant") and t["kind"] == "position"]

    # ---- row patterns and listwise deletion (exact, full frame)
    row_sum = df.select(pl.sum_horizontal([pl.col(c).is_null() for c in cols_missing]).alias("k")).to_series()
    complete = int((row_sum == 0).sum())
    fully_missing = int(df.select(pl.all_horizontal([pl.col(c).is_null() for c in df.columns])).to_series().sum())
    complete_pct = round(complete / total_rows * 100, 2)
    risk = "High" if complete_pct < 60 else "Moderate" if complete_pct < 85 else "Low"
    incomplete = total_rows - complete
    row_patterns = {
        "complete_rows": complete, "partial_missing_rows": max(0, incomplete - fully_missing),
        "fully_missing_rows": fully_missing, "complete_cases_percentage": complete_pct,
        "data_loss_risk_on_drop": risk,
        "avg_missing_cells_per_incomplete_row": round(float(row_sum.filter(row_sum > 0).mean() or 0), 2),
    }

    # column drop trade-off: how many rows come back if the sparsest columns go
    tradeoff = []
    for k in range(0, min(len(cols_missing), 8) + 1):
        remain = cols_missing[k:]
        pct = 100.0 if not remain else round(
            df.select(pl.all_horizontal([pl.col(c).is_not_null() for c in remain]).sum()).item() / total_rows * 100, 2)
        tradeoff.append({"columns_dropped": k, "last_dropped": cols_missing[k - 1] if k else None,
                         "complete_cases_pct": pct})

    # does deleting incomplete rows change the data? standardised mean difference
    bias_cols = []
    if len(targets) and complete and incomplete:
        comp_mask = ~M.any(1)
        for pname, x in num_pred.items():
            a, b = x[comp_mask], x[~comp_mask]
            a, b = a[~np.isnan(a)], b[~np.isnan(b)]
            if len(a) >= eff_min_group and len(b) >= eff_min_group:
                smd = abs(_cohens_d(a, b))
                if smd >= 0.1:
                    bias_cols.append({"column": pname, "std_mean_diff": round(smd, 3)})
        bias_cols.sort(key=lambda d: -d["std_mean_diff"])

    pcols = targets if len(cols_missing) > 40 else cols_missing
    pat_df = (df.select([pl.col(c).is_null().alias(c) for c in pcols]).group_by(pcols).len()
              .sort("len", descending=True).head(cfg.top_k))
    patterns = [{"missing_columns": [c for c in pcols if r[c]], "rows": r["len"],
                 "percentage": round(r["len"] / total_rows * 100, 2)} for r in pat_df.iter_rows(named=True)]

    mono = False
    if len(targets) >= 2:
        asc = M[:, np.argsort(M.sum(0))]
        mono = bool(np.all(asc[:, :-1] <= asc[:, 1:]))

    # ---- dataset-level verdict
    mechs = {d["mechanism"] for d in column_details.values()}
    if "MAR" in mechs:
        pattern_type = "MAR"
        advice = "Missingness depends on observed features. Use conditional imputation (MICE, KNN, or model-based) and avoid listwise deletion."
    elif "MNAR_suspected" in mechs:
        pattern_type = "MNAR_suspected"
        advice = "Some missingness has no observed explainer. Add indicator flags and run a sensitivity analysis."
    elif fully_missing > total_rows * 0.1:
        pattern_type = "Systematic"
        advice = "Many rows are entirely empty. Remove them before any other step."
    else:
        pattern_type = "MCAR"
        advice = "No dependence on observed data detected (consistent with MCAR, which cannot be proven). Median or mode imputation or listwise deletion is reasonable."
    if mcar_test.get("rejects_mcar") and pattern_type == "MCAR":
        pattern_type = "MAR_or_MNAR"
        advice = "Little's test rejects MCAR although no single predictor explains it. Possible nonlinear or unmeasured dependence."
    if mono:
        advice += " Rows follow a monotone dropout pattern; sequential or longitudinal imputation fits."

    return _py({
        "schema_version": SCHEMA_VERSION,
        "has_missing": True,
        "columns_affected": len(cols_missing),
        "column_details": column_details,
        "missing_correlations": corr_out,
        "missing_inter_correlations": inter[:10],
        "row_patterns": row_patterns,
        "inferred_pattern": pattern_type,
        "recommendation": advice,
        "mcar_test": mcar_test,
        "missingness_blocks": blocks,
        "missing_patterns": patterns,
        "monotone_dropout": mono,
        "row_order_drift": drift,
        "listwise_deletion": {
            "complete_cases_pct": complete_pct, "risk": risk,
            "bias_risk_columns": bias_cols[:5], "column_drop_tradeoff": tradeoff,
        },
        "disguised_missing": disguised,
        "meta": {"rows": total_rows, "rows_analysed": n_w, "stride": step,
                 "columns_deep_tested": len(targets), "tests_run": len(tests),
                 "fdr_method": "benjamini_hochberg", "scipy_available": _sps is not None},
    })