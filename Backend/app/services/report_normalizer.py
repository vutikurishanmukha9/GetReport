"""
report_normalizer.py
~~~~~~~~~~~~~~~~~~~~
Shared schema adapters bridging real analysis pipeline outputs to downstream
consumers (Typst, ReportLab, WeasyPrint, DatasetGraphBuilder, TableSemanticChunker,
and RAG Q&A) without synthesizing fake fallback figures.
"""
from __future__ import annotations

from typing import Any


def extract_ledger_issues(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Extract issue dicts from canonical issue_ledger.issues or legacy ledger keys."""
    if not isinstance(payload, dict):
        return []
    ledger = payload.get("issue_ledger")
    if isinstance(ledger, dict):
        issues = ledger.get("issues")
        if isinstance(issues, list):
            return [i for i in issues if isinstance(i, dict)]
    for legacy_key in ("ledger_issues", "issues"):
        legacy = payload.get(legacy_key)
        if isinstance(legacy, list):
            return [i for i in legacy if isinstance(i, dict)]
    return []


def extract_domain_info(analysis: dict[str, Any] | None) -> tuple[str, float]:
    """Return (primary_domain, confidence_0_to_1) from canonical or legacy analysis."""
    if not isinstance(analysis, dict):
        return "general", 0.0

    sem = analysis.get("semantic_analysis")
    if isinstance(sem, dict):
        dom = sem.get("domain")
        if isinstance(dom, dict) and dom.get("primary"):
            return str(dom["primary"]), float(dom.get("confidence", 0.0) or 0.0)
        if isinstance(dom, str) and dom:
            return dom, float(sem.get("confidence", 0.0) or 0.0)

    dom_legacy = analysis.get("domain")
    if isinstance(dom_legacy, dict) and dom_legacy.get("primary"):
        return str(dom_legacy["primary"]), float(dom_legacy.get("confidence", 0.0) or 0.0)
    if isinstance(dom_legacy, str) and dom_legacy:
        return dom_legacy, float(analysis.get("domain_confidence", 0.0) or 0.0)

    return "general", 0.0


def extract_confidence_map(
    payload: dict[str, Any] | None = None,
    analysis: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return mapping column_name -> confidence metrics from canonical or legacy structures."""
    result: dict[str, dict[str, Any]] = {}
    sources: list[Any] = []
    if isinstance(analysis, dict):
        sources.append(analysis.get("confidence_scores"))
        sources.append(analysis.get("confidence"))
    if isinstance(payload, dict):
        sources.append(payload.get("confidence_scores"))
        sources.append(payload.get("confidence"))

    for src in sources:
        if not isinstance(src, dict):
            continue
        cols = src.get("columns")
        if isinstance(cols, list):
            for entry in cols:
                if isinstance(entry, dict) and entry.get("column"):
                    result[str(entry["column"])] = entry
        elif isinstance(cols, dict):
            for col_name, entry in cols.items():
                if isinstance(entry, dict):
                    result[str(col_name)] = entry
        else:
            for col_name, entry in src.items():
                if col_name in ("columns", "dataset_overall", "grade", "summary"):
                    continue
                if isinstance(entry, dict):
                    result[str(col_name)] = entry
        if result:
            break
    return result


def extract_dataset_confidence_summary(
    payload: dict[str, Any] | None = None,
    analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return canonical dataset-level confidence metrics across production and mock schemas."""
    if analysis is None and isinstance(payload, dict):
        analysis = payload.get("analysis") if isinstance(payload.get("analysis"), dict) else {}

    sources: list[Any] = []
    if isinstance(analysis, dict):
        sources.append(analysis.get("confidence_scores"))
        sources.append(analysis.get("confidence"))
    if isinstance(payload, dict):
        sources.append(payload.get("confidence_scores"))
        sources.append(payload.get("confidence"))

    for src in sources:
        if not isinstance(src, dict):
            continue
        score = src.get("dataset_confidence")
        if score is None:
            score = src.get("dataset_overall")
        grade = src.get("dataset_grade") or src.get("grade")
        crit = src.get("critical_issues", [])
        if score is not None or grade is not None or crit:
            return {
                "dataset_confidence": score,
                "dataset_grade": grade,
                "critical_issues": crit if isinstance(crit, list) else [],
            }

    if isinstance(analysis, dict) and analysis.get("quality_score") is not None:
        q_score = analysis["quality_score"]
        grade = "A" if q_score >= 90 else "B" if q_score >= 80 else "C"
        return {"dataset_confidence": q_score, "dataset_grade": grade, "critical_issues": []}

    return {"dataset_confidence": None, "dataset_grade": None, "critical_issues": []}


def extract_strong_correlations(analysis: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return strong correlations list from canonical top-level or nested legacy location."""
    if not isinstance(analysis, dict):
        return []
    top = analysis.get("strong_correlations")
    if isinstance(top, list) and top:
        return [c for c in top if isinstance(c, dict)]
    nested = (analysis.get("correlation") or {}).get("strong_correlations")
    if isinstance(nested, list):
        return [c for c in nested if isinstance(c, dict)]
    return []


def extract_column_profiles(
    analysis: dict[str, Any] | None,
    info: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build normalized per-column profile dict from legacy columns or production summary/info."""
    if not isinstance(analysis, dict):
        analysis = {}
    legacy_cols = analysis.get("columns")
    if isinstance(legacy_cols, dict) and legacy_cols:
        return {str(k): v for k, v in legacy_cols.items() if isinstance(v, dict)}

    profiles: dict[str, dict[str, Any]] = {}
    dtypes = info.get("dtypes", {}) if isinstance(info, dict) else {}
    missing_info = info.get("missing_values", {}) if isinstance(info, dict) else {}

    sem = analysis.get("semantic_analysis")
    roles = sem.get("column_roles", {}) if isinstance(sem, dict) else {}

    summary = analysis.get("summary")
    if isinstance(summary, dict):
        for col_name, stats in summary.items():
            if not isinstance(stats, dict):
                continue
            role_entry = roles.get(col_name, {}) if isinstance(roles, dict) else {}
            miss_entry = missing_info.get(col_name, {}) if isinstance(missing_info, dict) else {}
            profiles[str(col_name)] = {
                "dtype": str(dtypes.get(col_name, stats.get("dtype", "numeric"))),
                "semantic_type": role_entry.get("role") or role_entry.get("semantic_type") or "numeric",
                "missing_pct": float(miss_entry.get("percentage", 0.0) or 0.0) if isinstance(miss_entry, dict) else 0.0,
                "stats": stats,
            }

    cat_dist = analysis.get("categorical_distribution")
    if isinstance(cat_dist, dict):
        for col_name, cat_stats in cat_dist.items():
            if not isinstance(cat_stats, dict) or col_name in profiles:
                continue
            role_entry = roles.get(col_name, {}) if isinstance(roles, dict) else {}
            miss_entry = missing_info.get(col_name, {}) if isinstance(missing_info, dict) else {}
            profiles[str(col_name)] = {
                "dtype": str(dtypes.get(col_name, "categorical")),
                "semantic_type": role_entry.get("role") or role_entry.get("semantic_type") or "categorical",
                "missing_pct": float(miss_entry.get("percentage", 0.0) or 0.0) if isinstance(miss_entry, dict) else 0.0,
                "stats": cat_stats,
            }

    if isinstance(info, dict) and isinstance(info.get("columns"), list):
        for col_name in info["columns"]:
            col_str = str(col_name)
            if col_str not in profiles:
                role_entry = roles.get(col_str, {}) if isinstance(roles, dict) else {}
                miss_entry = missing_info.get(col_str, {}) if isinstance(missing_info, dict) else {}
                profiles[col_str] = {
                    "dtype": str(dtypes.get(col_str, "unknown")),
                    "semantic_type": role_entry.get("role") or role_entry.get("semantic_type") or "unknown",
                    "missing_pct": float(miss_entry.get("percentage", 0.0) or 0.0) if isinstance(miss_entry, dict) else 0.0,
                    "stats": {},
                }

    return profiles


def flatten_recommendations(recs: Any) -> list[dict[str, Any]]:
    """Flatten dict-grouped recommendations or lists into a consistent list of items."""
    if isinstance(recs, list):
        items: list[dict[str, Any]] = []
        for entry in recs:
            if isinstance(entry, dict):
                items.append(entry)
            elif isinstance(entry, str) and entry.strip():
                items.append({"title": entry.strip(), "description": ""})
        return items

    if not isinstance(recs, dict):
        return []

    explicit_items = recs.get("items")
    if isinstance(explicit_items, list) and explicit_items:
        return flatten_recommendations(explicit_items)

    flattened: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    category_order = ("domain_specific", "data_quality", "analysis", "visualization")
    ordered_keys = list(category_order) + [k for k in recs.keys() if k not in category_order and k not in ("items", "summary", "total_count")]

    for cat in ordered_keys:
        bucket = recs.get(cat)
        if not isinstance(bucket, list):
            continue
        for entry in bucket:
            if isinstance(entry, dict):
                title = str(entry.get("title") or entry.get("recommendation") or entry.get("action") or "")
                dedup_key = f"{cat}:{title}:{entry.get('column', '')}"
                if dedup_key in seen_titles:
                    continue
                seen_titles.add(dedup_key)
                item = dict(entry)
                item.setdefault("category", cat.replace("_", " ").title())
                flattened.append(item)
            elif isinstance(entry, str) and entry.strip():
                flattened.append({"title": entry.strip(), "description": "", "category": cat.replace("_", " ").title()})
    return flattened


def extract_benford_summary(analysis: dict[str, Any] | None) -> dict[str, Any] | None:
    """Extract real Benford analysis findings stored under confidence_scores or top-level benford."""
    if not isinstance(analysis, dict):
        return None
    existing = analysis.get("benford")
    if isinstance(existing, dict) and existing.get("columns"):
        return existing

    conf = analysis.get("confidence_scores")
    if not isinstance(conf, dict):
        return None

    cols = conf.get("columns", [])
    if isinstance(cols, dict):
        iterable = [{"column": k, **v} for k, v in cols.items() if isinstance(v, dict)]
    elif isinstance(cols, list):
        iterable = [c for c in cols if isinstance(c, dict)]
    else:
        return None

    benford_cols: list[dict[str, Any]] = []
    for col_entry in iterable:
        ben = col_entry.get("benford_analysis")
        if not isinstance(ben, dict) or not ben:
            continue
        if ben.get("applicable") is False:
            continue
        chi_sq = ben.get("chi_square")
        if chi_sq is None and ben.get("mad") is None and not ben.get("deviation_level"):
            continue
        benford_cols.append({
            "column": col_entry.get("column", "—"),
            "chi_square": round(float(chi_sq), 2) if isinstance(chi_sq, (int, float)) else chi_sq,
            "p_value": ben.get("p_value"),
            "mad": round(float(ben["mad"]), 4) if isinstance(ben.get("mad"), (int, float)) else ben.get("mad"),
            "deviation_level": ben.get("deviation_level"),
            "suspicious": bool(ben.get("suspicious") or ben.get("is_anomalous") or ben.get("non_conforming")),
        })

    if not benford_cols:
        return None
    return {"threshold": 15.51, "columns": benford_cols}


def extract_symbolic_equations(analysis: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Extract discovered symbolic equations from smart_schema or top-level analysis."""
    if not isinstance(analysis, dict):
        return []
    raw = analysis.get("symbolic_equations")
    if not raw and isinstance(analysis.get("smart_schema"), dict):
        raw = analysis["smart_schema"].get("symbolic_equations")
    if not isinstance(raw, list):
        return []

    normalized: list[dict[str, Any]] = []
    for eq in raw:
        if not isinstance(eq, dict):
            continue
        formula = eq.get("equation") or eq.get("formula") or ""
        if not formula:
            continue
        item = dict(eq)
        item["equation"] = formula
        item["formula"] = formula
        normalized.append(item)
    return normalized


def extract_time_series_entries(analysis: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Extract normalized time-series trend entries without fabricating defaults."""
    if not isinstance(analysis, dict):
        return []
    existing = analysis.get("time_series")
    if isinstance(existing, list) and existing:
        return [e for e in existing if isinstance(e, dict)]

    tsa = analysis.get("time_series_analysis")
    if not isinstance(tsa, dict) or not tsa:
        return []
    if tsa.get("has_time_series") is False:
        return []

    entries: list[dict[str, Any]] = []
    date_col = tsa.get("time_column") or tsa.get("date_column") or ""
    trends = tsa.get("trends") or tsa.get("analyses") or {}
    if isinstance(trends, dict):
        for col_name, tdata in trends.items():
            if not isinstance(tdata, dict):
                continue
            entries.append({
                "column": col_name,
                "time_column": date_col or col_name,
                "method": tdata.get("method") or "Mann-Kendall / Sen's Slope",
                "direction": tdata.get("direction") or tdata.get("trend") or "—",
                "slope": tdata.get("slope") if tdata.get("slope") is not None else tdata.get("sen_slope"),
                "p_value": tdata.get("p_value"),
                "r_squared": tdata.get("r_squared"),
                "z_score": tdata.get("z_score") if tdata.get("z_score") is not None else tdata.get("z_stat"),
                "notes": tdata.get("interpretation") or tdata.get("notes") or (
                    f"Trend over {date_col}: {tdata.get('direction', 'detected')}"
                ),
            })
    elif isinstance(trends, list):
        for tdata in trends:
            if isinstance(tdata, dict):
                entries.append(tdata)
    return entries


def extract_ml_summary(analysis: dict[str, Any] | None) -> dict[str, Any] | None:
    """Bridge supervised_learning / executive_ml into normalized AutoML view."""
    if not isinstance(analysis, dict):
        return None
    existing = analysis.get("automl")
    if isinstance(existing, dict) and existing:
        return existing

    exec_ml = analysis.get("executive_ml")
    sup = analysis.get("supervised_learning")
    src = exec_ml if isinstance(exec_ml, dict) and exec_ml else sup
    if not isinstance(src, dict) or not src:
        return None
    if src.get("skipped") or src.get("status") == "skipped":
        return None

    target = src.get("target") or src.get("target_column")
    if not target and isinstance(sup, dict):
        target = sup.get("target") or sup.get("target_column")
    if not target:
        return None

    drivers_raw = src.get("drivers") or src.get("top_drivers") or src.get("feature_importance") or []
    drivers: list[dict[str, Any]] = []
    if isinstance(drivers_raw, dict):
        for feat, val in drivers_raw.items():
            drivers.append({"feature": feat, "importance": val})
    elif isinstance(drivers_raw, list):
        for item in drivers_raw:
            if isinstance(item, dict):
                drivers.append({
                    "feature": item.get("feature") or item.get("column") or item.get("name", "—"),
                    "importance": item.get("importance") if item.get("importance") is not None else item.get("score", 0),
                })

    metrics = src.get("metrics") or {}
    if not metrics and isinstance(sup, dict):
        metrics = sup.get("metrics") or {}

    return {
        "target": target,
        "task": src.get("task") or src.get("task_type") or "—",
        "model": src.get("model") or src.get("best_model") or src.get("model_name") or "—",
        "metrics": metrics,
        "drivers": drivers[:10],
    }


def extract_unsupervised_clusters(analysis: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Bridge feature_engineering.clusters or unsupervised_learning.personas."""
    if not isinstance(analysis, dict):
        return []
    fe = analysis.get("feature_engineering")
    if isinstance(fe, dict) and isinstance(fe.get("clusters"), list) and fe["clusters"]:
        return [c for c in fe["clusters"] if isinstance(c, dict)]

    unsup = analysis.get("unsupervised_learning")
    if not isinstance(unsup, dict):
        return []
    personas = unsup.get("personas") or unsup.get("clusters") or []
    if not isinstance(personas, list):
        return []
    normalized: list[dict[str, Any]] = []
    for p in personas:
        if isinstance(p, dict):
            share_val = p.get("share") if p.get("share") is not None else (p.get("pct") if p.get("pct") is not None else p.get("percentage", 0.0))
            desc_val = p.get("description") or p.get("summary") or p.get("profile") or ""
            normalized.append({
                "name": p.get("name") or p.get("label") or f"Cluster {p.get('cluster_id', len(normalized))}",
                "size": p.get("size") or p.get("count") or 0,
                "share": share_val,
                "pct": share_val,
                "description": desc_val,
                "summary": desc_val,
            })
    return normalized


def extract_triage_summary(cleaning_report: dict[str, Any] | None) -> dict[str, int]:
    """Map row_intelligence reason codes and legacy aliases safely with zero fake defaults."""
    if not isinstance(cleaning_report, dict):
        return {}
    rt = cleaning_report.get("row_triage")
    if not isinstance(rt, dict):
        return {}
    reasons = (rt.get("summary") or {}).get("by_reason") or rt.get("by_reason") or {}
    if not isinstance(reasons, dict):
        reasons = {}

    blank = int(reasons.get("blank_row", 0) or 0) + int(reasons.get("fully_blank_rows", 0) or 0)
    ws = int(reasons.get("whitespace_only", 0) or 0) + int(reasons.get("whitespace_only_rows", 0) or 0)
    rep_hdr = int(reasons.get("repeated_header", 0) or 0) + int(reasons.get("repeated_mid_file_header_rows", 0) or 0)
    subtotals = int(reasons.get("summary_row", 0) or 0) + int(reasons.get("subtotal_summary_rows", 0) or 0)
    grand_tot = int(reasons.get("grand_total", 0) or 0) + int(reasons.get("grand_total_rows", 0) or 0)
    footer = int(reasons.get("footer_note", 0) or 0) + int(reasons.get("sparse_footer_rows", 0) or 0)
    exact_dup = int(reasons.get("exact_duplicate", 0) or 0) + int(reasons.get("near_duplicate_rows", 0) or 0)
    sparse = int(reasons.get("sparse_row", 0) or 0)
    dups_removed = int(cleaning_report.get("duplicate_rows_removed", 0) or 0)
    non_data = int(cleaning_report.get("non_data_rows_dropped", 0) or 0)

    return {
        "duplicate_rows_removed": dups_removed or exact_dup,
        "blank_whitespace_rows": blank + ws,
        "repeated_headers": rep_hdr,
        "subtotal_grand_total_rows": subtotals + grand_tot,
        "footer_rows": footer,
        "sparse_rows": sparse,
        "non_data_rows_dropped": non_data,
        "exact_duplicate": exact_dup,
    }


def extract_before_after_metrics(
    cleaning_report: dict[str, Any] | None,
    comparison_report: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return real before/after comparison metrics if available (never synthetic numbers)."""
    if isinstance(cleaning_report, dict):
        ba = cleaning_report.get("before_after")
        if isinstance(ba, list) and ba:
            return [m for m in ba if isinstance(m, dict)]

    if not isinstance(comparison_report, dict):
        return []
    summary = comparison_report.get("summary")
    if not isinstance(summary, dict) or not summary:
        return []

    rows_b = summary.get("rows_before")
    rows_a = summary.get("rows_after")
    metrics: list[dict[str, Any]] = []
    if rows_b is not None and rows_a is not None:
        metrics.append({
            "metric": "Row Count",
            "before": rows_b,
            "after": rows_a,
            "delta": int(rows_a) - int(rows_b),
        })
    cols_b = summary.get("columns_before")
    cols_a = summary.get("columns_after")
    if cols_b is not None and cols_a is not None:
        metrics.append({
            "metric": "Column Count",
            "before": cols_b,
            "after": cols_a,
            "delta": int(cols_a) - int(cols_b),
        })
    nulls_b = summary.get("missing_before") if summary.get("missing_before") is not None else summary.get("nulls_before")
    nulls_a = summary.get("missing_after") if summary.get("missing_after") is not None else summary.get("nulls_after")
    if nulls_b is not None and nulls_a is not None:
        metrics.append({
            "metric": "Missing Cells",
            "before": nulls_b,
            "after": nulls_a,
            "delta": int(nulls_a) - int(nulls_b),
        })
    comp_b = summary.get("completeness_before")
    comp_a = summary.get("completeness_after")
    if comp_b is not None and comp_a is not None:
        metrics.append({
            "metric": "Completeness (%)",
            "before": round(float(comp_b), 2),
            "after": round(float(comp_a), 2),
            "delta": round(float(comp_a) - float(comp_b), 2),
        })
    return metrics


def extract_security_scan(analysis: dict[str, Any] | None) -> dict[str, Any] | None:
    """Bridge real text_hazards into security_scan findings without synthetic counts."""
    if not isinstance(analysis, dict):
        return None
    existing = analysis.get("security_scan")
    if isinstance(existing, dict) and existing:
        return existing

    hazards = analysis.get("text_hazards")
    if not isinstance(hazards, dict) or not hazards:
        return None

    findings = hazards.get("findings") or hazards.get("summary") or hazards.get("by_type") or {}
    if isinstance(findings, list) and findings:
        return {"findings": findings, "total": len(findings)}
    if isinstance(findings, dict) and findings:
        rows: list[dict[str, Any]] = []
        total_cnt = 0
        for vec_name, count in findings.items():
            if not isinstance(count, (int, float)) or count <= 0:
                continue
            cnt = int(count)
            total_cnt += cnt
            sev = "High" if any(k in str(vec_name).lower() for k in ("xss", "sql", "formula", "cwe")) else "Medium"
            rows.append({
                "vector": str(vec_name).replace("_", " ").title(),
                "detections": cnt,
                "severity": sev,
                "recommendation": "Inspect flagged values and apply output escaping / formula sanitization",
            })
        if rows:
            return {"findings": rows, "total": hazards.get("total_hazards", total_cnt)}
    return None


def normalize_analysis_for_reports(
    analysis_results: dict[str, Any],
    job_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create a normalized shallow copy of analysis_results where canonical pipeline outputs
    are mapped cleanly for Typst, WeasyPrint, and ReportLab without inventing data.
    """
    data = dict(analysis_results or {})

    # 1. Benford
    benford = extract_benford_summary(data)
    if benford:
        data["benford"] = benford

    # 2. Symbolic equations
    sym_eqs = extract_symbolic_equations(data)
    if sym_eqs:
        data["symbolic_equations"] = sym_eqs

    # 3. Time series normalized list + preserved dict
    ts_entries = extract_time_series_entries(data)
    if ts_entries and not data.get("time_series"):
        data["time_series"] = ts_entries
    tsa = data.get("time_series_analysis")
    if isinstance(tsa, dict) and "analyses" not in tsa and isinstance(tsa.get("trends"), dict):
        tsa_copy = dict(tsa)
        seas_map = tsa.get("seasonality") if isinstance(tsa.get("seasonality"), dict) else {}
        tsa_copy["analyses"] = {
            col_name: {
                "trend": trend_val if isinstance(trend_val, dict) else {},
                "seasonality": seas_map.get(col_name, {}) if isinstance(seas_map.get(col_name), dict) else {},
            }
            for col_name, trend_val in tsa["trends"].items()
        }
        data["time_series_analysis"] = tsa_copy

    # 4. AutoML
    automl = extract_ml_summary(data)
    if automl:
        data["automl"] = automl

    # 5. Unsupervised personas -> feature_engineering.clusters
    clusters = extract_unsupervised_clusters(data)
    if clusters:
        fe = dict(data.get("feature_engineering") or {})
        if not fe.get("clusters"):
            fe["clusters"] = clusters
        data["feature_engineering"] = fe

    # 6. Security scan & PII
    sec = extract_security_scan(data)
    if sec:
        data["security_scan"] = sec
    raw_pii = data.get("pii") or (job_result.get("pii") if isinstance(job_result, dict) else None)
    if isinstance(raw_pii, dict) and raw_pii:
        findings = raw_pii.get("findings")
        if isinstance(findings, dict):
            pii_list = []
            for col_name, info in findings.items():
                if not isinstance(info, dict):
                    continue
                conf_label = str(info.get("confidence", "High")).title()
                sample_masked = info.get("sample_masked")
                policy_text = info.get("action_required") or "Mask or exclude from modeling"
                if sample_masked:
                    policy_text = f"{policy_text} (sample: {sample_masked})"
                pii_list.append({
                    "entity": info.get("pii_type") or info.get("entity") or "PII",
                    "column": info.get("column") or col_name,
                    "risk": "Critical" if info.get("pii_type") in ("SSN", "CREDIT_CARD") else conf_label,
                    "policy": policy_text,
                })
            if pii_list:
                data["pii"] = {**raw_pii, "findings": pii_list}
        elif isinstance(findings, list) and findings:
            data["pii"] = raw_pii

    # 7. Recommendations (preserve dict keys while adding flat items)
    recs = data.get("recommendations")
    flat_recs = flatten_recommendations(recs)
    if isinstance(recs, dict) and flat_recs and "items" not in recs:
        recs_copy = dict(recs)
        recs_copy["items"] = flat_recs
        data["recommendations"] = recs_copy
    elif isinstance(recs, list) and flat_recs:
        data["recommendations"] = {"items": flat_recs}

    # 8. Cleaning before/after + triage
    cr = data.get("cleaning_report")
    comp = data.get("comparison_report") or (job_result.get("comparison_report") if isinstance(job_result, dict) else None)
    if isinstance(cr, dict):
        cr_copy = dict(cr)
        ba = extract_before_after_metrics(cr_copy, comp)
        if ba and not cr_copy.get("before_after"):
            cr_copy["before_after"] = ba
        triage_summary = extract_triage_summary(cr_copy)
        if triage_summary and not cr_copy.get("triage_normalized"):
            cr_copy["triage_normalized"] = triage_summary
        data["cleaning_report"] = cr_copy

    # 9. Missing patterns alias for templates expecting `columns`
    mp = data.get("missing_patterns")
    if isinstance(mp, dict) and "columns" not in mp and isinstance(mp.get("column_details"), dict):
        mp_copy = dict(mp)
        inferred = mp.get("inferred_pattern") or mp.get("pattern") or "—"
        mp_copy["columns"] = {
            col: {
                "missing_count": det.get("count", 0),
                "missing_pct": det.get("percentage", 0.0),
                "pattern": det.get("pattern") or inferred,
                **det,
            }
            for col, det in mp["column_details"].items()
            if isinstance(det, dict)
        }
        data["missing_patterns"] = mp_copy

    # 10. Semantic analysis aliases
    sem = data.get("semantic_analysis")
    if isinstance(sem, dict):
        sem_copy = dict(sem)
        if "suggested_pairs" not in sem_copy and isinstance(sem_copy.get("suggested_analysis"), list):
            sem_copy["suggested_pairs"] = sem_copy["suggested_analysis"]
        roles = sem_copy.get("column_roles")
        if isinstance(roles, dict):
            sem_copy["column_roles"] = {
                col: {
                    "semantic_type": r.get("semantic_type") or r.get("role") or "—",
                    **r,
                }
                for col, r in roles.items()
                if isinstance(r, dict)
            }
        data["semantic_analysis"] = sem_copy

    # 11. Smart schema aliases (type_candidates / type_corrections & relationships)
    ss = data.get("smart_schema")
    if isinstance(ss, dict):
        ss_copy = dict(ss)
        raw_tc = ss_copy.get("type_corrections") or ss_copy.get("type_candidates")
        if isinstance(raw_tc, list):
            normalized_tc = [
                {
                    "current": tc.get("current") or tc.get("current_type") or "—",
                    "suggested": tc.get("suggested") or tc.get("suggested_type") or "—",
                    "current_type": tc.get("current_type") or tc.get("current") or "—",
                    "suggested_type": tc.get("suggested_type") or tc.get("suggested") or "—",
                    **tc,
                }
                for tc in raw_tc
                if isinstance(tc, dict)
            ]
            ss_copy["type_corrections"] = normalized_tc
            ss_copy["type_candidates"] = normalized_tc
        if isinstance(ss_copy.get("relationships"), list):
            ss_copy["relationships"] = [
                {
                    "from": rel.get("from") or rel.get("source") or "—",
                    "to": rel.get("to") or rel.get("target") or "—",
                    "source": rel.get("source") or rel.get("from") or "—",
                    "target": rel.get("target") or rel.get("to") or "—",
                    **rel,
                }
                for rel in ss_copy["relationships"]
                if isinstance(rel, dict)
            ]
        data["smart_schema"] = ss_copy

    # 12. Analysis decisions aliases
    dec = data.get("analysis_decisions")
    if isinstance(dec, dict) and isinstance(dec.get("decisions"), list):
        dec_copy = dict(dec)
        dec_copy["decisions"] = [
            {
                "analysis": d.get("analysis") or d.get("module") or d.get("name") or "—",
                "module": d.get("module") or d.get("analysis") or d.get("name") or "—",
                "decision": d.get("decision") or d.get("status") or "—",
                "status": d.get("status") or d.get("decision") or "—",
                **d,
            }
            for d in dec_copy["decisions"]
            if isinstance(d, dict)
        ]
        data["analysis_decisions"] = dec_copy

    return data
