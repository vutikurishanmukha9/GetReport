import re
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class QueryRouter:
    """
    LightRAG-inspired Dual-Level Query Router & Keyword Extractor.
    Decomposes user queries into:
    - Low-Level Keywords (ll_keywords): Exact column names, specific metrics, issue IDs.
    - High-Level Keywords (hl_keywords): Thematic concepts, dataset health, correlation patterns, governance.
    - Query Type: Dispatches to DuckDB OLAP, Local 1-Hop Graph Traversal, Global Thematic Graph, or Vector Chunks.
    """
    MATH_KEYWORDS = [
        "average", "avg", "mean", "median", "sum", "total", "count",
        "maximum", "max", "highest", "minimum", "min", "lowest",
        "std", "standard deviation", "variance", "quantile", "percentile",
        "ratio", "percentage", "proportion"
    ]
    
    RELATIONAL_KEYWORDS = [
        "correlation", "correlate", "correlated", "relationship", "relation",
        "depend", "dependency", "association", "linked", "connected", "influence",
        "impact", "affect", "cause", "why", "root cause", "pair", "feature"
    ]

    QUALITY_KEYWORDS = [
        "quality", "grade", "confidence", "clean", "cleaned", "cleaning",
        "issue", "issues", "alert", "alerts", "null", "missing", "nan",
        "duplicate", "duplicates", "outlier", "outliers", "anomaly", "anomalies",
        "benford", "forensic", "audit", "action", "remediation", "fix"
    ]

    GLOBAL_THEME_KEYWORDS = [
        "overall", "summary", "overview", "health", "dataset", "profile",
        "recommendation", "recommendations", "trend", "trends", "conclusion",
        "key takeaways", "insights", "general"
    ]

    def __init__(self, llm_client: Optional[Any] = None, model: Optional[str] = None):
        self.client = llm_client
        self.model = model

    def _extract_column_matches(self, query: str, available_columns: List[str]) -> List[str]:
        """Find columns mentioned in the query (case-insensitive, normalized spaces and underscores)."""
        if not available_columns:
            return []

        q_low = query.lower()
        matched = []
        for col in available_columns:
            clean_col = str(col).strip()
            if not clean_col:
                continue
            col_low = clean_col.lower()
            col_spaces = col_low.replace("_", " ")

            # Word boundary check or substring match
            pattern = rf"\b{re.escape(col_low)}\b|\b{re.escape(col_spaces)}\b"
            if re.search(pattern, q_low) or col_low in q_low or col_spaces in q_low:
                matched.append(clean_col)

        return list(dict.fromkeys(matched))

    async def route_query(
        self,
        query: str,
        available_columns: Optional[List[str]] = None,
        timeout: float = 2.0
    ) -> Dict[str, Any]:
        """
        Decomposes query into search parameters and dispatch flags.
        Fast, hybrid deterministic parser with LLM enhancement when available.
        """
        cols = available_columns or []
        q_low = query.lower().strip()
        
        # 1. Detect matching columns
        detected_cols = self._extract_column_matches(query, cols)

        # 2. Build Low-Level Keywords (ll_keywords)
        ll_keywords = list(detected_cols)
        for w in re.findall(r"\w+", q_low):
            if len(w) >= 3 and (w in cols or any(w in c.lower() for c in cols)):
                if w not in ll_keywords:
                    ll_keywords.append(w)

        # 3. Build High-Level Keywords (hl_keywords)
        hl_keywords = []
        for kw in self.RELATIONAL_KEYWORDS:
            if re.search(rf"\b{re.escape(kw)}\b", q_low):
                hl_keywords.append(kw)
        for kw in self.QUALITY_KEYWORDS:
            if re.search(rf"\b{re.escape(kw)}\b", q_low):
                hl_keywords.append(kw)
        for kw in self.GLOBAL_THEME_KEYWORDS:
            if re.search(rf"\b{re.escape(kw)}\b", q_low):
                hl_keywords.append(kw)

        # 4. Check for Exact Mathematical Query -> DuckDB OLAP
        has_math_op = any(re.search(rf"\b{re.escape(op)}\b", q_low) for op in self.MATH_KEYWORDS)
        duckdb_needed = has_math_op and (len(detected_cols) > 0 or "all" in q_low or "every" in q_low or "dataset" in q_low)

        # 5. Determine Primary Strategy
        is_relational = any(re.search(rf"\b{re.escape(kw)}\b", q_low) for kw in self.RELATIONAL_KEYWORDS)
        is_quality = any(re.search(rf"\b{re.escape(kw)}\b", q_low) for kw in self.QUALITY_KEYWORDS)
        is_global = any(re.search(rf"\b{re.escape(kw)}\b", q_low) for kw in self.GLOBAL_THEME_KEYWORDS) or (not detected_cols and not has_math_op)

        if duckdb_needed:
            query_type = "analytical_sql"
        elif (is_relational or is_quality) and detected_cols:
            query_type = "graph_local"
        elif is_relational or is_quality or is_global:
            query_type = "graph_global"
        else:
            query_type = "hybrid"

        # Fallback keyword defaults if empty
        if not hl_keywords:
            if is_quality:
                hl_keywords = ["quality alert", "data health"]
            elif is_relational:
                hl_keywords = ["correlation", "dependencies"]
            else:
                hl_keywords = ["overview", "summary"]

        result = {
            "query_type": query_type,
            "ll_keywords": ll_keywords,
            "hl_keywords": list(dict.fromkeys(hl_keywords)),
            "duckdb_sql_needed": duckdb_needed,
            "detected_columns": detected_cols,
        }

        # If LLM is available and query is ambiguous, we can refine via fast prompt
        if self.client and self.model and not duckdb_needed and len(q_low) > 30:
            try:
                import asyncio
                llm_task = self._refine_with_llm(query, cols)
                llm_res = await asyncio.wait_for(llm_task, timeout=timeout)
                if llm_res:
                    # Merge LLM results with deterministic flags
                    result["ll_keywords"] = list(dict.fromkeys(result["ll_keywords"] + llm_res.get("ll_keywords", [])))
                    result["hl_keywords"] = list(dict.fromkeys(result["hl_keywords"] + llm_res.get("hl_keywords", [])))
                    if "query_type" in llm_res and result["query_type"] == "hybrid":
                        result["query_type"] = llm_res["query_type"]
            except Exception as e:
                logger.debug(f"LLM query routing refinement skipped or timed out: {e}")

        logger.info(
            f"Routed query '{query[:50]}' -> type={result['query_type']}, "
            f"ll_kws={result['ll_keywords']}, hl_kws={result['hl_keywords']}, duckdb={duckdb_needed}"
        )
        return result

    async def _refine_with_llm(self, query: str, cols: List[str]) -> Optional[Dict[str, Any]]:
        """Optional fast LLM query decomposition."""
        if not self.client or not self.model:
            return None
        prompt = (
            f"You are a query parser for an analytics database.\n"
            f"Available dataset columns: {cols[:20]}\n"
            f"User query: \"{query}\"\n"
            f"Return JSON strictly formatted as: {{\"ll_keywords\": [string], \"hl_keywords\": [string], \"query_type\": \"graph_local\"|\"graph_global\"|\"hybrid\"}}"
        )
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=150,
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content
            return json.loads(content)
        except Exception:
            return None
