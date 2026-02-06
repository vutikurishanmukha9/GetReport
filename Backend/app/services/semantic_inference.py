"""
Semantic Column Intelligence Service

Rule-based semantic mapping to:
1. Detect data domain (education, sales, healthcare, etc.)
2. Map columns to semantic roles (identifier, metric, dimension, etc.)
3. Suggest meaningful analysis pairs based on domain knowledge
4. Handle naming variations through fuzzy pattern matching
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN KEYWORD PATTERNS
# Used to detect the overall domain/context of the dataset
# ═══════════════════════════════════════════════════════════════════════════════

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "education": [
        "student", "stud", "pupil", "learner",
        "marks", "score", "grade", "gpa", "cgpa", "percentage",
        "subject", "course", "class", "section", "semester", "term",
        "attendance", "present", "absent",
        "teacher", "faculty", "professor", "instructor",
        "exam", "test", "quiz", "assignment", "homework",
        "roll", "enrollment", "admission",
        "school", "college", "university", "institute",
    ],
    "sales_ecommerce": [
        "order", "purchase", "transaction", "sale",
        "customer", "cust", "buyer", "client",
        "product", "item", "sku", "upc", "ean",
        "price", "cost", "revenue", "profit", "margin", "discount",
        "quantity", "qty", "units", "amount",
        "cart", "basket", "checkout", "shipping",
        "invoice", "receipt", "bill",
        "store", "shop", "warehouse", "inventory",
    ],
    "hr_employee": [
        "employee", "emp", "staff", "worker", "personnel",
        "salary", "wage", "compensation", "pay", "ctc", "bonus",
        "department", "dept", "division", "team", "unit",
        "designation", "title", "position", "role", "job",
        "hire", "joining", "onboard", "termination", "resign",
        "manager", "supervisor", "reporting",
        "leave", "vacation", "pto", "sick",
        "performance", "rating", "review", "appraisal",
    ],
    "healthcare": [
        "patient", "medical", "health", "clinical",
        "diagnosis", "disease", "condition", "symptom",
        "doctor", "physician", "nurse", "specialist",
        "prescription", "medicine", "drug", "dosage",
        "hospital", "clinic", "ward", "admission",
        "blood", "bp", "pulse", "temperature", "weight", "height", "bmi",
        "lab", "test", "report", "xray", "scan", "mri",
        "insurance", "claim", "policy",
    ],
    "finance": [
        "account", "acct", "balance", "ledger",
        "transaction", "transfer", "payment", "deposit", "withdrawal",
        "credit", "debit", "loan", "interest", "emi",
        "invoice", "bill", "receipt", "voucher",
        "bank", "branch", "ifsc", "swift",
        "stock", "share", "equity", "dividend", "portfolio",
        "tax", "gst", "vat", "tds",
        "budget", "expense", "income", "revenue", "profit", "loss",
    ],
    "iot_sensor": [
        "sensor", "device", "node", "module",
        "temperature", "temp", "humidity", "pressure", "moisture",
        "reading", "measurement", "value", "data",
        "timestamp", "datetime", "time", "date",
        "latitude", "longitude", "location", "gps",
        "battery", "voltage", "current", "power",
        "status", "state", "online", "offline",
        "alert", "alarm", "threshold", "warning",
    ],
    "logistics": [
        "shipment", "shipping", "delivery", "dispatch",
        "tracking", "consignment", "parcel", "package",
        "warehouse", "inventory", "stock", "storage",
        "origin", "destination", "route", "distance",
        "carrier", "courier", "logistics", "transport",
        "weight", "dimension", "volume", "size",
        "eta", "arrival", "departure", "transit",
    ],
}

# ═══════════════════════════════════════════════════════════════════════════════
# COLUMN ROLE PATTERNS
# Used to classify each column into a semantic role
# ═══════════════════════════════════════════════════════════════════════════════

COLUMN_ROLE_PATTERNS: dict[str, list[str]] = {
    # Identifiers - exclude from numerical analysis
    "identifier": [
        "id", "uuid", "guid", "key", "code", "number", "num", "no",
        "index", "idx", "serial", "sno", "srno",
        "roll", "enrollment", "registration", "reference", "ref",
        "sku", "upc", "ean", "barcode", "isbn",
        "account", "acct", "policy", "ticket", "case",
    ],
    
    # Names/Labels - exclude from numerical analysis
    "name_label": [
        "name", "title", "label", "description", "desc", "remarks", "notes",
        "first", "last", "middle", "full",  # first_name, last_name
        "address", "street", "city", "state", "country", "zip", "pincode", "postal",
        "email", "phone", "mobile", "tel", "fax", "contact",
    ],
    
    # Date/Time - handle specially
    "datetime": [
        "date", "time", "timestamp", "datetime", "dt",
        "created", "updated", "modified", "deleted",
        "start", "end", "begin", "finish",
        "dob", "birth", "joining", "hire", "admission",
        "year", "month", "day", "week", "quarter",
    ],
    
    # Target Metrics - primary analysis focus
    "target_metric": [
        "total", "sum", "aggregate", "overall", "final",
        "marks", "score", "grade", "points", "rating",
        "revenue", "profit", "sales", "income", "earning",
        "performance", "result", "outcome",
    ],
    
    # Predictor/Feature Metrics - correlate with target
    "predictor_metric": [
        "attendance", "present", "absent", "leave",
        "experience", "tenure", "age", "years",
        "hours", "duration", "time_spent",
        "distance", "quantity", "count",
    ],
    
    # Percentage Metrics - values 0-100
    "percentage_metric": [
        "percent", "pct", "percentage", "ratio", "rate",
        "accuracy", "precision", "recall", "f1",
        "growth", "change", "variance", "deviation",
    ],
    
    # Financial Metrics
    "financial_metric": [
        "price", "cost", "amount", "value", "fee", "charge",
        "salary", "wage", "pay", "compensation", "bonus",
        "discount", "tax", "gst", "vat",
        "balance", "credit", "debit",
    ],
    
    # Categorical Dimensions - for grouping
    "dimension": [
        "category", "type", "class", "group", "segment",
        "department", "division", "team", "unit",
        "region", "zone", "area", "territory",
        "channel", "source", "medium", "platform",
        "status", "state", "stage", "phase",
        "gender", "sex", "age_group",
    ],
    
    # Boolean/Binary
    "boolean": [
        "is_", "has_", "can_", "should_", "was_", "will_",
        "active", "enabled", "disabled", "deleted", "verified",
        "flag", "indicator",
    ],
}

# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN-SPECIFIC ANALYSIS SUGGESTIONS
# Recommends which column pairs to correlate based on domain
# ═══════════════════════════════════════════════════════════════════════════════

DOMAIN_ANALYSIS_PAIRS: dict[str, list[tuple[str, str, str]]] = {
    "education": [
        ("attendance", "marks", "Attendance often predicts academic performance"),
        ("attendance", "grade", "Higher attendance typically correlates with better grades"),
        ("subject", "marks", "Subject-wise performance breakdown"),
        ("gender", "marks", "Gender-based performance analysis"),
    ],
    "sales_ecommerce": [
        ("quantity", "revenue", "Volume vs Revenue relationship"),
        ("discount", "quantity", "Impact of discounts on sales volume"),
        ("category", "revenue", "Revenue by product category"),
        ("region", "sales", "Regional sales performance"),
    ],
    "hr_employee": [
        ("experience", "salary", "Experience vs Compensation"),
        ("department", "salary", "Salary distribution by department"),
        ("performance", "salary", "Performance rating vs Pay"),
        ("tenure", "performance", "Tenure impact on performance"),
    ],
    "healthcare": [
        ("age", "diagnosis", "Age-related health patterns"),
        ("bmi", "condition", "BMI correlation with conditions"),
        ("gender", "diagnosis", "Gender-based health patterns"),
    ],
    "finance": [
        ("amount", "category", "Spending by category"),
        ("date", "balance", "Balance trend over time"),
        ("type", "amount", "Transaction type analysis"),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ColumnRole:
    """Semantic role assigned to a column."""
    column: str
    role: str
    confidence: float  # 0-1
    matched_patterns: list[str] = field(default_factory=list)

@dataclass
class DomainDetection:
    """Result of domain detection."""
    primary_domain: str
    confidence: float
    matched_keywords: list[str] = field(default_factory=list)
    alternative_domains: list[tuple[str, float]] = field(default_factory=list)

@dataclass
class AnalysisSuggestion:
    """Suggested analysis pair."""
    column_a: str
    column_b: str
    reason: str
    priority: int  # 1=high, 2=medium, 3=low

@dataclass 
class SemanticAnalysis:
    """Complete semantic analysis result."""
    domain: DomainDetection
    column_roles: dict[str, ColumnRole]
    suggested_pairs: list[AnalysisSuggestion]
    analytical_columns: list[str]
    identifier_columns: list[str]
    dimension_columns: list[str]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": {
                "primary": self.domain.primary_domain,
                "confidence": round(self.domain.confidence, 2),
                "matched_keywords": self.domain.matched_keywords[:10],
            },
            "column_roles": {
                col: {"role": role.role, "confidence": round(role.confidence, 2)}
                for col, role in self.column_roles.items()
            },
            "suggested_analysis": [
                {"columns": [s.column_a, s.column_b], "reason": s.reason, "priority": s.priority}
                for s in self.suggested_pairs[:5]
            ],
            "analytical_columns": self.analytical_columns,
            "identifier_columns": self.identifier_columns,
            "dimension_columns": self.dimension_columns,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_column_name(name: str) -> str:
    """Normalize column name for matching: lowercase, remove special chars."""
    # Convert camelCase to snake_case
    name = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)
    # Lowercase and remove non-alphanumeric (keep underscores)
    name = re.sub(r'[^a-z0-9_]', '_', name.lower())
    # Remove multiple underscores
    name = re.sub(r'_+', '_', name).strip('_')
    return name

def _fuzzy_match(text: str, patterns: list[str]) -> list[str]:
    """Check if any pattern matches (as substring or word boundary)."""
    text = _normalize_column_name(text)
    matches = []
    for pattern in patterns:
        pattern = pattern.lower()
        # Exact match or word boundary match
        if pattern in text or text.startswith(pattern) or text.endswith(pattern):
            matches.append(pattern)
        # Also check if pattern appears as a word
        elif re.search(rf'\b{re.escape(pattern)}\b', text):
            matches.append(pattern)
    return matches


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DETECTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def detect_domain(df: pl.DataFrame) -> DomainDetection:
    """
    Detect the primary domain of the dataset by analyzing column names.
    
    Returns:
        DomainDetection with primary domain, confidence, and matched keywords.
    """
    all_columns = " ".join(df.columns).lower()
    
    domain_scores: dict[str, tuple[int, list[str]]] = {}
    
    for domain, keywords in DOMAIN_KEYWORDS.items():
        matches = []
        for kw in keywords:
            if kw.lower() in all_columns:
                matches.append(kw)
        domain_scores[domain] = (len(matches), matches)
    
    # Sort by match count
    sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1][0], reverse=True)
    
    if not sorted_domains or sorted_domains[0][1][0] == 0:
        return DomainDetection(
            primary_domain="generic",
            confidence=0.0,
            matched_keywords=[],
            alternative_domains=[]
        )
    
    top_domain, (top_count, top_matches) = sorted_domains[0]
    
    # Calculate confidence (normalized by keyword list size)
    confidence = min(top_count / 5, 1.0)  # 5+ matches = 100% confidence
    
    # Get alternatives
    alternatives = [
        (domain, count / max(top_count, 1))
        for domain, (count, _) in sorted_domains[1:3]
        if count > 0
    ]
    
    return DomainDetection(
        primary_domain=top_domain,
        confidence=confidence,
        matched_keywords=top_matches,
        alternative_domains=alternatives
    )


def map_column_role(df: pl.DataFrame, column: str) -> ColumnRole:
    """
    Determine the semantic role of a single column.
    
    Uses:
    1. Name pattern matching
    2. Value distribution analysis
    3. Data type inference
    """
    col_name = _normalize_column_name(column)
    
    best_role = "unknown"
    best_confidence = 0.0
    all_matches: list[str] = []
    
    # Check each role pattern
    for role, patterns in COLUMN_ROLE_PATTERNS.items():
        matches = _fuzzy_match(column, patterns)
        if matches:
            confidence = min(len(matches) / 2, 1.0)  # 2+ matches = 100%
            if confidence > best_confidence:
                best_role = role
                best_confidence = confidence
                all_matches = matches
    
    # Value-based heuristics if no name match
    if best_confidence < 0.5:
        try:
            dtype = df[column].dtype
            series = df[column].drop_nulls()
            
            if series.len() > 0:
                # Check if it's a percentage (0-100 range)
                if dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
                    min_val, max_val = series.min(), series.max()
                    if min_val >= 0 and max_val <= 100:
                        # Could be percentage
                        if any(p in col_name for p in ["pct", "percent", "rate", "ratio"]):
                            best_role = "percentage_metric"
                            best_confidence = 0.8
                
                # Check uniqueness ratio for identifier detection
                uniqueness = series.n_unique() / series.len()
                if uniqueness > 0.95 and dtype in (pl.Int64, pl.Int32, pl.Utf8):
                    if best_confidence < 0.7:
                        best_role = "identifier"
                        best_confidence = 0.7
                
                # Low cardinality = likely dimension
                if dtype == pl.Utf8 and series.n_unique() <= 20:
                    if best_confidence < 0.6:
                        best_role = "dimension"
                        best_confidence = 0.6
        except Exception as e:
            logger.debug(f"Value heuristics failed for {column}: {e}")
    
    return ColumnRole(
        column=column,
        role=best_role,
        confidence=best_confidence,
        matched_patterns=all_matches
    )


def suggest_analysis_pairs(
    df: pl.DataFrame,
    domain: str,
    column_roles: dict[str, ColumnRole]
) -> list[AnalysisSuggestion]:
    """
    Suggest meaningful column pairs for analysis based on domain and roles.
    """
    suggestions: list[AnalysisSuggestion] = []
    
    # Get domain-specific pairs
    domain_pairs = DOMAIN_ANALYSIS_PAIRS.get(domain, [])
    
    for pattern_a, pattern_b, reason in domain_pairs:
        # Find matching columns
        matches_a = [c for c in df.columns if pattern_a in _normalize_column_name(c)]
        matches_b = [c for c in df.columns if pattern_b in _normalize_column_name(c)]
        
        for col_a in matches_a[:2]:  # Limit to 2 matches each
            for col_b in matches_b[:2]:
                if col_a != col_b:
                    suggestions.append(AnalysisSuggestion(
                        column_a=col_a,
                        column_b=col_b,
                        reason=reason,
                        priority=1
                    ))
    
    # Generic role-based suggestions
    metrics = [c for c, r in column_roles.items() if r.role in ("target_metric", "predictor_metric", "financial_metric")]
    dimensions = [c for c, r in column_roles.items() if r.role == "dimension"]
    
    # Metric vs Dimension (breakdown analysis)
    for metric in metrics[:3]:
        for dim in dimensions[:3]:
            if not any(s.column_a == metric and s.column_b == dim for s in suggestions):
                suggestions.append(AnalysisSuggestion(
                    column_a=metric,
                    column_b=dim,
                    reason=f"Analyze {metric} by {dim}",
                    priority=2
                ))
    
    # Metric vs Metric (correlation)
    for i, m1 in enumerate(metrics[:4]):
        for m2 in metrics[i+1:4]:
            if not any(s.column_a == m1 and s.column_b == m2 for s in suggestions):
                suggestions.append(AnalysisSuggestion(
                    column_a=m1,
                    column_b=m2,
                    reason=f"Correlation between {m1} and {m2}",
                    priority=2
                ))
    
    # Sort by priority
    suggestions.sort(key=lambda x: x.priority)
    
    return suggestions[:10]  # Top 10


def analyze_semantic_structure(df: pl.DataFrame) -> SemanticAnalysis:
    """
    Main entry point: Perform complete semantic analysis of the dataset.
    
    Returns:
        SemanticAnalysis with domain, column roles, and suggestions.
    """
    # 1. Detect domain
    domain = detect_domain(df)
    logger.info(f"Detected domain: {domain.primary_domain} (confidence: {domain.confidence:.0%})")
    
    # 2. Map column roles
    column_roles: dict[str, ColumnRole] = {}
    for col in df.columns:
        column_roles[col] = map_column_role(df, col)
    
    # 3. Categorize columns
    analytical_columns = [
        c for c, r in column_roles.items() 
        if r.role in ("target_metric", "predictor_metric", "percentage_metric", "financial_metric", "unknown")
        and r.role not in ("identifier", "name_label", "datetime")
    ]
    
    identifier_columns = [
        c for c, r in column_roles.items()
        if r.role in ("identifier", "name_label")
    ]
    
    dimension_columns = [
        c for c, r in column_roles.items()
        if r.role == "dimension"
    ]
    
    # 4. Suggest analysis pairs
    suggested_pairs = suggest_analysis_pairs(df, domain.primary_domain, column_roles)
    
    logger.info(f"Column classification: {len(analytical_columns)} analytical, {len(identifier_columns)} identifiers, {len(dimension_columns)} dimensions")
    
    return SemanticAnalysis(
        domain=domain,
        column_roles=column_roles,
        suggested_pairs=suggested_pairs,
        analytical_columns=analytical_columns,
        identifier_columns=identifier_columns,
        dimension_columns=dimension_columns
    )
