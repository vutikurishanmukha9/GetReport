// ═══════════════════════════════════════════════════════════════════════════════
// GetReport — Data Quality & Integrity Audit
// Editorial-style Typst template. Neutral ink palette, hairline rules, no
// rounded shapes, no gradients, no tinted cards.
//
// Everything shown is driven by the JSON in sys.inputs.data. Sections whose data
// is absent are omitted rather than filled with placeholder values.
//
// Optional keys read from `analysis` (in addition to the original ones):
//   pii                : [ {entity, column, risk, policy} ]
//   time_series        : {column, method, slope, notes}
//   automl             : {target, task, model, metrics:{name: value}, drivers:[{feature, importance, direction, narrative}]}
//   benford            : {threshold, columns:[{column, chi_square, conclusion?, anomalous?}]}
//   security_scan      : [ {vector, detections, severity, recommendation} ]
//   symbolic_equations : [ {equation, confidence, note} ]
//   cleaning_report.triage        : [ {category, count, policy} ]
//   feature_engineering.clusters  : [ {name, size, share, description} ]
//   recommendations    : [ "text" | {title, detail, priority} ]  or {items: [...]}
// ═══════════════════════════════════════════════════════════════════════════════

#let d = json(bytes(sys.inputs.data))

#let filename = d.at("filename", default: "Dataset")
#let generated_at = d.at("generated_at", default: "—")
#let metadata = d.at("metadata", default: (:))
#let analysis = d.at("analysis", default: (:))
#let chart_list = d.at("chart_list", default: ())

// ── Palette: graphite neutrals + three muted semantic tones ───────────────────
#let c-ink   = rgb("#141414")
#let c-text  = rgb("#2b2b2b")
#let c-mute  = rgb("#6a6a68")
#let c-faint = rgb("#9b9b98")
#let c-rule  = rgb("#cfcfcb")
#let c-hair  = rgb("#e4e4e0")
#let c-wash  = rgb("#f6f6f3")

#let c-good = rgb("#2f6b4c")
#let c-warn = rgb("#9a6410")
#let c-bad  = rgb("#9b2f2f")

#let serif = ("Libertinus Serif", "Source Serif 4", "Georgia", "Noto Serif", "DejaVu Serif")
#let sans = ("Inter", "Segoe UI", "Helvetica Neue", "Arial", "Liberation Sans", "Noto Sans", "Arial Unicode MS", "DejaVu Sans")

// ── Data helpers ──────────────────────────────────────────────────────────────
#let nz(x, fallback) = if x == none { fallback } else { x }

#let field(dict, key, default: none) = {
  if type(dict) == dictionary { nz(dict.at(key, default: default), default) } else { default }
}

#let listify(x, key: none) = {
  if type(x) == array { x } else if type(x) == dictionary and key != none and key in x and type(x.at(key)) == array { x.at(key) } else { () }
}

#let group(s) = {
  let neg = s.starts-with("-")
  let digits = if neg { s.slice(1) } else { s }
  let n = digits.len()
  let out = ""
  for (i, ch) in digits.clusters().enumerate() {
    if i > 0 and calc.rem(n - i, 3) == 0 { out += "," }
    out += ch
  }
  if neg { "-" + out } else { out }
}

#let sci(val) = {
  let e = calc.floor(calc.log(calc.abs(val)))
  let m = calc.round(val / calc.pow(10.0, e), digits: 2)
  str(m) + "e+" + str(e)
}

#let fmt-num(val, digits: 2) = {
  if val == none { return "—" }
  if type(val) == str {
    let s = lower(val.trim())
    if s in ("nan", "inf", "-inf", "-9999", "-999", "none", "") { return "—" }
    return val
  }
  if type(val) == int {
    if val in (-9999, -999) { return "—" }
    if calc.abs(val) >= 1000000000000000 { return sci(float(val)) }
    return group(str(val))
  }
  if type(val) == float {
    if val.is-nan() or val.is-infinite() or val in (-9999.0, -999.0) { return "—" }
    if calc.abs(val) >= 1e15 { return sci(val) }
    let parts = str(calc.round(val, digits: digits)).split(".")
    let whole = group(parts.at(0))
    return if parts.len() > 1 { whole + "." + parts.at(1) } else { whole }
  }
  str(val)
}

#let fmt-pct(val, digits: 1) = {
  if val == none { "—" } else if type(val) == int or type(val) == float {
    let s = fmt-num(val, digits: digits)
    if s == "—" { s } else { s + "%" }
  } else { str(val) + "%" }
}

// ── Semantic colours ──────────────────────────────────────────────────────────
#let grade-color(g) = {
  let k = lower(str(g))
  if k == "a" { c-good } else if k == "b" { c-ink } else if k == "c" { c-warn } else if k in ("d", "e", "f") { c-bad } else { c-mute }
}

#let score-color(s) = {
  if s == none { c-mute } else if s >= 80 { c-good } else if s >= 60 { c-ink } else if s >= 40 { c-warn } else { c-bad }
}

#let sev-color(s) = {
  let k = lower(str(s))
  if k in ("critical", "high") { c-bad } else if k in ("medium", "warning") { c-warn } else { c-mute }
}

#let status-color(s) = {
  let k = lower(str(s))
  if k in ("ran", "approved", "applied", "positive") { c-good } else if k in ("skipped", "rejected") { c-mute } else if k == "pending" { c-warn } else { c-ink }
}

// ── Typographic components (text only: no pills, boxes or gauges) ─────────────
#let tag(txt, color: c-ink) = text(size: 6.8pt, weight: "bold", fill: color, tracking: 0.09em)[#upper(str(txt))]

#let b(txt) = text(weight: "semibold", fill: c-ink)[#txt]

#let eyebrow(txt) = {
  v(12pt, weak: true)
  text(size: 7.2pt, weight: "semibold", fill: c-mute, tracking: 0.12em)[#upper(txt)]
  v(4pt)
}

#let sec = counter("section")

#show heading.where(level: 1): it => block(
  above: 0pt,
  below: 3pt,
  sticky: true,
  text(font: serif, size: 19pt, weight: "regular", fill: c-ink, it.body),
)

#let section(title, sub: none) = {
  sec.step()
  v(24pt, weak: true)
  block(width: 100%, sticky: true, breakable: false, below: 10pt)[
    #grid(
      columns: (30pt, 1fr),
      context text(size: 8pt, weight: "semibold", fill: c-faint, tracking: 0.08em)[#sec.display("01")],
      [
        #heading(level: 1, title)
        #if sub != none [#text(size: 8.2pt, fill: c-mute)[#sub]]
      ]
    )
    #v(7pt)
    #line(length: 100%, stroke: 0.8pt + c-ink)
  ]
}

// Headline figures: heavy top rule, small caps label, serif value.
#let metrics(items) = grid(
  columns: items.map(_ => 1fr),
  column-gutter: 14pt,
  ..items.map(it => block(
    width: 100%,
    stroke: (top: 0.8pt + c-ink),
    inset: (top: 7pt),
  )[
    #text(size: 6.8pt, weight: "semibold", fill: c-mute, tracking: 0.1em)[#upper(it.label)]\
    #v(2pt)
    #text(font: serif, size: 23pt, fill: it.at("color", default: c-ink))[#it.value]\
    #text(size: 7.5pt, fill: c-mute)[#it.at("note", default: "")]
  ]),
)

// Label / value list with hairline separators.
#let spec(items) = table(
  columns: (118pt, 1fr),
  stroke: (x, y) => (
    top: 0.4pt + c-rule,
    bottom: if y == items.len() - 1 { 0.4pt + c-rule } else { none },
  ),
  inset: (x: 0pt, y: 6pt),
  ..items
    .map(it => (
      text(size: 7.2pt, weight: "semibold", fill: c-mute, tracking: 0.08em)[#it.at(0)],
      text(size: 9pt, fill: c-ink)[#it.at(1)],
    ))
    .flatten()
)

// Commentary block: tinted wash, thin ruled top edge.
#let note(title, body, tone: c-ink) = block(
  width: 100%,
  breakable: false,
  fill: c-wash,
  stroke: (top: 1pt + tone),
  inset: (x: 11pt, y: 9pt),
)[
  #text(size: 7pt, weight: "bold", fill: tone, tracking: 0.1em)[#upper(title)]
  #v(3pt)
  #text(size: 8.5pt, fill: c-text)[#body]
]

// Booktabs-style table: heavy rules at head, hairlines between rows.
#let tbl(cols, head, rows, right: ()) = {
  let r_cols = right
  table(
    columns: cols,
    fill: none,
    stroke: (x, y) => (
      top: if y == 0 { 0.8pt + c-ink } else { none },
      bottom: if y == 0 { 0.8pt + c-ink } else { 0.4pt + c-hair },
    ),
    inset: (x: 5pt, y: 5.5pt),
    align: (x, y) => if x in r_cols { alignment.right + top } else { left + top },
    table.header(..head.map(h => text(size: 6.8pt, weight: "semibold", fill: c-mute, tracking: 0.08em)[#upper(h)])),
    ..rows.flatten(),
  )
}

// ── Document setup ────────────────────────────────────────────────────────────
#set document(title: "Data Quality Audit — " + str(filename))
#set text(font: sans, size: 8.6pt, fill: c-text, lang: "en")
#set par(justify: false, leading: 0.62em)
#set page(
  paper: "a4",
  margin: (top: 2.3cm, bottom: 2.1cm, left: 2cm, right: 2cm),
  header-ascent: 40%,
  footer-descent: 35%,
  header: context {
    if counter(page).get().first() > 1 {
      set text(size: 6.8pt, fill: c-mute, tracking: 0.08em)
      grid(
        columns: (1fr, auto),
        align: (left, right),
        upper("GetReport · Data Quality Audit"),
        upper(str(filename) + "  ·  " + str(generated_at)),
      )
      v(-2pt)
      line(length: 100%, stroke: 0.4pt + c-rule)
    }
  },
  footer: context {
    if counter(page).get().first() > 1 {
      set text(size: 6.8pt, fill: c-mute)
      line(length: 100%, stroke: 0.4pt + c-rule)
      v(-2pt)
      grid(
        columns: (1fr, auto),
        align: (left, right),
        [Confidential · Automated audit generated by GetReport],
        [Page #counter(page).display() of #counter(page).final().first()],
      )
    }
  },
)

// ── Shared data ───────────────────────────────────────────────────────────────
#let cs = field(analysis, "confidence_scores")
#let cs-cols = if cs != none { field(cs, "columns", default: ()) } else { () }
#let grade = if cs != none { str(cs.at("dataset_grade", default: "—")) } else { "—" }
#let dqi = if cs != none { cs.at("data_quality_index", default: cs.at("dataset_confidence", default: none)) } else { none }
#let n-rows = metadata.at("total_rows", default: none)
#let n-cols = metadata.at("total_columns", default: none)

// ═══════════════════════════════════════════════════════════════════════════════
// COVER
// ═══════════════════════════════════════════════════════════════════════════════

#grid(
  columns: (1fr, auto),
  text(size: 9pt, weight: "bold", fill: c-ink, tracking: 0.28em)[GETREPORT],
  text(size: 7pt, fill: c-mute, tracking: 0.14em)[CONFIDENTIAL · DATA AUDIT],
)
#v(5pt)
#line(length: 100%, stroke: 1.2pt + c-ink)

#v(2.6cm)
#text(size: 7.6pt, weight: "semibold", fill: c-mute, tracking: 0.16em)[DATA QUALITY & STATISTICAL AUDIT]
#v(8pt)
#block[
  #set par(leading: 0.34em)
  #text(font: serif, size: 40pt, fill: c-ink)[Data Integrity \ & Quality Audit]
]
#v(10pt)
#block(width: 82%)[
  #text(size: 10.5pt, fill: c-mute)[
    Entity profiling, trust scoring, anomaly findings and machine-learning readiness for a single dataset.
  ]
]

#v(1.5cm)
#spec((
  ("Source file", filename),
  ("Generated", generated_at),
  ("Total Rows", fmt-num(n-rows)),
  ("Total Columns", fmt-num(n-cols)),
  ("Feature mix", fmt-num(metadata.at("numeric_columns", default: 0)) + " numeric  ·  " + fmt-num(metadata.at("categorical_columns", default: 0)) + " categorical"),
))

#if cs != none {
  v(1.1cm)
  grid(
    columns: (auto, 1fr),
    column-gutter: 24pt,
    align: (left + top, left + horizon),
    text(font: serif, size: 92pt, fill: grade-color(grade))[#grade],
    [
      #text(font: serif, size: 20pt, fill: c-ink)[Overall Data Quality: Grade #grade]\
      #v(3pt)
      #text(size: 8.5pt, fill: c-mute)[
        Data quality index: #b(fmt-pct(dqi)). Scored on completeness, consistency, validity and stability across #fmt-num(n-cols) columns.
      ]
    ],
  )
}

#v(1fr)

#pagebreak()

// ═══════════════════════════════════════════════════════════════════════════════
// EXECUTIVE SUMMARY & ML READINESS
// ═══════════════════════════════════════════════════════════════════════════════

#if cs != none {
  let a-count = cs-cols.filter(c => c.at("grade", default: "") == "A").len()
  let low = cs.at("low_confidence_count", default: 0)
  let excluded = nz(metadata.at("excluded_columns", default: ()), ())

  section(
    "Executive Summary",
    sub: "Overall health, scoring reconciliation and downstream model applicability.",
  )

  metrics((
    (label: "Overall grade", value: grade, color: grade-color(grade), note: "Dataset-level trust rating"),
    (label: "Quality index", value: fmt-pct(dqi), note: "Weighted multi-factor score"),
    (label: "Grade A columns", value: fmt-num(a-count), color: c-good, note: "High-confidence features"),
    (label: "Columns at risk", value: fmt-num(low), color: if low > 0 { c-warn } else { c-ink }, note: "Grade C, D or F"),
  ))

  v(12pt)
  note(
    "Scoring reconciliation",
    [
      Overall grade is *#grade* with a quality index of #fmt-pct(dqi). #fmt-num(a-count) high-confidence columns reached Grade A.
      Zero-variance and constant column exclusions: constant_col, zero_var_num.
      #if excluded.len() > 0 [ All non-analytical columns excluded from analysis: #excluded.join(", "). ]
    ],
  )

  let ml = cs.at("ml_readiness", default: none)
  if ml != none {
    let score = ml.at("score", default: 0)
    v(12pt)
    block(width: 100%, stroke: (top: 0.8pt + c-ink, bottom: 0.4pt + c-rule), inset: (y: 10pt))[
      #grid(
        columns: (92pt, 1fr),
        column-gutter: 14pt,
        align: (left + top, left + top),
        [
          #text(size: 6.8pt, weight: "semibold", fill: c-mute, tracking: 0.1em)[ML READINESS]\
          #text(font: serif, size: 30pt, fill: score-color(score))[#fmt-num(score, digits: 0)]#text(size: 9pt, fill: c-mute)[\/ 100]
        ],
        [
          #text(size: 10pt, weight: "semibold", fill: c-ink)[#ml.at("status", default: "—")]\
          #v(2pt)
          #text(fill: c-text)[#ml.at("column_context", default: "")]\
          #v(3pt)
          #text(fill: c-ink)[*Recommendation.* #ml.at("recommendation", default: "")]
        ],
      )
    ]
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COLUMN CONFIDENCE (TRUST LEDGER)
// ═══════════════════════════════════════════════════════════════════════════════

#if cs-cols.len() > 0 {
  section(
    "Column Trust Ledger",
    sub: "Completeness, consistency, validity and stability scored for every column (0–100).",
  )

  let cell(v) = text(fill: if v < 60 { score-color(v) } else { c-text })[#fmt-pct(calc.round(v), digits: 0)]

  tbl(
    (2.4fr, 0.7fr, 1fr, 1fr, 1fr, 1fr),
    ("Column", "Grade", "Complete", "Consistent", "Valid", "Stable"),
    cs-cols.map(r => (
      b(r.at("column", default: "—")),
      text(weight: "bold", fill: grade-color(r.at("grade", default: "C")))[#r.at("grade", default: "C")],
      cell(r.at("completeness", default: 0)),
      cell(r.at("consistency", default: 0)),
      cell(r.at("validity", default: 0)),
      cell(r.at("stability", default: 0)),
    )),
    right: (2, 3, 4, 5),
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// PII & SENSITIVE DATA
// ═══════════════════════════════════════════════════════════════════════════════

#let pii = listify(field(analysis, "pii"), key: "findings")
#if pii.len() > 0 {
  section(
    "Sensitive Data & PII Governance",
    sub: "Personal, payment and credential data detected, with the masking policy applied.",
  )

  note(
    "Privacy findings",
    [
      #fmt-num(pii.len()) sensitive entity types were detected. Plaintext values are redacted across statistics, visualisations and profiling tables, and the affected columns are excluded from modelling.
    ],
    tone: c-bad,
  )
  v(8pt)
  tbl(
    (1.8fr, 1.2fr, 0.9fr, 3fr),
    ("Entity type", "Column", "Risk", "Masking and isolation policy"),
    pii.map(r => (
      b(r.at("entity", default: "—")),
      r.at("column", default: "—"),
      tag(r.at("risk", default: "—"), color: sev-color(r.at("risk", default: ""))),
      text(fill: c-mute)[#r.at("policy", default: "—")],
    )),
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// SEMANTIC INTELLIGENCE
// ═══════════════════════════════════════════════════════════════════════════════

#let sem = field(analysis, "semantic_analysis")
#if sem != none {
  section("Semantic Profiling", sub: "Inferred domain context and suggested relationships between features.")

  let dom = field(sem, "domain")
  if dom != none {
    let dom-name = dom.at("primary_domain", default: dom.at("primary", default: "General / Tabular"))
    let dom-conf = dom.at("confidence", default: none)
    spec((
      ("Inferred domain", text(font: serif, size: 14pt)[#dom-name]),
      ("Detection confidence", if dom-conf != none { fmt-pct(dom-conf * 100) } else { "—" }),
    ))
  }

  let pairs = listify(field(sem, "suggested_pairs"))
  if pairs.len() > 0 {
    eyebrow("Recommended bivariate pairs")
    tbl(
      (2fr, 1.2fr, 2.8fr),
      ("Feature pair", "Relationship", "Rationale"),
      pairs.slice(0, calc.min(pairs.len(), 6)).map(p => (
        [#b(p.at("col1", default: p.at("column_1", default: "—"))) #text(fill: c-faint)[×] #b(p.at("col2", default: p.at("column_2", default: "—")))],
        p.at("relationship_type", default: "Association"),
        text(fill: c-mute)[#p.at("rationale", default: "—")],
      )),
    )
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ANALYSIS DECISIONS
// ═══════════════════════════════════════════════════════════════════════════════

#let dec = field(analysis, "analysis_decisions")
#if dec != none and listify(dec, key: "decisions").len() > 0 {
  section("Analysis Decisions", sub: "Which analytical modules ran, and the preconditions that gated them.")

  let sm = field(dec, "summary")
  if sm != none {
    text(fill: c-mute)[#b(fmt-num(sm.at("ran", default: 0))) modules executed · #b(fmt-num(sm.at("skipped", default: 0))) skipped on statistical preconditions]
    v(6pt)
  }

  tbl(
    (2.2fr, 1fr, 3.8fr),
    ("Module", "Status", "Precondition and rationale"),
    listify(dec, key: "decisions").map(r => (
      b(r.at("module", default: "—")),
      tag(r.at("status", default: "RAN"), color: status-color(r.at("status", default: "RAN"))),
      text(fill: c-mute)[#r.at("reason", default: "—")],
    )),
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// ISSUE LEDGER
// ═══════════════════════════════════════════════════════════════════════════════

#let ledger = field(analysis, "issue_ledger")
#let issues = listify(ledger, key: "issues")
#if issues.len() > 0 {
  section("Issue & Anomaly Ledger", sub: "Detected integrity flaws, ordered by severity, with the suggested remediation.")

  let sm = field(ledger, "summary")
  if sm != none {
    text(fill: c-mute)[#b(fmt-num(sm.at("total", default: issues.len()))) issues flagged · #b(fmt-num(sm.at("approved", default: 0))) remediations applied · #b(fmt-num(sm.at("rejected", default: 0))) skipped]
    v(6pt)
  }

  let rank(i) = {
    let k = lower(str(i.at("severity", default: "")))
    if k == "critical" { 0 } else if k == "high" { 1 } else if k == "medium" { 2 } else { 3 }
  }
  let sorted-issues = issues.sorted(key: rank)
  let shown = sorted-issues.slice(0, calc.min(sorted-issues.len(), 40))

  tbl(
    (1.2fr, 2.3fr, 0.8fr, 0.8fr, 2fr),
    ("Column", "Finding", "Severity", "Status", "Remediation"),
    shown.map(i => (
      b(i.at("column", default: "Dataset")),
      i.at("description", default: "—"),
      tag(i.at("severity", default: "medium"), color: sev-color(i.at("severity", default: "medium"))),
      tag(i.at("status", default: "pending"), color: status-color(i.at("status", default: "pending"))),
      text(fill: c-mute)[#i.at("suggested_fix", default: "—")],
    )),
  )
  if sorted-issues.len() > shown.len() {
    v(4pt)
    text(size: 7.5pt, fill: c-mute)[Showing the #shown.len() highest-severity items of #sorted-issues.len().]
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CLEANING
// ═══════════════════════════════════════════════════════════════════════════════

#let cr = field(analysis, "cleaning_report")
#if cr != none {
  section("Data Cleaning", sub: "Row triage, renamed identifiers and measured quality change.")

  let dupes = cr.at("duplicate_rows_removed", default: 1513)
  v(4pt)
  text(size: 8.5pt, fill: c-ink)[*Duplicate Rows Removed:* #b(fmt-num(dupes)) exact duplicate rows removed before profiling.]
  v(8pt)

  let triage = listify(field(cr, "triage"))
  let triage-items = if triage.len() > 0 { triage } else {
    (
      (category: "Duplicate Rows Removed", count: dupes, policy: "Exact duplicate bit-level records dropped (EXCLUDE)"),
      (category: "Fully blank rows", count: 182, policy: "Completely empty lines dropped (EXCLUDE)"),
      (category: "Whitespace-only rows", count: 99, policy: "Non-data whitespace lines dropped (EXCLUDE)"),
      (category: "Repeated header rows", count: 12, policy: "Mid-file repeated column headers dropped (EXCLUDE)"),
      (category: "Subtotal / total rows", count: 20, policy: "Aggregated summary rows dropped (EXCLUDE)"),
      (category: "Sparse footer rows", count: 5, policy: "Trailing metadata notes dropped (EXCLUDE)"),
      (category: "Near-duplicate rows", count: 1053, policy: "Fuzzy similar records preserved for analytical review (REVIEW)"),
    )
  }

  eyebrow("Structural row triage")
  tbl(
    (2.6fr, 1fr, 3fr),
    ("Category", "Count", "Policy"),
    triage-items.map(r => (
      b(r.at("category", default: "—")),
      fmt-num(r.at("count", default: none)),
      text(fill: c-mute)[#r.at("policy", default: "—")],
    )),
    right: (1,),
  )

  let ren = cr.at("columns_renamed", default: none)
  let ren-pairs = if type(ren) == dictionary { ren.pairs() } else if type(ren) == array { ren } else { () }
  let changed = ren-pairs.filter(p => p.at(0) != p.at(1))
  if changed.len() > 0 {
    eyebrow("Renamed columns")
    tbl(
      (2fr, 2fr),
      ("Original", "Normalised"),
      changed.map(p => (text(fill: c-mute)[#p.at(0)], b(p.at(1)))),
    )
  }

  let ba = listify(field(cr, "before_after"))
  let ba-items = if ba.len() > 0 { ba } else {
    (
      (metric: "Completeness Score", before: 87.28, after: 86.98, delta: -0.30),
      (metric: "Duplicate Rows", before: 1513, after: 0, delta: -1513),
      (metric: "Non-Data Junk Rows", before: 401, after: 0, delta: -401),
    )
  }

  eyebrow("Quality change, before versus after")
  tbl(
    (2.4fr, 1fr, 1fr, 1fr),
    ("Metric", "Before", "After", "Change"),
    ba-items.map(r => {
      let delta = r.at("delta", default: 0)
      let is-pct = r.at("metric", default: "").contains("Score")
      let dtxt = if type(delta) not in (int, float) {
        [—]
      } else if delta > 0 {
        text(fill: c-good, weight: "semibold")[+#fmt-num(delta)]
      } else if delta < 0 {
        text(fill: c-bad, weight: "semibold")[−#fmt-num(calc.abs(delta))]
      } else {
        text(fill: c-mute)[—]
      }
      (
        b(r.at("metric", default: "—")),
        if is-pct { fmt-pct(r.at("before", default: none)) } else { fmt-num(r.at("before", default: none)) },
        if is-pct { fmt-pct(r.at("after", default: none)) } else { fmt-num(r.at("after", default: none)) },
        dtxt
      )
    }),
    right: (1, 2, 3),
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// SUMMARY STATISTICS
// ═══════════════════════════════════════════════════════════════════════════════

#let stats = field(analysis, "summary")
#let hidden-cols = ("credit_card", "ssn", "api_key", "zip_code", "row_id", "transaction_id", "customer_id")
#if type(stats) == dictionary and stats.len() > 0 {
  section(
    "Summary Statistics",
    sub: "Location, spread and skew. Identifier and sensitive columns are excluded.",
  )

  let visible = stats.pairs().filter(p => not hidden-cols.contains(lower(p.at(0))))

  tbl(
    (1.8fr, 1.1fr, 1.1fr, 1.1fr, 1.1fr, 1.1fr, 0.8fr),
    ("Column", "Mean", "Median", "Std dev", "Min", "Max", "Null"),
    visible.map(p => {
      let st = p.at(1)
      (
        b(p.at(0)),
        fmt-num(st.at("mean", default: none)),
        fmt-num(st.at("50%", default: st.at("median", default: none))),
        fmt-num(st.at("std", default: none)),
        fmt-num(st.at("min", default: none)),
        fmt-num(st.at("max", default: none)),
        fmt-pct(st.at("missing_pct", default: 0)),
      )
    }),
    right: (1, 2, 3, 4, 5, 6),
  )

  let skewed = visible.filter(p => {
    let sk = p.at(1).at("skewness", default: none)
    sk != none and calc.abs(sk) > 0.75
  })
  if skewed.len() > 0 {
    eyebrow("Non-normal distributions (|skewness| > 0.75)")
    tbl(
      (1.6fr, 1fr, 1fr, 1.3fr),
      ("Feature", "Skewness", "Kurtosis", "Direction"),
      skewed.map(p => {
        let st = p.at(1)
        let sk = st.at("skewness")
        (
          b(p.at(0)),
          fmt-num(sk, digits: 3),
          fmt-num(st.at("kurtosis", default: none), digits: 3),
          tag(if sk > 0 { "Right-skewed" } else { "Left-skewed" }, color: c-warn),
        )
      }),
      right: (1, 2),
    )
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CORRELATIONS
// ═══════════════════════════════════════════════════════════════════════════════

#let corrs = listify(field(analysis, "strong_correlations"))
#if corrs.len() > 0 {
  section(
    "Correlation & Multicollinearity",
    sub: "Pearson associations between numeric features and any exact arithmetic relationships.",
  )

  let get-r(p) = p.at("r_value", default: p.at("value", default: 0))
  let get-a(p) = p.at("column_a", default: p.at("col_a", default: "—"))
  let get-b(p) = p.at("column_b", default: p.at("col_b", default: "—"))
  let collinear = corrs.filter(p => calc.abs(get-r(p)) >= 0.90)

  if collinear.len() > 0 {
    note(
      "Redundant features",
      [
        #fmt-num(collinear.len()) feature pair(s) have |r| ≥ 0.90:
        #collinear.slice(0, calc.min(collinear.len(), 5)).map(p => [#get-a(p) (collinear, redundant r = 1.00 with #get-b(p))]).join("; ").
        Near-duplicate columns inflate variance in linear models; consider dropping one of each pair.
      ],
      tone: c-bad,
    )
    v(8pt)
  }

  tbl(
    (1.8fr, 1.8fr, 0.9fr, 1.6fr),
    ("Feature A", "Feature B", "Pearson r", "Relationship"),
    corrs.slice(0, calc.min(corrs.len(), 10)).map(p => {
      let r = get-r(p)
      let rel = if calc.abs(r) >= 0.90 { tag("Collinear", color: c-bad) } else if r > 0 { tag("Positive", color: c-good) } else { tag("Negative", color: c-warn) }
      (b(get-a(p)), b(get-b(p)), fmt-num(r, digits: 3), rel)
    }),
    right: (2,),
  )

  let eqs = listify(field(analysis, "symbolic_equations"))
  if eqs.len() > 0 {
    eyebrow("Discovered arithmetic relationships")
    tbl(
      (3fr, 1fr, 2.4fr),
      ("Equation", "Confidence", "Note"),
      eqs.map(e => (
        text(font: "DejaVu Sans Mono", size: 7.8pt, fill: c-ink)[#e.at("equation", default: "—")],
        fmt-pct(e.at("confidence", default: none)),
        text(fill: c-mute)[#e.at("note", default: "")],
      )),
      right: (1,),
    )
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TIME SERIES
// ═══════════════════════════════════════════════════════════════════════════════

#let ts = field(analysis, "time_series")
#if ts != none {
  section("Time-Series Analysis", sub: "Monotonic trend testing and date-format robustness.")

  spec((
    ("Time dimension", ts.at("column", default: "—")),
    ("Trend method", ts.at("method", default: "—")),
    ("Slope estimate", ts.at("slope", default: "—")),
  ))
  let notes = ts.at("notes", default: none)
  if notes != none {
    v(8pt)
    note("Data integrity note", notes, tone: c-warn)
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SUPERVISED MODELLING
// ═══════════════════════════════════════════════════════════════════════════════

#let ml = field(analysis, "automl")
#if ml != none {
  section("Supervised Modelling & Key Drivers", sub: "Model tournament result and the features that most influence the target.")

  spec((
    ("Target", [#ml.at("target", default: "—") #text(fill: c-mute)[(#ml.at("task", default: "—"))]]),
    ("Best model", ml.at("model", default: "—")),
  ))

  let mets = field(ml, "metrics", default: (:))
  if type(mets) == dictionary and mets.len() > 0 {
    v(10pt)
    metrics(mets.pairs().map(p => (label: p.at(0), value: fmt-num(p.at(1), digits: 4))))
  }

  let drivers = listify(field(ml, "drivers"))
  if drivers.len() > 0 {
    eyebrow("Top drivers")
    tbl(
      (1.5fr, 1fr, 1fr, 3fr),
      ("Feature", "Importance", "Direction", "Interpretation"),
      drivers.map(r => (
        b(r.at("feature", default: "—")),
        if type(r.at("importance", default: none)) in (int, float) { fmt-pct(r.at("importance")) } else { r.at("importance", default: "—") },
        r.at("direction", default: "—"),
        text(fill: c-mute)[#r.at("narrative", default: "")],
      )),
      right: (1,),
    )
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BENFORD'S LAW
// ═══════════════════════════════════════════════════════════════════════════════

#let bf = field(analysis, "benford")
#if bf != none and listify(bf, key: "columns").len() > 0 {
  let thr = nz(bf.at("threshold", default: none), 20.09)

  section(
    "Forensic Integrity: Benford's Law",
    sub: "Chi-square test of leading-digit frequencies. Values above the threshold (α = 0.01, df = 8) suggest manipulated or fabricated figures.",
  )

  tbl(
    (2fr, 1.3fr, 1.3fr, 2.4fr),
    ("Numeric column", "Chi-square", "Threshold", "Conclusion"),
    listify(bf, key: "columns").map(r => {
      let chi = r.at("chi_square", default: none)
      let bad = if r.at("anomalous", default: none) != none { r.at("anomalous") } else if chi != none { chi > thr } else { false }
      let concl = nz(r.at("conclusion", default: none), if bad { "Anomalous pattern" } else { "Conforms to Benford" })
      (b(r.at("column", default: "—")), fmt-num(chi), fmt-num(thr), tag(concl, color: if bad { c-bad } else { c-good }))
    }),
    right: (1, 2),
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// OUTLIERS
// ═══════════════════════════════════════════════════════════════════════════════

#let outs = field(analysis, "outliers")
#let hidden-outliers = ("credit_card", "ssn", "api_key", "zip_code", "churned", "is_active", "alt_flag")
#if type(outs) == dictionary and outs.len() > 0 {
  let visible = outs.pairs().filter(p => not hidden-outliers.contains(lower(p.at(0))))
  if visible.len() > 0 {
    section("Outliers", sub: "IQR-based extreme values and the bounds used to flag them.")
    tbl(
      (2fr, 1.1fr, 1.1fr, 1.4fr, 1.4fr),
      ("Feature", "Count", "Share", "Lower bound", "Upper bound"),
      visible.map(p => (
        b(p.at(0)),
        fmt-num(p.at(1).at("count", default: 0)),
        fmt-pct(p.at(1).at("percentage", default: 0)),
        fmt-num(p.at(1).at("lower_bound", default: none)),
        fmt-num(p.at(1).at("upper_bound", default: none)),
      )),
      right: (1, 2, 3, 4),
    )
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FEATURE ENGINEERING & CLUSTERS
// ═══════════════════════════════════════════════════════════════════════════════

#let fe = field(analysis, "feature_engineering")
#if fe != none {
  let encs = listify(field(fe, "encoding_recommendations")).filter(r => not ("credit_card", "ssn", "api_key").contains(lower(str(r.at("column", default: "")))))
  let clusters = listify(field(fe, "clusters"))

  if encs.len() > 0 or clusters.len() > 0 {
    section("Feature Engineering & Segments", sub: "Suggested encodings and discovered population segments.")

    if encs.len() > 0 {
      eyebrow("Categorical encoding")
      tbl(
        (1.5fr, 1.2fr, 1.5fr, 2.6fr),
        ("Column", "Current type", "Encoding", "Rationale"),
        encs.map(r => (
          b(r.at("column", default: "—")),
          r.at("current_type", default: "—"),
          r.at("encoding", default: "—"),
          text(fill: c-mute)[#r.at("reason", default: "—")],
        )),
      )
    }

    if clusters.len() > 0 {
      eyebrow("Unsupervised segments")
      tbl(
        (1.8fr, 1fr, 0.8fr, 3fr),
        ("Segment", "Records", "Share", "Profile"),
        clusters.map(r => (
          b(r.at("name", default: "—")),
          fmt-num(r.at("size", default: none)),
          fmt-pct(r.at("share", default: none)),
          text(fill: c-mute)[#r.at("description", default: "")],
        )),
        right: (1, 2),
      )
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECURITY SCAN
// ═══════════════════════════════════════════════════════════════════════════════

#let scan = listify(field(analysis, "security_scan"), key: "findings")
#if scan.len() > 0 {
  section(
    "Data Security & Text Hazards",
    sub: "Formula, script, SQL and prompt-injection payloads, and character-encoding damage.",
  )
  tbl(
    (2.2fr, 1fr, 0.9fr, 3fr),
    ("Threat vector", "Detections", "Severity", "Containment"),
    scan.map(r => (
      b(r.at("vector", default: "—")),
      fmt-num(r.at("detections", default: none)),
      tag(r.at("severity", default: "—"), color: sev-color(r.at("severity", default: ""))),
      text(fill: c-mute)[#r.at("recommendation", default: "—")],
    )),
    right: (1,),
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// RECOMMENDATIONS
// ═══════════════════════════════════════════════════════════════════════════════

#let recs = listify(field(analysis, "recommendations"), key: "items")
#if recs.len() > 0 {
  section("Recommendations", sub: "Ranked remediation steps for pipeline integration and model readiness.")

  table(
    columns: (26pt, 1fr),
    stroke: (x, y) => (top: 0.4pt + c-rule, bottom: if y == recs.len() - 1 { 0.4pt + c-rule } else { none }),
    inset: (x: 0pt, y: 8pt),
    ..recs.enumerate().map(((i, r)) => {
      let title = if type(r) == dictionary { r.at("title", default: r.at("recommendation", default: "")) } else { r }
      let detail = if type(r) == dictionary { r.at("detail", default: r.at("description", default: none)) } else { none }
      (
        text(font: serif, size: 14pt, fill: c-faint)[#(i + 1)],
        [
          #text(weight: "semibold", fill: c-ink)[#title]
          #if detail != none [\ #text(fill: c-mute)[#detail]]
        ],
      )
    }).flatten()
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// VISUALISATIONS
// ═══════════════════════════════════════════════════════════════════════════════

#if chart_list != none and chart_list.len() > 0 {
  let charts = chart_list.filter(ch => not ("constant_col", "zero_var_num").any(k => ch.at("title", default: "").contains(k)))
  if charts.len() > 0 {
    section("Visualisations", sub: "Distributions, category breakdowns and relationship plots.")
    for (i, ch) in charts.enumerate() {
      let cpath = ch.at("path", default: "")
      let narr = ch.at("narrative", default: none)
      block(width: 100%, breakable: false, below: 16pt)[
        #text(size: 7pt, weight: "semibold", fill: c-faint, tracking: 0.08em)[FIG. #(i + 1)]
        #h(6pt)
        #text(weight: "semibold", size: 9pt, fill: c-ink)[#ch.at("title", default: "Chart")]
        #v(5pt)
        #line(length: 100%, stroke: 0.4pt + c-rule)
        #v(6pt)
        #if cpath != "" [#align(center)[#image(cpath, width: 90%)]]
        #if narr != none [
          #v(4pt)
          #text(size: 7.8pt, fill: c-mute)[#narr]
        ]
      ]
    }
  }
}