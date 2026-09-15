// ═══════════════════════════════════════════════════════════════════════════════
// GetReport — Enterprise Data Intelligence Audit Report
// High-End Executive Typst PDF Template (Zero-Install Rust Engine)
// ═══════════════════════════════════════════════════════════════════════════════

#let d = json(bytes(sys.inputs.data))

#let filename = d.at("filename", default: "Dataset")
#let generated_at = d.at("generated_at", default: "—")
#let metadata = d.at("metadata", default: (:))
#let analysis = d.at("analysis", default: (:))
#let chart_list = d.at("chart_list", default: ())

// ── Color System ──────────────────────────────────────────────────────────────
#let c-obsidian = rgb("#090d16")     // Deep obsidian 950
#let c-slate-900 = rgb("#0f172a")   // Slate 900
#let c-slate-800 = rgb("#1e293b")   // Slate 800
#let c-slate-700 = rgb("#334155")   // Slate 700
#let c-slate-500 = rgb("#64748b")   // Slate 500
#let c-slate-300 = rgb("#cbd5e1")   // Slate 300
#let c-slate-100 = rgb("#f1f5f9")   // Slate 100
#let c-slate-50 = rgb("#f8fafc")    // Slate 50
#let c-border = rgb("#e2e8f0")      // Border 200

#let c-blue-600 = rgb("#2563eb")    // Electric blue
#let c-blue-50 = rgb("#eff6ff")     // Soft blue
#let c-emerald-600 = rgb("#059669") // Emerald green
#let c-emerald-50 = rgb("#ecfdf5")  // Soft green
#let c-amber-600 = rgb("#d97706")   // Amber warning
#let c-amber-50 = rgb("#fffbeb")    // Soft amber
#let c-rose-600 = rgb("#e11d48")    // Crimson rose
#let c-rose-50 = rgb("#fff1f2")     // Soft rose
#let c-indigo-600 = rgb("#4f46e5")  // Indigo

// ── Helper Utilities ──────────────────────────────────────────────────────────
#let fmt-num(val, digits: 2) = {
  if val == none { "—" }
  else if type(val) == int { str(val) }
  else if type(val) == float { str(calc.round(val, digits: digits)) }
  else { str(val) }
}

#let fmt-pct(val, digits: 1) = {
  if val == none { "—" }
  else if type(val) == int or type(val) == float { str(calc.round(val, digits: digits)) + "%" }
  else { str(val) + "%" }
}

// ── Modern Pill Badges ────────────────────────────────────────────────────────
#let badge(lbl, bg: c-slate-100, fg: c-slate-700) = {
  box(
    fill: bg,
    radius: 3pt,
    inset: (x: 6pt, y: 2.5pt),
    baseline: 0%,
    text(fill: fg, weight: "bold", size: 7pt)[#lbl]
  )
}

#let badge-grade(grade) = {
  let g = lower(str(grade))
  if g == "a" { badge(grade, bg: rgb("#dcfce7"), fg: rgb("#15803d")) }
  else if g == "b" { badge(grade, bg: rgb("#dbeafe"), fg: rgb("#1d4ed8")) }
  else if g == "c" { badge(grade, bg: rgb("#fef3c7"), fg: rgb("#b45309")) }
  else { badge(grade, bg: rgb("#fee2e2"), fg: rgb("#b91c1c")) }
}

#let badge-severity(sev) = {
  let s = lower(str(sev))
  if s == "high" or s == "critical" { badge(upper(sev), bg: c-rose-50, fg: c-rose-600) }
  else if s == "medium" or s == "warning" { badge(upper(sev), bg: c-amber-50, fg: c-amber-600) }
  else { badge(upper(sev), bg: c-emerald-50, fg: c-emerald-600) }
}

#let badge-status(st) = {
  let s = lower(str(st))
  if s == "ran" or s == "approved" or s == "positive" { badge(st, bg: c-emerald-50, fg: c-emerald-600) }
  else if s == "skipped" or s == "rejected" { badge(st, bg: c-slate-100, fg: c-slate-500) }
  else { badge(st, bg: c-blue-50, fg: c-blue-600) }
}

// ── Native Vector Graphics Components ─────────────────────────────────────────

#let vector-progress-bar(score, width: 60pt, height: 5pt, max-val: 100) = {
  let s = if score == none { 0 } else { calc.min(calc.max(score, 0), max-val) }
  let pct = s / max-val
  let fill-col = if s >= 80 { c-emerald-600 } else if s >= 60 { c-blue-600 } else if s >= 40 { c-amber-600 } else { c-rose-600 }
  box(baseline: 20%, width: width, height: height)[
    #place(top + left, rect(width: width, height: height, fill: rgb("#e2e8f0"), radius: 2.5pt))
    #if pct > 0 [
      #place(top + left, rect(width: width * pct, height: height, fill: fill-col, radius: 2.5pt))
    ]
  ]
}

#let vector-distribution-bar(valid-pct, missing-pct: 0, width: 65pt, height: 5pt) = {
  let v-val = if valid-pct == none { 100 } else { calc.min(calc.max(valid-pct, 0), 100) }
  let m-val = if missing-pct == none { 0 } else { calc.min(calc.max(missing-pct, 0), 100) }
  let v-pct = v-val / 100
  let m-pct = m-val / 100
  box(baseline: 20%, width: width, height: height)[
    #place(top + left, rect(width: width, height: height, fill: rgb("#fee2e2"), radius: 2.5pt))
    #if v-pct > 0 [
      #place(top + left, rect(width: width * v-pct, height: height, fill: c-emerald-600, radius: 2.5pt))
    ]
  ]
}

#let vector-kpi-gauge(score, label: "SCORE", size: 40pt) = {
  let s = if score == none { 0 } else { calc.min(calc.max(score, 0), 100) }
  let col = if s >= 80 { c-emerald-600 } else if s >= 60 { c-blue-600 } else if s >= 40 { c-amber-600 } else { c-rose-600 }
  box(
    width: size,
    height: size,
    radius: size / 2,
    stroke: 2.5pt + col,
    fill: rgb("#f8fafc"),
    [
      #align(center + horizon)[
        #text(size: 9pt, weight: "bold", fill: col)[#calc.round(s)%]\
        #text(size: 4.5pt, weight: "bold", fill: c-slate-500)[#label]
      ]
    ]
  )
}

// ── Section Header Component ──────────────────────────────────────────────────
#let section-title(title, subtitle: none, num: none, accent: c-blue-600) = {
  v(14pt)
  block(width: 100%)[
    #if num != none [
      #text(size: 7.5pt, weight: "bold", fill: accent, tracking: 0.12em)[#upper(num)]
      #v(1pt)
    ]
    #grid(
      columns: (4pt, 1fr),
      gutter: 8pt,
      rect(width: 4pt, height: 16pt, fill: accent, radius: 2pt),
      align(horizon)[
        #text(size: 13pt, weight: "bold", fill: c-slate-900)[#title]
      ]
    )
    #if subtitle != none [
      #v(2pt)
      #text(size: 8pt, fill: c-slate-500)[#subtitle]
    ]
  ]
  v(6pt)
}

// ── Executive Callout Box ─────────────────────────────────────────────────────
#let callout(title: none, body, border-color: c-blue-600, bg-color: c-slate-50) = {
  block(
    width: 100%,
    fill: bg-color,
    stroke: (left: 3.5pt + border-color, rest: 0.5pt + c-border),
    radius: (right: 4pt),
    inset: (x: 10pt, y: 8pt),
    [
      #if title != none [
        #text(weight: "bold", size: 9pt, fill: c-slate-900)[#title]
        #v(3pt)
      ]
      #text(size: 8pt, fill: c-slate-700)[#body]
    ]
  )
  v(4pt)
}

// ── Document Configuration ────────────────────────────────────────────────────
#set document(title: "GetReport Audit — " + filename, author: "GetReport Data Labs")
#set page(
  paper: "a4",
  margin: (top: 2.2cm, bottom: 2.0cm, left: 1.8cm, right: 1.8cm),
  header: context {
    if here().page() > 1 [
      #grid(
        columns: (1fr, auto),
        align(left + horizon)[
          #text(size: 7.5pt, weight: "bold", fill: c-slate-800)[GETREPORT DATA INTELLIGENCE]
          #text(size: 7.5pt, fill: c-slate-300)[ • ]
          #text(size: 7.5pt, fill: c-slate-500)[#filename]
        ],
        align(right + horizon)[
          #text(size: 7pt, fill: c-slate-500)[#generated_at]
        ]
      )
      #v(3pt)
      #line(length: 100%, stroke: 0.5pt + c-border)
    ]
  },
  footer: context {
    if here().page() > 1 [
      #line(length: 100%, stroke: 0.5pt + c-border)
      #v(4pt)
      #grid(
        columns: (1fr, auto),
        align(left + horizon)[
          #text(size: 7pt, fill: c-slate-500)[Confidential • Enterprise Data Quality & Integrity Engine]
        ],
        align(right + horizon)[
          #box(
            fill: c-slate-100,
            radius: 3pt,
            inset: (x: 6pt, y: 2.5pt),
            text(size: 7pt, weight: "bold", fill: c-slate-700)[
              Page #counter(page).display("1 of 1", both: true)
            ]
          )
        ]
      )
    ]
  }
)
#set text(font: ("Outfit", "Inter", "Liberation Sans", "DejaVu Sans", "Arial"), size: 8.5pt, fill: c-slate-800)
#set par(justify: true, leading: 0.6em)

// ═══════════════════════════════════════════════════════════════════════════════
// COVER PAGE — EXECUTIVE AUDIT REPORT
// ═══════════════════════════════════════════════════════════════════════════════

#v(1cm)
#grid(
  columns: (1fr, auto),
  [
    #box(
      fill: rgb("#0f172a"),
      radius: 3pt,
      inset: (x: 8pt, y: 4pt),
      text(size: 8pt, weight: "bold", fill: rgb("#38bdf8"), tracking: 0.15em)[GETREPORT DATA LABS]
    )
  ],
  align(right + horizon)[
    #text(size: 7.5pt, weight: "bold", fill: c-slate-500, tracking: 0.1em)[ENTERPRISE AUDIT CERTIFICATION]
  ]
)

#v(1.5cm)

#block(width: 100%)[
  #text(size: 11pt, weight: "bold", fill: c-blue-600, tracking: 0.12em)[DATA QUALITY & STATISTICAL AUDIT REPORT]\
  #v(6pt)
  #text(size: 26pt, weight: "bold", fill: c-obsidian)[
    Autonomous Data Intelligence & Integrity Audit
  ]\
  #v(8pt)
  #text(size: 11pt, fill: c-slate-500)[
    Deep Entity Profiling, Statistical Trust Ledger, Anomaly Ledger & Machine Learning Readiness
  ]
]

#v(1.2cm)

// Metadata Matrix Grid
#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  block(
    fill: c-slate-50,
    stroke: 0.5pt + c-border,
    radius: 4pt,
    inset: 12pt,
    [
      #text(size: 7.5pt, weight: "bold", fill: c-slate-500, tracking: 0.1em)[SOURCE ARTIFACT]\
      #v(2pt)
      #text(size: 12pt, weight: "bold", fill: c-slate-900)[#filename]\
      #v(2pt)
      #text(size: 7.5pt, fill: c-slate-500)[Audit Timestamp: #generated_at]
    ]
  ),
  block(
    fill: c-slate-50,
    stroke: 0.5pt + c-border,
    radius: 4pt,
    inset: 12pt,
    [
      #text(size: 7.5pt, weight: "bold", fill: c-slate-500, tracking: 0.1em)[DATASET DIMENSIONALITY]\
      #v(2pt)
      #text(size: 12pt, weight: "bold", fill: c-slate-900)[
        #metadata.at("total_rows", default: 0) Rows × #metadata.at("total_columns", default: 0) Columns
      ]\
      #v(2pt)
      #text(size: 7.5pt, fill: c-slate-500)[
        #metadata.at("numeric_columns", default: 0) Numerical • #metadata.at("categorical_columns", default: 0) Categorical Features
      ]
    ]
  )
)

#v(10pt)

// Executive Health & Trust Matrix Card
#if "confidence_scores" in analysis and analysis.confidence_scores != none [
  #let cs = analysis.confidence_scores
  #let grade-val = cs.at("dataset_grade", default: "B")
  #let grade-color = if lower(grade-val) == "a" { c-emerald-600 } else if lower(grade-val) == "b" { c-blue-600 } else if lower(grade-val) == "c" { c-amber-600 } else { c-rose-600 }

  #block(
    width: 100%,
    fill: rgb("#0f172a"),
    radius: 6pt,
    inset: 16pt,
    [
      #grid(
        columns: (auto, 1fr),
        gutter: 18pt,
        box(
          fill: grade-color,
          radius: 6pt,
          inset: (x: 16pt, y: 12pt),
          align(center + horizon)[
            #text(size: 8pt, weight: "bold", fill: white)[OVERALL]\
            #text(size: 26pt, weight: "bold", fill: white)[#grade-val]\
            #text(size: 7pt, weight: "bold", fill: white)[GRADE]
          ]
        ),
        [
          #text(size: 12pt, weight: "bold", fill: white)[Executive Quality Scorecard]\
          #v(2pt)
          #text(size: 8pt, fill: rgb("#94a3b8"))[
            Evaluated across 4 core trust vectors: Completeness, Semantic Consistency, Value Validity, and Distribution Stability.
          ]
          #v(6pt)
          #grid(
            columns: (1fr, 1fr, 1fr),
            gutter: 8pt,
            [
              #text(size: 7pt, fill: rgb("#94a3b8"))[HIGH CONFIDENCE]\
              #text(size: 13pt, weight: "bold", fill: rgb("#34d399"))[#cs.at("high_confidence_count", default: 0) Cols]
            ],
            [
              #text(size: 7pt, fill: rgb("#94a3b8"))[LOW CONFIDENCE]\
              #text(size: 13pt, weight: "bold", fill: if cs.at("low_confidence_count", default: 0) > 0 { rgb("#fbbf24") } else { rgb("#94a3b8") })[#cs.at("low_confidence_count", default: 0) Cols]
            ],
            [
              #text(size: 7pt, fill: rgb("#94a3b8"))[ANALYTICAL READY]\
              #text(size: 13pt, weight: "bold", fill: white)[#metadata.at("analytical_numeric_columns", default: 0) Features]
            ]
          )
        ]
      )
    ]
  )
]

#v(2.5cm)

#line(length: 100%, stroke: 0.5pt + c-border)
#v(6pt)
#grid(
  columns: (1fr, auto),
  text(size: 7pt, fill: c-slate-500)[
    PIPELINE SIGNATURE: ZERO-COPY TYPST RUST ENGINE • GETREPORT CORE v2.4
  ],
  text(size: 7pt, weight: "bold", fill: c-slate-500)[
    STRICTLY CONFIDENTIAL
  ]
)

#pagebreak()

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 01 — EXECUTIVE SUMMARY & ML READINESS
// ═══════════════════════════════════════════════════════════════════════════════

#section-title(
  "Executive Summary & Machine Learning Readiness",
  subtitle: "High-level health synthesis and downstream model applicability.",
  num: "SECTION 01",
  accent: c-emerald-600
)

#if "confidence_scores" in analysis and analysis.confidence_scores != none [
  #let cs = analysis.confidence_scores
  #let grade-val = cs.at("dataset_grade", default: "B")
  #let grade-color = if lower(grade-val) == "a" { c-emerald-600 } else if lower(grade-val) == "b" { c-blue-600 } else if lower(grade-val) == "c" { c-amber-600 } else { c-rose-600 }

  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 8pt,
    block(
      fill: c-slate-50, stroke: 0.5pt + c-border, radius: 4pt, inset: 10pt,
      [
        #text(size: 7pt, weight: "bold", fill: c-slate-500)[OVERALL RATING]\
        #v(3pt)
        #text(size: 18pt, weight: "bold", fill: grade-color)[Grade #grade-val]\
        #v(1pt)
        #text(size: 7pt, fill: c-slate-500)[Aggregate Health Score]
      ]
    ),
    block(
      fill: c-slate-50, stroke: 0.5pt + c-border, radius: 4pt, inset: 10pt,
      [
        #text(size: 7pt, weight: "bold", fill: c-slate-500)[VERIFIED ROBUST]\
        #v(3pt)
        #text(size: 18pt, weight: "bold", fill: c-emerald-600)[#cs.at("high_confidence_count", default: 0) Cols]\
        #v(1pt)
        #text(size: 7pt, fill: c-slate-500)[High Confidence (A/B)]
      ]
    ),
    block(
      fill: c-slate-50, stroke: 0.5pt + c-border, radius: 4pt, inset: 10pt,
      [
        #text(size: 7pt, weight: "bold", fill: c-slate-500)[AT RISK]\
        #v(3pt)
        #text(size: 18pt, weight: "bold", fill: if cs.at("low_confidence_count", default: 0) > 0 { c-amber-600 } else { c-slate-500 })[#cs.at("low_confidence_count", default: 0) Cols]\
        #v(1pt)
        #text(size: 7pt, fill: c-slate-500)[Requires Cleansing]
      ]
    ),
    block(
      fill: c-slate-50, stroke: 0.5pt + c-border, radius: 4pt, inset: 10pt,
      [
        #text(size: 7pt, weight: "bold", fill: c-slate-500)[ANALYTICAL FEATURES]\
        #v(3pt)
        #text(size: 18pt, weight: "bold", fill: c-slate-900)[#metadata.at("analytical_numeric_columns", default: 0)]\
        #v(1pt)
        #text(size: 7pt, fill: c-slate-500)[Continuous Numeric]
      ]
    )
  )

  #v(6pt)

  #if "ml_readiness" in cs and cs.ml_readiness != none [
    #let ml = cs.ml_readiness
    #let ml-score = ml.at("score", default: 0)
    #let ml-status = ml.at("status", default: "Moderate")
    #let ml-bg = if ml-score >= 80 { c-emerald-50 } else if ml-score >= 50 { c-blue-50 } else { c-amber-50 }
    #let ml-border = if ml-score >= 80 { c-emerald-600 } else if ml-score >= 50 { c-blue-600 } else { c-amber-600 }

    #block(
      width: 100%,
      fill: ml-bg,
      stroke: (left: 3.5pt + ml-border, rest: 0.5pt + c-border),
      radius: (right: 4pt),
      inset: 10pt,
      [
        #grid(
          columns: (auto, 1fr),
          gutter: 12pt,
          align: (center + horizon, left + horizon),
          vector-kpi-gauge(ml-score, label: "ML-READY", size: 44pt),
          [
            #text(size: 9.5pt, weight: "bold", fill: c-slate-900)[Status: #ml-status]\
            #text(size: 8pt, fill: c-slate-700)[#ml.at("column_context", default: "")]\
            #v(2pt)
            #text(size: 8pt, fill: c-slate-900)[*Recommendation:* #ml.at("recommendation", default: "")]
          ]
        )
      ]
    )
  ]
]

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 02 — COLUMN CONFIDENCE TRUST LEDGER
// ═══════════════════════════════════════════════════════════════════════════════

#if "confidence_scores" in analysis and analysis.confidence_scores != none and "columns" in analysis.confidence_scores [
  #section-title(
    "Column Confidence Scores (Trust Ledger)",
    subtitle: "Granular multi-factor quality scoring for every dataset column.",
    num: "SECTION 02",
    accent: c-blue-600
  )

  #let cs-cols = analysis.confidence_scores.columns
  #let t-rows = ()
  #for col-data in cs-cols [
    #let col-name = col-data.at("column", default: "—")
    #let score = calc.round(col-data.at("overall", default: 0))
    #let grade = col-data.at("grade", default: "C")
    #let factors = "Completeness " + str(calc.round(col-data.at("completeness", default: 0))) + "%, Consistency " + str(calc.round(col-data.at("consistency", default: 0))) + "%, Validity " + str(calc.round(col-data.at("validity", default: 0))) + "%, Stability " + str(calc.round(col-data.at("stability", default: 0))) + "%"
    
    #t-rows.push([*#col-name*])
    #t-rows.push([
      #grid(
        columns: (auto, 1fr),
        gutter: 5pt,
        align: horizon,
        text(size: 7.5pt, weight: "bold")[#score%],
        vector-progress-bar(score, width: 44pt, height: 5pt)
      )
    ])
    #t-rows.push(badge-grade(grade))
    #t-rows.push(text(size: 7.5pt, fill: c-slate-500)[#factors])
  ]

  #table(
    columns: (1.4fr, 1.2fr, 0.6fr, 3fr),
    fill: (col, row) => if row == 0 { c-slate-900 } else if calc.even(row) { c-slate-50 } else { white },
    stroke: (col, row) => if row == 0 { none } else { (top: 0.5pt + c-border, bottom: none) },
    inset: (x: 8pt, y: 5.5pt),
    align: (col, row) => if row == 0 { center + horizon } else if col == 2 { center + horizon } else { left + horizon },
    table.header(
      text(fill: white, weight: "bold")[Column Name],
      text(fill: white, weight: "bold")[Score],
      text(fill: white, weight: "bold")[Grade],
      text(fill: white, weight: "bold")[Trust Metric Breakdown],
    ),
    ..t-rows
  )
]

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 03 — SEMANTIC INTELLIGENCE & DOMAIN DETECTION
// ═══════════════════════════════════════════════════════════════════════════════

#if "semantic_analysis" in analysis and analysis.semantic_analysis != none [
  #let sem = analysis.semantic_analysis
  #section-title(
    "Semantic Intelligence & Entity Profiling",
    subtitle: "Inferred domain context, semantic types, and relationship pairings.",
    num: "SECTION 03",
    accent: c-indigo-600
  )

  #if "domain" in sem and sem.domain != none [
    #let dom = sem.domain
    #let dname = dom.at("primary", default: "Generic Business")
    #let dconf = calc.round(dom.at("confidence", default: 0) * 100)
    #callout(
      title: "Inferred Business Domain: " + dname,
      [Autonomous domain detection matched this dataset to *#dname* with *#dconf%* statistical confidence. Automated benchmark rules and validation profiles for this domain have been applied.],
      border-color: c-indigo-600,
      bg-color: rgb("#f5f3ff")
    )
  ]

  #if "column_roles" in sem and sem.column_roles != none [
    #let r-rows = ()
    #for (cname, rdata) in sem.column_roles [
      #let role = rdata.at("role", default: "attribute")
      #let stype = rdata.at("semantic_type", default: "—")
      #r-rows.push([*#cname*])
      #r-rows.push(badge(role, bg: c-blue-50, fg: c-blue-600))
      #r-rows.push(text(fill: c-slate-500)[#stype])
    ]
    #table(
      columns: (1.5fr, 1.2fr, 2fr),
      fill: (col, row) => if row == 0 { c-slate-900 } else if calc.even(row) { c-slate-50 } else { white },
      stroke: (col, row) => if row == 0 { none } else { (top: 0.5pt + c-border, bottom: none) },
      inset: (x: 8pt, y: 5pt),
      table.header(
        text(fill: white, weight: "bold")[Column],
        text(fill: white, weight: "bold")[Inferred Entity Role],
        text(fill: white, weight: "bold")[Semantic Classification],
      ),
      ..r-rows
    )
  ]

  #if "suggested_pairs" in sem and sem.suggested_pairs != none and sem.suggested_pairs.len() > 0 [
    #v(4pt)
    #text(weight: "bold", size: 8.5pt)[Recommended Analysis Pairs:]
    #list(
      ..sem.suggested_pairs.map(p => [
        *#p.at("col_a", default: "")* ↔ *#p.at("col_b", default: "")*: #p.at("reason", default: "")
      ])
    )
  ]
]

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 04 — AUTONOMOUS ANALYSIS DECISIONS
// ═══════════════════════════════════════════════════════════════════════════════

#if "analysis_decisions" in analysis and analysis.analysis_decisions != none [
  #let dec = analysis.analysis_decisions
  #section-title(
    "Autonomous Analysis Decision Log",
    subtitle: "Deterministic profiling rationale explaining executed and bypassed analyses.",
    num: "SECTION 04",
    accent: c-slate-700
  )

  #if "summary" in dec and dec.summary != none [
    #text(size: 8pt, fill: c-slate-500)[
      *#dec.summary.at("ran", default: 0)* procedures executed · *#dec.summary.at("skipped", default: 0)* procedures bypassed based on data prerequisite tests.
    ]
    #v(4pt)
  ]

  #if "decisions" in dec and dec.decisions != none and dec.decisions.len() > 0 [
    #let d-rows = ()
    #for d-item in dec.decisions [
      #let dname = d-item.at("analysis", default: d-item.at("name", default: "—"))
      #let dran = d-item.at("decision", default: "") == "ran" or d-item.at("ran", default: false)
      #let dreason = d-item.at("reason", default: "—")
      #d-rows.push([*#dname*])
      #d-rows.push(if dran { badge-status("Ran") } else { badge-status("Skipped") })
      #d-rows.push(text(size: 7.5pt, fill: c-slate-500)[#dreason])
    ]
    #table(
      columns: (1.5fr, 1fr, 3fr),
      fill: (col, row) => if row == 0 { c-slate-900 } else if calc.even(row) { c-slate-50 } else { white },
      stroke: (col, row) => if row == 0 { none } else { (top: 0.5pt + c-border, bottom: none) },
      inset: (x: 8pt, y: 5pt),
      align: (col, row) => if row == 0 { center + horizon } else if col == 1 { center + horizon } else { left + horizon },
      table.header(
        text(fill: white, weight: "bold")[Analysis Procedure],
        text(fill: white, weight: "bold")[Status],
        text(fill: white, weight: "bold")[Pre-condition Evaluation],
      ),
      ..d-rows
    )
  ]
]

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 05 — ISSUE & ANOMALY LEDGER
// ═══════════════════════════════════════════════════════════════════════════════

#if "issue_ledger" in analysis and analysis.issue_ledger != none and "issues" in analysis.issue_ledger and analysis.issue_ledger.issues.len() > 0 [
  #let ledger = analysis.issue_ledger
  #section-title(
    "Data Quality Issue & Anomaly Ledger",
    subtitle: "Detected data integrity flaws, missingness spikes, and suggested actions.",
    num: "SECTION 05",
    accent: c-amber-600
  )

  #if "summary" in ledger and ledger.summary != none [
    #text(size: 8pt, fill: c-slate-500)[
      *#ledger.summary.at("total", default: 0)* total issues flagged · *#ledger.summary.at("approved", default: 0)* remediations applied · *#ledger.summary.at("rejected", default: 0)* skipped
    ]
    #v(4pt)
  ]

  #let iss-rows = ()
  #for issue in ledger.issues.slice(0, calc.min(ledger.issues.len(), 18)) [
    #let col = issue.at("column", default: "Dataset")
    #let desc = issue.at("description", default: "—")
    #let sev = issue.at("severity", default: "medium")
    #let st = issue.at("status", default: "pending")
    #let fix = issue.at("suggested_fix", default: "—")

    #iss-rows.push([*#col*])
    #iss-rows.push(text(size: 7.5pt)[#desc])
    #iss-rows.push(badge-severity(sev))
    #iss-rows.push(badge-status(st))
    #iss-rows.push(text(size: 7.5pt, fill: c-slate-500)[#fix])
  ]

  #table(
    columns: (1.2fr, 2fr, 0.9fr, 0.9fr, 2fr),
    fill: (col, row) => if row == 0 { c-slate-900 } else if calc.even(row) { c-slate-50 } else { white },
    stroke: (col, row) => if row == 0 { none } else { (top: 0.5pt + c-border, bottom: none) },
    inset: (x: 8pt, y: 5pt),
    align: (col, row) => if row == 0 { center + horizon } else if col == 2 or col == 3 { center + horizon } else { left + horizon },
    table.header(
      text(fill: white, weight: "bold")[Target Column],
      text(fill: white, weight: "bold")[Identified Anomaly],
      text(fill: white, weight: "bold")[Severity],
      text(fill: white, weight: "bold")[Status],
      text(fill: white, weight: "bold")[Remediation],
    ),
    ..iss-rows
  )
]

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 06 — DATA CLEANING & QUALITY DELTAS
// ═══════════════════════════════════════════════════════════════════════════════

#if "cleaning_report" in analysis and analysis.cleaning_report != none [
  #let cr = analysis.cleaning_report
  #section-title(
    "Data Cleaning & Net Quality Deltas",
    subtitle: "Autonomous cleaning operations applied and before/after quality metrics.",
    num: "SECTION 06",
    accent: c-emerald-600
  )

  #if "steps_applied" in cr and cr.steps_applied != none and cr.steps_applied.len() > 0 [
    #let cl-rows = ()
    #for step in cr.steps_applied [
      #cl-rows.push([*#step.at("name", default: "—")*])
      #cl-rows.push([#step.at("action", default: "—")])
      #cl-rows.push(text(size: 7.5pt, fill: c-slate-500)[#step.at("impact", default: "—")])
    ]
    #table(
      columns: (1.5fr, 2fr, 2fr),
      fill: (col, row) => if row == 0 { c-slate-900 } else if calc.even(row) { c-slate-50 } else { white },
      stroke: (col, row) => if row == 0 { none } else { (top: 0.5pt + c-border, bottom: none) },
      inset: (x: 8pt, y: 5pt),
      table.header(
        text(fill: white, weight: "bold")[Cleaning Step],
        text(fill: white, weight: "bold")[Action Executed],
        text(fill: white, weight: "bold")[Dataset Impact],
      ),
      ..cl-rows
    )
  ]

  #if "before_after" in cr and cr.before_after != none and cr.before_after.len() > 0 [
    #v(6pt)
    #text(weight: "bold", size: 8.5pt)[Quality Improvement (Before vs. After):]
    #v(3pt)
    #let ba-rows = ()
    #for item in cr.before_after [
      #let met = item.at("metric", default: "—")
      #let bef = item.at("before", default: "—")
      #let aft = item.at("after", default: "—")
      #let delta = item.at("delta", default: 0)

      #ba-rows.push([*#met*])
      #ba-rows.push([#bef])
      #ba-rows.push([#aft])
      #ba-rows.push(
        if delta > 0 {
          text(fill: c-emerald-600, weight: "bold")[▲ +#delta]
        } else if delta < 0 {
          text(fill: c-rose-600, weight: "bold")[▼ #delta]
        } else {
          text(fill: c-slate-500)[—]
        }
      )
    ]
    #table(
      columns: (2fr, 1.2fr, 1.2fr, 1.2fr),
      fill: (col, row) => if row == 0 { c-slate-900 } else if calc.even(row) { c-slate-50 } else { white },
      stroke: (col, row) => if row == 0 { none } else { (top: 0.5pt + c-border, bottom: none) },
      inset: (x: 8pt, y: 5pt),
      align: (col, row) => if row == 0 { center + horizon } else if col >= 1 { center + horizon } else { left + horizon },
      table.header(
        text(fill: white, weight: "bold")[Metric],
        text(fill: white, weight: "bold")[Before],
        text(fill: white, weight: "bold")[After],
        text(fill: white, weight: "bold")[Net Delta],
      ),
      ..ba-rows
    )
  ]
]

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 07 — SUMMARY STATISTICS & ADVANCED DISTRIBUTIONS
// ═══════════════════════════════════════════════════════════════════════════════

#if "summary" in analysis and analysis.summary != none and analysis.summary.len() > 0 [
  #section-title(
    "Summary Statistics & Feature Distributions",
    subtitle: "Parametric metrics, spread, and non-normal skewness analysis.",
    num: "SECTION 07",
    accent: c-slate-900
  )

  #let st-rows = ()
  #for (cname, st) in analysis.summary [
    #st-rows.push([*#cname*])
    #st-rows.push([#fmt-num(st.at("mean", default: none))])
    #st-rows.push([#fmt-num(st.at("50%", default: none))])
    #st-rows.push([#fmt-num(st.at("std", default: none))])
    #st-rows.push([#fmt-num(st.at("min", default: none))])
    #st-rows.push([#fmt-num(st.at("max", default: none))])
    #st-rows.push([#fmt-pct(st.at("missing_pct", default: 0))])
  ]

  #table(
    columns: (1.5fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    fill: (col, row) => if row == 0 { c-slate-900 } else if calc.even(row) { c-slate-50 } else { white },
    stroke: (col, row) => if row == 0 { none } else { (top: 0.5pt + c-border, bottom: none) },
    inset: (x: 8pt, y: 5pt),
    align: (col, row) => if row == 0 { center + horizon } else if col >= 1 { center + horizon } else { left + horizon },
    table.header(
      text(fill: white, weight: "bold")[Column],
      text(fill: white, weight: "bold")[Mean],
      text(fill: white, weight: "bold")[Median],
      text(fill: white, weight: "bold")[Std Dev],
      text(fill: white, weight: "bold")[Min],
      text(fill: white, weight: "bold")[Max],
      text(fill: white, weight: "bold")[Null %],
    ),
    ..st-rows
  )

  // Non-Normal Distributions
  #let skewed = ()
  #for (cname, st) in analysis.summary [
    #let sk = st.at("skewness", default: none)
    #if sk != none and (sk > 0.75 or sk < -0.75) [
      #skewed.push((col: cname, skewness: sk, kurtosis: st.at("kurtosis", default: 0)))
    ]
  ]

  #if skewed.len() > 0 [
    #v(6pt)
    #text(weight: "bold", size: 8.5pt)[Non-Normal Feature Distributions (|skewness| > 0.75):]
    #v(3pt)
    #let sk-rows = ()
    #for item in skewed [
      #let dir = if item.skewness > 0 { "Right-skewed" } else { "Left-skewed" }
      #sk-rows.push([*#item.col*])
      #sk-rows.push([#fmt-num(item.skewness, digits: 3)])
      #sk-rows.push([#fmt-num(item.kurtosis, digits: 3)])
      #sk-rows.push(badge(dir, bg: c-amber-50, fg: c-amber-600))
    ]
    #table(
      columns: (1.5fr, 1fr, 1fr, 1.2fr),
      fill: (col, row) => if row == 0 { c-slate-900 } else if calc.even(row) { c-slate-50 } else { white },
      stroke: (col, row) => if row == 0 { none } else { (top: 0.5pt + c-border, bottom: none) },
      inset: (x: 8pt, y: 5pt),
      align: (col, row) => if row == 0 { center + horizon } else if col >= 1 { center + horizon } else { left + horizon },
      table.header(
        text(fill: white, weight: "bold")[Feature],
        text(fill: white, weight: "bold")[Skewness],
        text(fill: white, weight: "bold")[Kurtosis],
        text(fill: white, weight: "bold")[Direction],
      ),
      ..sk-rows
    )
  ]
]

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 08 — STRONG CORRELATIONS & MULTICOLLINEARITY
// ═══════════════════════════════════════════════════════════════════════════════

#if "strong_correlations" in analysis and analysis.strong_correlations != none and analysis.strong_correlations.len() > 0 [
  #section-title(
    "Correlation Intelligence & Multicollinearity",
    subtitle: "Bivariate Pearson associations and high-redundancy warnings.",
    num: "SECTION 08",
    accent: c-blue-600
  )

  #let corrs = analysis.strong_correlations
  #let top-pos = corrs.filter(c => c.at("direction", default: "") == "positive")
  #let top-neg = corrs.filter(c => c.at("direction", default: "") == "negative")

  #if top-pos.len() > 0 or top-neg.len() > 0 [
    #let msg = []
    #if top-pos.len() > 0 [
      #let p = top-pos.at(0)
      #let ca = p.at("column_a", default: p.at("col_a", default: ""))
      #let cb = p.at("column_b", default: p.at("col_b", default: ""))
      #let rval = fmt-num(calc.abs(p.at("r_value", default: p.at("value", default: 0))), digits: 2)
      #msg = msg + [*Strongest Positive Correlation:* #emph(ca) and #emph(cb) co-vary strongly (r = +#rval). Higher values in one feature reliably track higher values in the other.]
    ]
    #if top-neg.len() > 0 [
      #let n = top-neg.at(0)
      #let ca = n.at("column_a", default: n.at("col_a", default: ""))
      #let cb = n.at("column_b", default: n.at("col_b", default: ""))
      #let rval = fmt-num(n.at("r_value", default: n.at("value", default: 0)), digits: 2)
      #if top-pos.len() > 0 { msg = msg + [\ ] }
      #msg = msg + [*Strongest Inverse Relationship:* #emph(ca) and #emph(cb) move inversely (r = #rval). Increases in one feature accompany decreases in the other.]
    ]
    #callout(title: "Automated Correlation Intelligence", msg, border-color: c-blue-600, bg-color: c-blue-50)
  ]

  #let co-rows = ()
  #for pair in corrs.slice(0, calc.min(corrs.len(), 10)) [
    #let ca = pair.at("column_a", default: pair.at("col_a", default: "—"))
    #let cb = pair.at("column_b", default: pair.at("col_b", default: "—"))
    #let rval = pair.at("r_value", default: pair.at("value", default: 0))
    #let is-pos = rval > 0
    #co-rows.push([*#ca*])
    #co-rows.push([*#cb*])
    #co-rows.push([#fmt-num(rval, digits: 3)])
    #co-rows.push(
      if is-pos { badge("Positive", bg: c-emerald-50, fg: c-emerald-600) }
      else { badge("Negative", bg: c-rose-50, fg: c-rose-600) }
    )
  ]
  #table(
    columns: (2fr, 2fr, 1.2fr, 1.2fr),
    fill: (col, row) => if row == 0 { c-slate-900 } else if calc.even(row) { c-slate-50 } else { white },
    stroke: (col, row) => if row == 0 { none } else { (top: 0.5pt + c-border, bottom: none) },
    inset: (x: 8pt, y: 5pt),
    align: (col, row) => if row == 0 { center + horizon } else if col >= 2 { center + horizon } else { left + horizon },
    table.header(
      text(fill: white, weight: "bold")[Feature A],
      text(fill: white, weight: "bold")[Feature B],
      text(fill: white, weight: "bold")[Pearson r],
      text(fill: white, weight: "bold")[Relationship],
    ),
    ..co-rows
  )

  // Multicollinearity Warnings (|r| > 0.85)
  #let mc-pairs = corrs.filter(c => calc.abs(c.at("r_value", default: c.at("value", default: 0))) > 0.85)
  #if mc-pairs.len() > 0 [
    #v(6pt)
    #callout(
      title: "Multicollinearity Warning (|r| > 0.85)",
      [The following feature pairs exhibit critical linear dependency. Consider feature pruning or dimensionality reduction (PCA) prior to linear or logistic regression modeling.],
      border-color: c-rose-600,
      bg-color: c-rose-50
    )
    #let mc-rows = ()
    #for pair in mc-pairs [
      #let ca = pair.at("column_a", default: pair.at("col_a", default: "—"))
      #let cb = pair.at("column_b", default: pair.at("col_b", default: "—"))
      #let rval = fmt-num(calc.abs(pair.at("r_value", default: pair.at("value", default: 0))), digits: 3)
      #mc-rows.push([#ca])
      #mc-rows.push([#cb])
      #mc-rows.push(text(fill: c-rose-600, weight: "bold")[#rval])
    ]
    #table(
      columns: (2fr, 2fr, 1.2fr),
      fill: (col, row) => if row == 0 { c-slate-900 } else if calc.even(row) { c-slate-50 } else { white },
      stroke: (col, row) => if row == 0 { none } else { (top: 0.5pt + c-border, bottom: none) },
      inset: (x: 8pt, y: 5pt),
      align: (col, row) => if row == 0 { center + horizon } else if col == 2 { center + horizon } else { left + horizon },
      table.header(
        text(fill: white, weight: "bold")[Feature A],
        text(fill: white, weight: "bold")[Feature B],
        text(fill: white, weight: "bold")[|Correlation|],
      ),
      ..mc-rows
    )
  ]
]

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 09 — OUTLIER DETECTION & MISSINGNESS PATTERNS
// ═══════════════════════════════════════════════════════════════════════════════

#if "outliers" in analysis and analysis.outliers != none and analysis.outliers.len() > 0 [
  #section-title(
    "Outlier Risk Profiling (Tukey IQR Method)",
    subtitle: "Upper and lower fence calculations flagging anomalous records.",
    num: "SECTION 09",
    accent: c-amber-600
  )

  #let out-rows = ()
  #for (cname, odata) in analysis.outliers [
    #let out-pct = odata.at("percentage", default: 0)
    #out-rows.push([*#cname*])
    #out-rows.push([#odata.at("count", default: 0)])
    #out-rows.push([
      #grid(
        columns: (auto, 1fr),
        gutter: 4pt,
        align: horizon,
        text(size: 7.5pt, weight: "bold")[#fmt-pct(out-pct)],
        vector-progress-bar(out-pct, width: 35pt, height: 4.5pt, max-val: 20)
      )
    ])
    #out-rows.push([#fmt-num(odata.at("lower_bound", default: 0))])
    #out-rows.push([#fmt-num(odata.at("upper_bound", default: 0))])
  ]
  #table(
    columns: (1.8fr, 0.8fr, 1.4fr, 1.1fr, 1.1fr),
    fill: (col, row) => if row == 0 { c-slate-900 } else if calc.even(row) { c-slate-50 } else { white },
    stroke: (col, row) => if row == 0 { none } else { (top: 0.5pt + c-border, bottom: none) },
    inset: (x: 8pt, y: 5pt),
    align: (col, row) => if row == 0 { center + horizon } else if col >= 1 { center + horizon } else { left + horizon },
    table.header(
      text(fill: white, weight: "bold")[Column],
      text(fill: white, weight: "bold")[Outliers],
      text(fill: white, weight: "bold")[Outlier %],
      text(fill: white, weight: "bold")[Lower Fence],
      text(fill: white, weight: "bold")[Upper Fence],
    ),
    ..out-rows
  )
]

#if "missing_patterns" in analysis and analysis.missing_patterns != none and "columns" in analysis.missing_patterns [
  #let mp-cols = analysis.missing_patterns.columns
  #let mp-with-nulls = mp-cols.pairs().filter(p => p.at(1).at("missing_pct", default: 0) > 0)
  #if mp-with-nulls.len() > 0 [
    #v(8pt)
    #text(weight: "bold", size: 9pt)[Missing Value Patterns:]
    #v(3pt)
    #let mp-rows = ()
    #for (cname, mp) in mp-with-nulls [
      #let m-pct = mp.at("missing_pct", default: 0)
      #mp-rows.push([*#cname*])
      #mp-rows.push([#mp.at("missing_count", default: 0)])
      #mp-rows.push([
        #grid(
          columns: (auto, 1fr),
          gutter: 5pt,
          align: horizon,
          text(size: 7.5pt, weight: "bold")[#fmt-pct(m-pct)],
          vector-distribution-bar(100 - m-pct, missing-pct: m-pct, width: 45pt, height: 5pt)
        )
      ])
      #mp-rows.push(badge(mp.at("pattern", default: "random"), bg: c-slate-100, fg: c-slate-800))
    ]
    #table(
      columns: (1.8fr, 1fr, 1.6fr, 1.4fr),
      fill: (col, row) => if row == 0 { c-slate-900 } else if calc.even(row) { c-slate-50 } else { white },
      stroke: (col, row) => if row == 0 { none } else { (top: 0.5pt + c-border, bottom: none) },
      inset: (x: 8pt, y: 5pt),
      align: (col, row) => if row == 0 { center + horizon } else if col >= 1 { center + horizon } else { left + horizon },
      table.header(
        text(fill: white, weight: "bold")[Column Name],
        text(fill: white, weight: "bold")[Missing Count],
        text(fill: white, weight: "bold")[Missing Ratio],
        text(fill: white, weight: "bold")[Observed Pattern],
      ),
      ..mp-rows
    )
  ]
]

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 10 — FEATURE ENGINEERING & AI INSIGHTS
// ═══════════════════════════════════════════════════════════════════════════════

#if "feature_engineering" in analysis and analysis.feature_engineering != none [
  #let fe = analysis.feature_engineering
  #section-title(
    "Feature Engineering & Transformation Advisory",
    subtitle: "Prescriptive recommendations for encoding, normalization, and scaling.",
    num: "SECTION 10",
    accent: c-emerald-600
  )

  #if "encoding_recommendations" in fe and fe.encoding_recommendations != none and fe.encoding_recommendations.len() > 0 [
    #text(weight: "bold", size: 8.5pt)[Categorical Encoding Recommendations:]
    #v(3pt)
    #let enc-rows = ()
    #for rec in fe.encoding_recommendations [
      #enc-rows.push([*#rec.at("column", default: "—")*])
      #enc-rows.push([#rec.at("current_type", default: "—")])
      #enc-rows.push(badge(rec.at("encoding", default: "—"), bg: c-blue-50, fg: c-blue-600))
      #enc-rows.push(text(size: 7.5pt, fill: c-slate-500)[#rec.at("reason", default: "—")])
    ]
    #table(
      columns: (1.5fr, 1.2fr, 1.5fr, 2.5fr),
      fill: (col, row) => if row == 0 { c-slate-900 } else if calc.even(row) { c-slate-50 } else { white },
      stroke: (col, row) => if row == 0 { none } else { (top: 0.5pt + c-border, bottom: none) },
      inset: (x: 8pt, y: 5pt),
      table.header(
        text(fill: white, weight: "bold")[Column],
        text(fill: white, weight: "bold")[Current Type],
        text(fill: white, weight: "bold")[Suggested Encoding],
        text(fill: white, weight: "bold")[Rationale],
      ),
      ..enc-rows
    )
  ]

  #if "scaling_recommendations" in fe and fe.scaling_recommendations != none and fe.scaling_recommendations.len() > 0 [
    #v(6pt)
    #text(weight: "bold", size: 8.5pt)[Feature Scaling Recommendations:]
    #v(3pt)
    #let sc-rows = ()
    #for rec in fe.scaling_recommendations [
      #sc-rows.push([*#rec.at("column", default: "—")*])
      #sc-rows.push(badge(rec.at("scaler", default: "—"), bg: c-emerald-50, fg: c-emerald-600))
      #sc-rows.push(text(size: 7.5pt, fill: c-slate-500)[#rec.at("reason", default: "—")])
    ]
    #table(
      columns: (1.5fr, 1.5fr, 3fr),
      fill: (col, row) => if row == 0 { c-slate-900 } else if calc.even(row) { c-slate-50 } else { white },
      stroke: (col, row) => if row == 0 { none } else { (top: 0.5pt + c-border, bottom: none) },
      inset: (x: 8pt, y: 5pt),
      table.header(
        text(fill: white, weight: "bold")[Column],
        text(fill: white, weight: "bold")[Suggested Scaler],
        text(fill: white, weight: "bold")[Rationale],
      ),
      ..sc-rows
    )
  ]
]

// AI Insights Callouts
#if "insights" in analysis and analysis.insights != none [
  #let ins = analysis.insights
  #v(8pt)
  #text(weight: "bold", size: 9pt)[AI Synthesis & Strategic Intelligence:]
  #v(4pt)
  #if type(ins) == str [
    #callout(title: "Executive Synthesis", ins, border-color: c-emerald-600, bg-color: c-emerald-50)
  ] else if type(ins) == dictionary [
    #if "insights_text" in ins and ins.insights_text != none [
      #callout(title: "Executive Synthesis", ins.insights_text, border-color: c-emerald-600, bg-color: c-emerald-50)
    ] else [
      #for (key, val) in ins [
        #if key not in ("model_used", "prompt_tokens", "completion_tokens", "total_tokens", "response_time_ms", "retries_attempted", "success") [
          #let vstr = if type(val) == str { val } else { str(val) }
          #callout(title: upper(key.replace("_", " ")), vstr, border-color: c-blue-600)
        ]
      ]
    ]
  ]
]

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 11 — ACTIONABLE RECOMMENDATIONS
// ═══════════════════════════════════════════════════════════════════════════════

#if "recommendations" in analysis and analysis.recommendations != none [
  #let recs = analysis.recommendations
  #let all-recs = ()
  #for k in ("domain_specific", "data_quality", "analysis", "visualization") [
    #if k in recs and recs.at(k) != none [
      #for item in recs.at(k) [
        #all-recs.push(item)
      ]
    ]
  ]

  #if all-recs.len() > 0 [
    #section-title(
      "Strategic Remediations & Recommendations",
      subtitle: "Prioritized steps for engineering teams and analytical stakeholders.",
      num: "SECTION 11",
      accent: c-emerald-600
    )
    #for r in all-recs.slice(0, calc.min(all-recs.len(), 8)) [
      #let title = r.at("title", default: "Recommendation")
      #let prio = r.at("priority", default: "Medium")
      #let desc = r.at("description", default: r.at("detail", default: ""))
      #let act = r.at("action", default: none)

      #block(
        fill: c-slate-50,
        stroke: (left: 3.5pt + if lower(prio) == "high" { c-rose-600 } else if lower(prio) == "low" { c-emerald-600 } else { c-amber-600 }, rest: 0.5pt + c-border),
        radius: (right: 4pt),
        inset: 10pt,
        [
          #grid(
            columns: (1fr, auto),
            text(weight: "bold", size: 9pt, fill: c-slate-900)[#title],
            badge-severity(prio)
          )
          #v(2pt)
          #text(size: 8pt, fill: c-slate-700)[#desc]
          #if act != none [
            #v(2pt)
            #text(size: 7.5pt, weight: "bold", fill: c-blue-600)[Action Required: #act]
          ]
        ]
      )
      #v(4pt)
    ]
  ]
]

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 12 — DATA VISUALIZATION GALLERY
// ═══════════════════════════════════════════════════════════════════════════════

#if chart_list != none and chart_list.len() > 0 [
  #section-title(
    "Data Visualizations & Statistical Graphics",
    subtitle: "High-resolution distribution plots, correlation matrices, and spread diagrams.",
    num: "SECTION 12",
    accent: c-slate-900
  )

  #for ch in chart_list [
    #let cname = ch.at("title", default: "Chart")
    #let cpath = ch.at("path", default: "")
    #let narr = ch.at("narrative", default: none)

    #block(
      width: 100%,
      stroke: 0.5pt + c-border,
      radius: 4pt,
      inset: 10pt,
      fill: white,
      [
        #text(weight: "bold", size: 10pt, fill: c-slate-900)[#cname]
        #v(6pt)
        #align(center)[#image(cpath, width: 94%)]
        #if narr != none [
          #v(6pt)
          #block(
            fill: c-slate-50,
            stroke: (left: 2.5pt + c-blue-600, rest: 0.5pt + c-border),
            radius: (right: 3pt),
            inset: (x: 8pt, y: 6pt),
            [#text(size: 7.5pt, fill: c-slate-800)[*Key Analytical Takeaway:* #narr]]
          )
        ]
      ]
    )
    #v(12pt)
  ]
]

// ═══════════════════════════════════════════════════════════════════════════════
// REPORT END / AUDIT CERTIFICATION
// ═══════════════════════════════════════════════════════════════════════════════

#v(20pt)
#line(length: 100%, stroke: 0.5pt + c-border)
#v(8pt)
#align(center)[
  #text(size: 9pt, weight: "bold", fill: c-slate-900)[GETREPORT DATA INTELLIGENCE PLATFORM]\
  #text(size: 7.5pt, fill: c-slate-500)[Automated Audit & Profiling Engine • Generated on #generated_at]\
  #text(size: 7pt, fill: c-slate-500)[
    Compiled with the zero-copy, memory-safe Typst Rust compilation engine • under 25MB Peak RAM footprint
  ]
]
