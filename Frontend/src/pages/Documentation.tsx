import { useState, useEffect, useMemo } from "react";
import { Link } from "react-router-dom";
import { 
  FileSpreadsheet, Gauge, ShieldCheck, Code2, Copy, 
  Check, Layers, Database, Terminal, Search, CheckCircle2, 
  ArrowRight
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";
import { CtaBanner } from "@/components/CtaBanner";
import { useToast } from "@/hooks/use-toast";

interface EndpointSpec {
  id: string;
  category: "ingestion" | "audit" | "duckdb" | "analytics" | "companion" | "export";
  method: "POST" | "GET" | "DELETE";
  path: string;
  title: string;
  description: string;
  curlExample: string;
  pythonExample: string;
  tsExample: string;
  requestBody?: string;
  responseBody: string;
  headers?: Record<string, string>;
}

const API_ENDPOINTS: EndpointSpec[] = [
  {
    id: "upload",
    category: "ingestion",
    method: "POST",
    path: "/api/upload",
    title: "Zero-Copy Tabular File Ingestion",
    description: "Memory-maps uploaded CSV, Excel, Parquet, or JSONL files directly into Polars ChunkedArrays. Automatically decodes BOM/CP1252/UTF-8 encodings and returns inferred Arrow schemas.",
    headers: {
      "Content-Type": "multipart/form-data"
    },
    curlExample: `curl -X POST https://getreport.onrender.com/api/upload \\
  -F "file=@transactions_q3.parquet"`,
    pythonExample: `import requests

url = "https://getreport.onrender.com/api/upload"
with open("transactions_q3.parquet", "rb") as f:
    response = requests.post(url, files={"file": f})

data = response.json()
print("Task ID:", data["task_id"])
print("Rows detected:", data["info"]["rows"])`,
    tsExample: `const formData = new FormData();
formData.append("file", fileInput.files[0]);

const res = await fetch("https://getreport.onrender.com/api/upload", {
  method: "POST",
  body: formData,
});
const { task_id, info } = await res.json();
console.log(\`Task \${task_id} ready with \${info.rows} rows\`);`,
    requestBody: `// Multipart Form Data Payload
file: BinaryFile (e.g., .csv, .xlsx, .parquet, .jsonl)`,
    responseBody: `{
  "task_id": "job_9481ad2b",
  "filename": "transactions_q3.parquet",
  "status": "ready",
  "info": {
    "rows": 248500,
    "columns": ["tx_id", "user_id", "amount", "timestamp", "category"],
    "dtypes": {
      "tx_id": "Int64",
      "user_id": "Utf8",
      "amount": "Float64",
      "timestamp": "Datetime[ns]",
      "category": "Categorical"
    }
  },
  "cleaning_report": {
    "exact_duplicates_purged": 18,
    "encoding": "UTF-8 (Arrow Zero-Copy)"
  }
}`
  },
  {
    id: "generate-report",
    category: "audit",
    method: "POST",
    path: "/api/generate-report",
    title: "Execute Remediation DAG & Generate Audit Report",
    description: "Applies approved issue transformations across the dataset DAG, re-computes confidence matrices, and compiles an executive audit report with PDF export readiness.",
    headers: {
      "Content-Type": "application/json"
    },
    curlExample: `curl -X POST https://getreport.onrender.com/api/generate-report \\
  -H "Content-Type: application/json" \\
  -d '{
    "task_id": "job_9481ad2b",
    "decisions": {
      "issue_dedup": "approve",
      "issue_currency_clean": "approve",
      "issue_outlier_cap": "reject"
    }
  }'`,
    pythonExample: `import requests

payload = {
    "task_id": "job_9481ad2b",
    "decisions": {
        "issue_dedup": "approve",
        "issue_currency_clean": "approve"
    }
}
res = requests.post("https://getreport.onrender.com/api/generate-report", json=payload)
report = res.json()
print("Overall Confidence Score:", report["dataset_confidence_score"])`,
    tsExample: `const res = await fetch("https://getreport.onrender.com/api/generate-report", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    task_id: "job_9481ad2b",
    decisions: { issue_dedup: "approve" }
  })
});
const report = await res.json();
console.log("Audit Confidence:", report.dataset_confidence_score);`,
    requestBody: `{
  "task_id": "job_9481ad2b",
  "decisions": {
    "issue_dedup": "approve",
    "issue_currency_clean": "approve",
    "issue_outlier_cap": "reject"
  }
}`,
    responseBody: `{
  "status": "completed",
  "dataset_confidence_score": 96.4,
  "confidence_grade": "A",
  "pillars": {
    "completeness": 98.1,
    "consistency": 95.4,
    "validity": 96.0,
    "stability": 96.1
  },
  "columns_audited": 5,
  "download_url": "/api/jobs/job_9481ad2b/export/pdf"
}`
  },
  {
    id: "query-sql",
    category: "duckdb",
    method: "POST",
    path: "/api/jobs/{task_id}/query-sql",
    title: "DuckDB Analytical In-Process SQL Query",
    description: "Executes hardware-accelerated SQL queries against the cleaned in-memory Parquet dataset via DuckDB. Enforces read-only sandboxing and Jinja/mustache parameter bindings.",
    headers: {
      "Content-Type": "application/json"
    },
    curlExample: `curl -X POST https://getreport.onrender.com/api/jobs/job_9481ad2b/query-sql \\
  -H "Content-Type: application/json" \\
  -d '{
    "sql": "SELECT category, COUNT(*) as cnt, ROUND(AVG(amount), 2) as avg_spend FROM dataset GROUP BY category ORDER BY avg_spend DESC LIMIT 10",
    "max_rows": 100
  }'`,
    pythonExample: `import requests

url = "https://getreport.onrender.com/api/jobs/job_9481ad2b/query-sql"
query = """
SELECT category, AVG(amount) AS avg_amt
FROM dataset
GROUP BY category
ORDER BY avg_amt DESC
"""
res = requests.post(url, json={"sql": query, "max_rows": 50})
results = res.json()
print("Execution Time:", results["duration_ms"], "ms")
for row in results["data"]:
    print(row)`,
    tsExample: `const res = await fetch("/api/jobs/job_9481ad2b/query-sql", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    sql: "SELECT * FROM dataset WHERE amount > {{ min_val }}",
    params: { min_val: 500 }
  })
});
const { data, duration_ms } = await res.json();
console.log(\`Fetched \${data.length} records in \${duration_ms}ms\`);`,
    requestBody: `{
  "sql": "SELECT category, AVG(amount) as avg_amt FROM dataset WHERE category = {{ target_cat }} GROUP BY category",
  "params": {
    "target_cat": "Electronics"
  },
  "max_rows": 1000
}`,
    responseBody: `{
  "success": true,
  "duration_ms": 2.14,
  "row_count": 1,
  "columns": ["category", "avg_amt"],
  "column_types": ["VARCHAR", "DOUBLE"],
  "data": [
    { "category": "Electronics", "avg_amt": 412.85 }
  ]
}`
  },
  {
    id: "sql-schema",
    category: "duckdb",
    method: "GET",
    path: "/api/jobs/{task_id}/sql-schema",
    title: "SQL Lab Schema Tree Introspection",
    description: "Inspects all active in-memory tables, columns, physical data types, and preview samples for SQL Lab editor autocomplete and schema tree explorers.",
    curlExample: `curl -X GET https://getreport.onrender.com/api/jobs/job_9481ad2b/sql-schema`,
    pythonExample: `import requests

res = requests.get("https://getreport.onrender.com/api/jobs/job_9481ad2b/sql-schema")
schema = res.json()
for tbl in schema["tables"]:
    print(f"Table {tbl['table_name']}: {tbl['row_count']} rows, {tbl['column_count']} columns")`,
    tsExample: `const res = await fetch("/api/jobs/job_9481ad2b/sql-schema");
const schema = await res.json();
console.log("Registered tables:", schema.total_tables);`,
    responseBody: `{
  "engine": "DuckDB-InProcess",
  "total_tables": 1,
  "tables": [
    {
      "table_name": "dataset",
      "row_count": 248500,
      "column_count": 5,
      "columns": [
        { "name": "tx_id", "type": "BIGINT", "nullable": false, "sample_values": ["101", "102", "103"] },
        { "name": "amount", "type": "DOUBLE", "nullable": true, "sample_values": ["49.99", "120.00", "15.50"] },
        { "name": "category", "type": "VARCHAR", "nullable": true, "sample_values": ["Retail", "Tech", "Food"] }
      ]
    }
  ]
}`
  },
  {
    id: "explore-query",
    category: "analytics",
    method: "POST",
    path: "/api/analytics/explore/query",
    title: "Superset Explore Studio Aggregation Query",
    description: "Computes ad-hoc multi-dimensional metric aggregations, group-bys, and distribution histograms for Apache Superset BI visualizations.",
    headers: {
      "Content-Type": "application/json"
    },
    curlExample: `curl -X POST https://getreport.onrender.com/api/analytics/explore/query \\
  -H "Content-Type: application/json" \\
  -d '{
    "task_id": "job_9481ad2b",
    "metrics": [{ "aggregate": "SUM", "column": "amount" }],
    "groupby": ["category"],
    "row_limit": 50
  }'`,
    pythonExample: `import requests

payload = {
    "task_id": "job_9481ad2b",
    "metrics": [{"aggregate": "AVG", "column": "amount"}],
    "groupby": ["category"],
    "order_by": [{"column": "amount", "direction": "DESC"}]
}
res = requests.post("https://getreport.onrender.com/api/analytics/explore/query", json=payload)
print(res.json()["data"])`,
    tsExample: `const res = await fetch("/api/analytics/explore/query", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    task_id: "job_9481ad2b",
    metrics: [{ aggregate: "COUNT", column: "*" }],
    groupby: ["category"]
  })
});
const { data } = await res.json();`,
    requestBody: `{
  "task_id": "job_9481ad2b",
  "metrics": [
    { "aggregate": "SUM", "column": "amount" },
    { "aggregate": "COUNT", "column": "tx_id" }
  ],
  "groupby": ["category"],
  "order_by": [{ "column": "amount", "direction": "DESC" }],
  "row_limit": 100
}`,
    responseBody: `{
  "success": true,
  "chart_type": "bar",
  "row_count": 3,
  "data": [
    { "category": "Enterprise", "SUM_amount": 1420500.0, "COUNT_tx_id": 850 },
    { "category": "Retail", "SUM_amount": 894000.0, "COUNT_tx_id": 12400 },
    { "category": "Services", "SUM_amount": 341200.0, "COUNT_tx_id": 4100 }
  ]
}`
  },
  {
    id: "chat-stream",
    category: "companion",
    method: "POST",
    path: "/api/jobs/{task_id}/chat/stream",
    title: "Streaming Ephemeral RAG Companion",
    description: "Server-Sent Events (SSE) streaming endpoint answering questions regarding column skewness, outlier clusters, statistical anomalies, or correlation drivers.",
    headers: {
      "Content-Type": "application/json",
      "Accept": "text/event-stream"
    },
    curlExample: `curl -N -X POST https://getreport.onrender.com/api/jobs/job_9481ad2b/chat/stream \\
  -H "Content-Type: application/json" \\
  -d '{"query": "Why did the validity score drop in column transaction_date?"}'`,
    pythonExample: `import requests

url = "https://getreport.onrender.com/api/jobs/job_9481ad2b/chat/stream"
res = requests.post(url, json={"query": "Which columns have the highest null rate?"}, stream=True)

for line in res.iter_lines():
    if line:
        print(line.decode("utf-8"))`,
    tsExample: `const response = await fetch("/api/jobs/job_9481ad2b/chat/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query: "Summarize dataset health." })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  console.log(decoder.decode(value));
}`,
    requestBody: `{
  "query": "Which columns exhibit the strongest bivariate correlations?",
  "top_k": 3
}`,
    responseBody: `data: {"chunk": "Based on the correlation matrix, "}
data: {"chunk": "'amount' and 'tax_rate' show a high Pearson coefficient of r = 0.88 "}
data: {"chunk": "(p < 0.001). Conversely, 'user_age' exhibits near-zero correlation."}
data: {"done": true, "sources": [{"pillar": "stability", "metric": "correlation"}]}`
  },
  {
    id: "export-gx",
    category: "export",
    method: "GET",
    path: "/api/jobs/{task_id}/export-gx",
    title: "Great Expectations Suite Compilation",
    description: "Compiles all audited hygiene constraints (null bounds, schema expectations, value sets, date formats) into a production Great Expectations JSON suite.",
    curlExample: `curl -X GET https://getreport.onrender.com/api/jobs/job_9481ad2b/export-gx \\
  -o great_expectations_suite.json`,
    pythonExample: `import requests
import json

res = requests.get("https://getreport.onrender.com/api/jobs/job_9481ad2b/export-gx")
gx_suite = res.json()

with open("gx_suite.json", "w") as f:
    json.dump(gx_suite, f, indent=2)
print("Saved GX Suite with", len(gx_suite["expectations"]), "expectations")`,
    tsExample: `const res = await fetch("/api/jobs/job_9481ad2b/export-gx");
const gxSuite = await res.json();
console.log("Generated GX Suite:", gxSuite.expectation_suite_name);`,
    responseBody: `{
  "expectation_suite_name": "getreport_audit_job_9481ad2b",
  "ge_cloud_id": null,
  "expectations": [
    {
      "expectation_type": "expect_column_values_to_not_be_null",
      "kwargs": { "column": "tx_id" }
    },
    {
      "expectation_type": "expect_column_values_to_be_between",
      "kwargs": { "column": "amount", "min_value": 0.0, "max_value": 150000.0 }
    },
    {
      "expectation_type": "expect_column_values_to_be_in_set",
      "kwargs": { "column": "category", "value_set": ["Retail", "Tech", "Food", "Services"] }
    }
  ]
}`
  },
  {
    id: "export-report",
    category: "export",
    method: "GET",
    path: "/api/jobs/{task_id}/export/{format}",
    title: "Multi-Format Audit Export Engine",
    description: "Downloads the compiled audit dossier in requested format (pdf, html, xlsx, or json) with executive summary tables and provenance metadata.",
    curlExample: `curl -X GET https://getreport.onrender.com/api/jobs/job_9481ad2b/export/pdf \\
  -o executive_audit_report.pdf`,
    pythonExample: `import requests

res = requests.get("https://getreport.onrender.com/api/jobs/job_9481ad2b/export/xlsx")
with open("cleaned_audit.xlsx", "wb") as f:
    f.write(res.content)
print("Exported Excel dossier successfully")`,
    tsExample: `const format = "pdf";
const response = await fetch(\`/api/jobs/job_9481ad2b/export/\${format}\`);
const blob = await response.blob();
const downloadUrl = window.URL.createObjectURL(blob);
const a = document.createElement("a");
a.href = downloadUrl;
a.download = \`audit_report.\${format}\`;
a.click();`,
    responseBody: `// Binary file stream returned with Content-Disposition attachment:
Content-Type: application/pdf | text/html | application/vnd.openxmlformats-officedocument.spreadsheetml.sheet | application/json`
  }
];

const SECTIONS = [
  { id: "supported-formats", title: "1. Ingestion & Formats", icon: FileSpreadsheet, tag: "IO" },
  { id: "confidence-scoring", title: "2. Confidence Scoring", icon: Gauge, tag: "MATH" },
  { id: "issue-dag", title: "3. Transformation DAG", icon: Layers, tag: "DAG" },
  { id: "duckdb-sql", title: "4. DuckDB & BI Studio", icon: Database, tag: "OLAP" },
  { id: "api-endpoints", title: "5. REST API Reference", icon: Code2, tag: "API" },
  { id: "security", title: "6. Security & Sandboxing", icon: ShieldCheck, tag: "SEC" }
];

export const Documentation = () => {
  const { toast } = useToast();
  const [activeSection, setActiveSection] = useState<string>("supported-formats");
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [selectedCategory, setSelectedCategory] = useState<string>("all");

  useEffect(() => {
    document.title = "Developer Documentation & Technical Reference | GetReport";
  }, []);

  // Active section scroll spy via IntersectionObserver
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.id);
          }
        });
      },
      {
        rootMargin: "-20% 0px -70% 0px",
        threshold: 0
      }
    );

    SECTIONS.forEach(({ id }) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });

    return () => observer.disconnect();
  }, []);

  const handleCopy = (text: string, identifier: string, label: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(identifier);
    toast({ 
      title: "Copied to Clipboard", 
      description: `${label} copied successfully.` 
    });
    setTimeout(() => setCopiedKey(null), 2000);
  };

  const filteredEndpoints = useMemo(() => {
    return API_ENDPOINTS.filter((ep) => {
      const matchesCategory = selectedCategory === "all" || ep.category === selectedCategory;
      const matchesSearch = 
        ep.path.toLowerCase().includes(searchQuery.toLowerCase()) ||
        ep.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        ep.description.toLowerCase().includes(searchQuery.toLowerCase());
      return matchesCategory && matchesSearch;
    });
  }, [searchQuery, selectedCategory]);

  return (
    <div className="min-h-screen flex flex-col bg-[#07080a] text-slate-100 selection:bg-white/20 selection:text-white font-sans antialiased">
      <Header onReset={() => {}} showReset={false} />

      <main className="flex-1 pt-20 sm:pt-24 pb-20">
        
        {/* Hero Section: Machined Raycast/Linear Obsidian Aesthetic */}
        <div className="relative border-b border-white/10 bg-radial from-white/[0.04] via-transparent to-transparent py-12 sm:py-16 overflow-hidden">
          {/* Subtle Ambient Background Mesh */}
          <div className="absolute inset-0 pointer-events-none opacity-40">
            <div className="absolute -top-32 left-1/2 -translate-x-1/2 w-[720px] h-[340px] bg-gradient-to-b from-white/10 via-transparent to-transparent blur-3xl rounded-full" />
          </div>

          <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-6xl relative z-10 space-y-6 text-center">
            
            {/* Eyebrow Pill */}
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/[0.05] border border-white/10 backdrop-blur-md shadow-xs">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
              <span className="font-mono text-[11px] font-semibold uppercase tracking-wider text-slate-300">
                GetReport Technical Hub • Core v2.4.0
              </span>
            </div>

            {/* Main Headline */}
            <h1 className="text-3xl sm:text-5xl md:text-6xl font-display font-extrabold tracking-[-0.03em] text-white leading-[1.08] max-w-4xl mx-auto">
              Engine architecture &amp; <span className="text-slate-400">REST API specification.</span>
            </h1>

            {/* Subhead */}
            <p className="text-sm sm:text-base md:text-lg text-slate-400 max-w-2xl mx-auto leading-relaxed font-sans font-normal">
              Zero-copy Polars ingestion, in-process DuckDB OLAP execution, 4-pillar confidence mathematics, and headless endpoints for production data pipelines.
            </p>

            {/* Quick Action Ribbon */}
            <div className="pt-2 flex flex-wrap items-center justify-center gap-3">
              <Button
                asChild
                variant="launch"
                size="sm"
                className="h-10 px-5 rounded-xl font-display font-bold text-xs gap-2 group cursor-pointer"
              >
                <Link to="/workspace">
                  <span>Interactive Workspace</span>
                  <ArrowRight className="h-3.5 w-3.5 text-black transition-transform duration-200 group-hover:translate-x-0.5" />
                </Link>
              </Button>

              <Button
                asChild
                variant="outline"
                size="sm"
                className="h-10 px-4 rounded-xl border-white/15 bg-white/[0.04] text-slate-200 hover:bg-white/[0.08] hover:text-white transition-all text-xs font-mono gap-2"
              >
                <a href="#api-endpoints">
                  <Terminal className="h-3.5 w-3.5 text-slate-400" />
                  <span>Jump to REST Endpoints</span>
                </a>
              </Button>
            </div>

            {/* Quick Filter Search Bar */}
            <div className="pt-4 max-w-xl mx-auto">
              <div className="relative">
                <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
                <Input
                  type="text"
                  placeholder="Filter endpoints, SQL specs, or quality algorithms (e.g., 'parquet', 'duckdb', 'gx')..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="h-11 pl-10 pr-4 bg-[#0c0d12]/90 border border-white/15 rounded-xl text-xs sm:text-sm text-slate-200 placeholder:text-slate-500 focus-visible:ring-1 focus-visible:ring-white/40 focus-visible:border-white/30 transition-all font-mono shadow-inner"
                />
                {searchQuery && (
                  <button
                    onClick={() => setSearchQuery("")}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-xs font-mono text-slate-500 hover:text-white px-1.5 py-0.5 rounded bg-white/5"
                  >
                    Clear
                  </button>
                )}
              </div>
            </div>

          </div>
        </div>

        {/* Main Content Grid: Sticky Sidebar + Structured Chapters */}
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl pt-8 sm:pt-12">
          
          {/* Mobile Quick-Jump Horizontal Strip (< lg) */}
          <div className="lg:hidden sticky top-16 z-30 bg-[#07080a]/95 backdrop-blur-xl border-b border-white/10 py-2.5 -mx-4 px-4 sm:-mx-6 sm:px-6 mb-6">
            <div className="flex items-center gap-2 overflow-x-auto scrollbar-none py-1">
              {SECTIONS.map(({ id, title, tag }) => {
                const isSelected = activeSection === id;
                return (
                  <a
                    key={id}
                    href={`#${id}`}
                    className={`px-3 py-1.5 rounded-lg text-xs font-mono whitespace-nowrap transition-all flex items-center gap-1.5 shrink-0 ${
                      isSelected
                        ? "bg-white text-black font-bold shadow-xs"
                        : "bg-white/[0.05] text-slate-300 hover:bg-white/[0.08] hover:text-white border border-white/5"
                    }`}
                  >
                    <span className="text-[10px] font-bold opacity-60">[{tag}]</span>
                    <span>{title.split(". ")[1] || title}</span>
                  </a>
                );
              })}
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-10">
            
            {/* Desktop Left Sticky Navigation (4 Cols) */}
            <div className="hidden lg:block lg:col-span-4">
              <div className="sticky top-24 space-y-4">
                
                {/* Outer Double-Bezel Container */}
                <div className="rounded-2xl p-1 bg-white/[0.03] border border-white/10 shadow-premium">
                  <div className="rounded-[calc(1rem-2px)] bg-[#0c0d12] p-4 space-y-3">
                    
                    <div className="flex items-center justify-between pb-2.5 border-b border-white/10">
                      <span className="font-mono text-[11px] uppercase tracking-wider font-bold text-slate-400 flex items-center gap-2">
                        <Terminal className="h-3.5 w-3.5 text-slate-300" />
                        <span>Document Index</span>
                      </span>
                      <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/20">
                        Live Tracking
                      </span>
                    </div>

                    <nav className="space-y-1">
                      {SECTIONS.map(({ id, title, icon: NavIcon, tag }) => {
                        const isCurrent = activeSection === id;
                        return (
                          <a
                            key={id}
                            href={`#${id}`}
                            className={`flex items-center justify-between px-3 py-2 rounded-xl text-xs font-mono transition-all duration-150 group ${
                              isCurrent
                                ? "bg-white/10 text-white font-bold border border-white/15 shadow-inner"
                                : "text-slate-400 hover:text-white hover:bg-white/[0.04]"
                            }`}
                          >
                            <div className="flex items-center gap-2.5">
                              <NavIcon className={`h-4 w-4 shrink-0 transition-colors ${
                                isCurrent ? "text-emerald-400" : "text-slate-500 group-hover:text-slate-300"
                              }`} />
                              <span>{title}</span>
                            </div>
                            <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                              isCurrent ? "bg-white/15 text-white" : "text-slate-600 group-hover:text-slate-400"
                            }`}>
                              {tag}
                            </span>
                          </a>
                        );
                      })}
                    </nav>

                  </div>
                </div>

                {/* System Specs Snapshot Card */}
                <div className="rounded-2xl p-1 bg-white/[0.03] border border-white/10">
                  <div className="rounded-[calc(1rem-2px)] bg-[#0c0d12] p-4 text-xs font-mono space-y-2 text-slate-400">
                    <span className="font-bold text-white text-[11px] uppercase tracking-wider block pb-1 border-b border-white/10">
                      Runtime Invariants
                    </span>
                    <div className="flex justify-between py-0.5">
                      <span>OLAP Execution:</span>
                      <span className="text-slate-200 font-semibold">DuckDB In-Process</span>
                    </div>
                    <div className="flex justify-between py-0.5">
                      <span>Columnar Parser:</span>
                      <span className="text-slate-200 font-semibold">Polars v1.x (SIMD)</span>
                    </div>
                    <div className="flex justify-between py-0.5">
                      <span>Memory Sandboxing:</span>
                      <span className="text-emerald-400 font-semibold">0-Disk Ephemeral</span>
                    </div>
                    <div className="flex justify-between py-0.5">
                      <span>Report Engine:</span>
                      <span className="text-slate-200 font-semibold">WeasyPrint PDF</span>
                    </div>
                  </div>
                </div>

              </div>
            </div>

            {/* Right Main Content Stream (8 Cols) */}
            <div className="lg:col-span-8 space-y-16">
              
              {/* SECTION 1: Supported File Formats */}
              <section id="supported-formats" className="space-y-5 scroll-mt-28">
                <div className="flex items-center gap-3 pb-3 border-b border-white/10">
                  <div className="h-10 w-10 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 flex items-center justify-center shrink-0">
                    <FileSpreadsheet className="h-5 w-5" />
                  </div>
                  <div>
                    <h2 className="text-xl sm:text-2xl font-display font-bold text-white tracking-tight">
                      1. Ingestion Engine &amp; Supported Formats
                    </h2>
                    <p className="text-xs sm:text-sm text-slate-400 font-sans">
                      Hardware-accelerated binary streaming with automatic dialect resolution.
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {[
                    {
                      format: "CSV & TSV Streaming",
                      engine: "Polars ChunkedArray",
                      desc: "Samples up to 10,000 rows to auto-detect delimiters (, ; \\t |), handling Latin-1, CP1252, and UTF-8 with zero BOM panics.",
                      throughput: "Up to 1.2 GB/sec",
                      memory: "12MB vs 160MB in Pandas"
                    },
                    {
                      format: "Excel (.xlsx, .xls)",
                      engine: "calamine Rust Parser",
                      desc: "Bypasses standard Python XML DOM overhead via native Rust calamine routines, enforcing zip-bomb decompression protection.",
                      throughput: "14x faster than openpyxl",
                      memory: "Streaming worksheet reader"
                    },
                    {
                      format: "Apache Parquet",
                      engine: "Native Zero-Copy Arrow",
                      desc: "Direct memory-mapped projection pushdown. Reads only requested schema columns directly into in-memory buffers.",
                      throughput: "Sub-5ms columnar scans",
                      memory: "Exact zero-copy pointer pass"
                    },
                    {
                      format: "JSON Lines (.jsonl)",
                      engine: "Schema Unification",
                      desc: "Streamed row-by-row ndjson parsing with automated schema leveling, nested key extraction, and null coalescing.",
                      throughput: "High-density event logs",
                      memory: "Streaming backpressure buffer"
                    }
                  ].map((item) => (
                    <div 
                      key={item.format}
                      className="rounded-2xl p-1 bg-white/[0.03] border border-white/10 hover:border-white/20 transition-all duration-200"
                    >
                      <div className="rounded-[calc(1rem-2px)] bg-[#0c0d12] p-4 sm:p-5 space-y-3 h-full flex flex-col justify-between">
                        <div className="space-y-1.5">
                          <div className="flex items-center justify-between gap-2">
                            <span className="text-sm font-display font-bold text-white">{item.format}</span>
                            <Badge variant="outline" className="text-[10px] font-mono border-white/10 text-emerald-400 bg-emerald-500/5">
                              {item.engine}
                            </Badge>
                          </div>
                          <p className="text-xs text-slate-400 leading-relaxed font-sans">
                            {item.desc}
                          </p>
                        </div>

                        <div className="pt-2 border-t border-white/5 grid grid-cols-2 gap-2 text-[11px] font-mono text-slate-400">
                          <div>
                            <span className="block text-[10px] text-slate-500 uppercase">Throughput</span>
                            <span className="text-slate-200">{item.throughput}</span>
                          </div>
                          <div>
                            <span className="block text-[10px] text-slate-500 uppercase">Footprint</span>
                            <span className="text-emerald-400">{item.memory}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Example Ingestion Code Snippet */}
                <div className="rounded-2xl p-1 bg-white/[0.03] border border-white/10">
                  <div className="rounded-[calc(1rem-2px)] bg-[#0c0d12] p-4 sm:p-5 space-y-3 font-mono text-xs">
                    <div className="flex items-center justify-between text-slate-400 pb-2 border-b border-white/10">
                      <span className="font-bold text-slate-200 flex items-center gap-2">
                        <Terminal className="h-3.5 w-3.5 text-emerald-400" />
                        <span>Polars Streaming Buffer Implementation</span>
                      </span>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleCopy(`import polars as pl

# Zero-copy memory-mapped ingestion with robust dialect detection
df = pl.read_csv(
    file_buffer,
    infer_schema_length=10000,
    try_parse_dates=True,
    truncate_ragged_lines=False,
    ignore_errors=False
)
print(f"Ingested {df.height} rows across {df.width} columns in 18ms")`, "snippet_ingest", "Code")}
                        className="h-7 px-2 font-mono text-[11px] gap-1.5 text-slate-400 hover:text-white"
                      >
                        {copiedKey === "snippet_ingest" ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                        <span>{copiedKey === "snippet_ingest" ? "Copied" : "Copy"}</span>
                      </Button>
                    </div>
                    <pre className="p-3 bg-black/60 rounded-xl border border-white/5 text-slate-300 text-[11px] overflow-x-auto leading-relaxed">
                      <code>{`import polars as pl

# Zero-copy memory-mapped ingestion with robust dialect detection
df = pl.read_csv(
    file_buffer,
    infer_schema_length=10000,
    try_parse_dates=True,
    truncate_ragged_lines=False,
    ignore_errors=False
)
print(f"Ingested {df.height} rows across {df.width} columns in 18ms")`}</code>
                    </pre>
                  </div>
                </div>
              </section>

              {/* SECTION 2: Confidence Scoring Algorithm */}
              <section id="confidence-scoring" className="space-y-5 scroll-mt-28">
                <div className="flex items-center gap-3 pb-3 border-b border-white/10">
                  <div className="h-10 w-10 rounded-xl bg-purple-500/10 border border-purple-500/20 text-purple-400 flex items-center justify-center shrink-0">
                    <Gauge className="h-5 w-5" />
                  </div>
                  <div>
                    <h2 className="text-xl sm:text-2xl font-display font-bold text-white tracking-tight">
                      2. Mathematical Confidence Scoring Engine
                    </h2>
                    <p className="text-xs sm:text-sm text-slate-400 font-sans">
                      Formulation across 4 orthogonal data quality dimensions with letter-grade calibration.
                    </p>
                  </div>
                </div>

                {/* Mathematical Formula Card */}
                <div className="rounded-2xl p-1 bg-white/[0.03] border border-white/10">
                  <div className="rounded-[calc(1rem-2px)] bg-[#0c0d12] p-5 sm:p-6 space-y-4">
                    <div className="p-3.5 bg-black/60 rounded-xl border border-white/10 font-mono text-xs sm:text-sm text-emerald-400 shadow-inner">
                      <code>Score = (Completeness × 0.35) + (Consistency × 0.25) + (Validity × 0.25) + (Stability × 0.15)</code>
                    </div>

                    <p className="text-xs text-slate-400 font-sans leading-relaxed">
                      Every ingested column is assigned a score from 0.0 to 100.0 based on rigorous statistical heuristics. Weights strictly sum to 1.0, penalizing masking sentinels, format drift, and outlier dispersion.
                    </p>

                    {/* 4 Pillars Detailed Grid */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 pt-2">
                      {[
                        {
                          title: "1. Completeness (Weight: 35%)",
                          color: "emerald",
                          desc: "Penalizes null occurrences, whitespace-only cells, and masked sentinel NaN strings (e.g., 'N/A', 'null', '-999', 'UNKNOWN', 'None').",
                          metric: "Null Rate & Masked Sentinels"
                        },
                        {
                          title: "2. Consistency (Weight: 25%)",
                          color: "blue",
                          desc: "Measures data type homogeneity, datetime format alignment (ISO-8601 vs mixed formats), and casing variances across text fields.",
                          metric: "Type Purity & Format Variance"
                        },
                        {
                          title: "3. Validity (Weight: 25%)",
                          color: "amber",
                          desc: "Checks values against strict business rules, postal codes, positive currency constraints, email syntax, and Benford's Law distribution.",
                          metric: "Boundary & Rule Adherence"
                        },
                        {
                          title: "4. Stability (Weight: 15%)",
                          color: "purple",
                          desc: "Scans for zero-variance constant variables, kurtosis dispersion tails, Population Stability Index (PSI), and segment drift.",
                          metric: "Dispersion & Drift Index"
                        }
                      ].map((pillar) => (
                        <div key={pillar.title} className="p-3.5 rounded-xl bg-white/[0.02] border border-white/5 space-y-1.5">
                          <span className="font-mono text-xs font-bold text-white block">
                            {pillar.title}
                          </span>
                          <p className="text-[11px] text-slate-400 font-sans leading-relaxed">
                            {pillar.desc}
                          </p>
                          <div className="pt-1">
                            <span className="text-[10px] font-mono text-slate-500 uppercase">Audit Target: </span>
                            <span className="text-[10px] font-mono text-slate-300">{pillar.metric}</span>
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* Grade Threshold Calibration */}
                    <div className="pt-3 border-t border-white/10">
                      <span className="text-xs font-mono font-bold text-slate-300 block mb-2 uppercase tracking-wider">
                        Confidence Letter Grade Scale
                      </span>
                      <div className="grid grid-cols-5 gap-2 text-center font-mono text-xs">
                        <div className="p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
                          <span className="font-bold text-sm block">A</span>
                          <span className="text-[10px] text-slate-400">90 - 100%</span>
                        </div>
                        <div className="p-2 rounded-lg bg-blue-500/10 border border-blue-500/20 text-blue-400">
                          <span className="font-bold text-sm block">B</span>
                          <span className="text-[10px] text-slate-400">80 - 89%</span>
                        </div>
                        <div className="p-2 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-400">
                          <span className="font-bold text-sm block">C</span>
                          <span className="text-[10px] text-slate-400">70 - 79%</span>
                        </div>
                        <div className="p-2 rounded-lg bg-orange-500/10 border border-orange-500/20 text-orange-400">
                          <span className="font-bold text-sm block">D</span>
                          <span className="text-[10px] text-slate-400">60 - 69%</span>
                        </div>
                        <div className="p-2 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-400">
                          <span className="font-bold text-sm block">F</span>
                          <span className="text-[10px] text-slate-400">&lt; 60%</span>
                        </div>
                      </div>
                    </div>

                  </div>
                </div>
              </section>

              {/* SECTION 3: Transformation DAG & Issue Ledger */}
              <section id="issue-dag" className="space-y-5 scroll-mt-28">
                <div className="flex items-center gap-3 pb-3 border-b border-white/10">
                  <div className="h-10 w-10 rounded-xl bg-blue-500/10 border border-blue-500/20 text-blue-400 flex items-center justify-center shrink-0">
                    <Layers className="h-5 w-5" />
                  </div>
                  <div>
                    <h2 className="text-xl sm:text-2xl font-display font-bold text-white tracking-tight">
                      3. Issue Ledger &amp; Remediation Transformation DAG
                    </h2>
                    <p className="text-xs sm:text-sm text-slate-400 font-sans">
                      Automated audit with human-in-the-loop decision locking across 9 structural categories.
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
                  {[
                    { name: "Exact & PK Duplicates", fix: "Polars unique() deduplication", code: "df.unique(subset=['id'])" },
                    { name: "Currency Strings", fix: "Strip symbols & cast to Float64", code: "pl.col('amt').str.replace(r'[$€,]', '')" },
                    { name: "Extreme Outliers", fix: "Winsorization at 1st/99th percentiles", code: "pl.col('val').clip(p01, p99)" },
                    { name: "Contextual Missing", fix: "Domain-aware median/mode imputation", code: "pl.col('x').fill_null(strategy='median')" },
                    { name: "Constant Variables", fix: "Zero-variance feature isolation", code: "drop_columns([zero_variance_cols])" },
                    { name: "Mixed Date Encodings", fix: "Chronological ISO-8601 normalization", code: "pl.col('dt').str.to_datetime()" },
                    { name: "Masked Sentinels", fix: "Replace '-999', 'N/A' with true nulls", code: "pl.when().then(None).otherwise()" },
                    { name: "Whitespace Padding", fix: "Trim leading/trailing Unicode spaces", code: "pl.col('txt').str.strip_chars()" },
                    { name: "Inconsistent Casing", fix: "Standardize titlecase or lowercase", code: "pl.col('cat').str.to_lowercase()" }
                  ].map((issue) => (
                    <div key={issue.name} className="p-3.5 rounded-xl bg-white/[0.02] border border-white/5 space-y-1.5 hover:border-white/15 transition-all">
                      <div className="flex items-center gap-1.5">
                        <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                        <span className="font-bold text-white text-xs font-sans">{issue.name}</span>
                      </div>
                      <p className="text-[11px] text-slate-400 font-sans leading-tight">{issue.fix}</p>
                      <code className="text-[10px] font-mono text-slate-500 block truncate pt-1">{issue.code}</code>
                    </div>
                  ))}
                </div>
              </section>

              {/* SECTION 4: DuckDB In-Process SQL Lab & BI Studio */}
              <section id="duckdb-sql" className="space-y-5 scroll-mt-28">
                <div className="flex items-center gap-3 pb-3 border-b border-white/10">
                  <div className="h-10 w-10 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-400 flex items-center justify-center shrink-0">
                    <Database className="h-5 w-5" />
                  </div>
                  <div>
                    <h2 className="text-xl sm:text-2xl font-display font-bold text-white tracking-tight">
                      4. In-Process DuckDB SQL Lab &amp; BI Studio
                    </h2>
                    <p className="text-xs sm:text-sm text-slate-400 font-sans">
                      Sub-millisecond analytical query execution with zero container disk writes.
                    </p>
                  </div>
                </div>

                <div className="rounded-2xl p-1 bg-white/[0.03] border border-white/10">
                  <div className="rounded-[calc(1rem-2px)] bg-[#0c0d12] p-5 sm:p-6 space-y-4 text-xs font-mono">
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <span className="font-bold text-white font-sans text-sm block">
                          Engine Invariants &amp; Security
                        </span>
                        <ul className="space-y-1.5 text-slate-400 font-sans text-xs">
                          <li className="flex items-start gap-2">
                            <Check className="h-3.5 w-3.5 text-emerald-400 shrink-0 mt-0.5" />
                            <span><strong>Zero-Copy Parquet:</strong> Uses Apache Arrow memory tables to query datasets without serializing to disk.</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <Check className="h-3.5 w-3.5 text-emerald-400 shrink-0 mt-0.5" />
                            <span><strong>C++ Engine Sandbox:</strong> Sets <code>enable_external_access = false</code> to block filesystem access.</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <Check className="h-3.5 w-3.5 text-emerald-400 shrink-0 mt-0.5" />
                            <span><strong>Strict Read-Only:</strong> Blocks DDL, DML, statement chaining, and dangerous table functions.</span>
                          </li>
                        </ul>
                      </div>

                      <div className="space-y-2">
                        <span className="font-bold text-white font-sans text-sm block">
                          Jinja Parameter Binding
                        </span>
                        <p className="text-slate-400 font-sans text-xs leading-relaxed">
                          Safe mustache parameter replacement (<code>{'{{ param_name }}'}</code>) transforms template placeholders into positional SQL parameters to prevent SQL injection.
                        </p>
                        <div className="p-2.5 bg-black/60 rounded-xl border border-white/5 text-[11px] text-slate-300">
                          <code>SELECT * FROM dataset WHERE department = &#123;&#123; dept &#125;&#125;</code>
                        </div>
                      </div>
                    </div>

                  </div>
                </div>
              </section>

              {/* SECTION 5: REST API Endpoint Specifications */}
              <section id="api-endpoints" className="space-y-5 scroll-mt-28">
                <div className="flex items-center justify-between pb-3 border-b border-white/10">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-xl bg-purple-500/10 border border-purple-500/20 text-purple-400 flex items-center justify-center shrink-0">
                      <Code2 className="h-5 w-5" />
                    </div>
                    <div>
                      <h2 className="text-xl sm:text-2xl font-display font-bold text-white tracking-tight">
                        5. Headless REST API Endpoint Specifications
                      </h2>
                      <p className="text-xs sm:text-sm text-slate-400 font-sans">
                        Production JSON &amp; SSE endpoints for programmatic auditing and CI/CD pipelines.
                      </p>
                    </div>
                  </div>
                </div>

                {/* Category Filter Pills */}
                <div className="flex items-center gap-2 overflow-x-auto scrollbar-none py-1">
                  {[
                    { id: "all", label: "All Endpoints" },
                    { id: "ingestion", label: "Ingestion" },
                    { id: "audit", label: "Report Audit" },
                    { id: "duckdb", label: "DuckDB SQL" },
                    { id: "analytics", label: "BI Explore" },
                    { id: "companion", label: "RAG Stream" },
                    { id: "export", label: "Exports" }
                  ].map((cat) => (
                    <button
                      key={cat.id}
                      onClick={() => setSelectedCategory(cat.id)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all cursor-pointer whitespace-nowrap ${
                        selectedCategory === cat.id
                          ? "bg-white text-black font-bold shadow-xs"
                          : "bg-white/[0.04] text-slate-400 hover:text-white hover:bg-white/[0.08] border border-white/5"
                      }`}
                    >
                      {cat.label}
                    </button>
                  ))}
                </div>

                {/* Endpoints List */}
                <div className="space-y-6">
                  {filteredEndpoints.map((ep) => (
                    <div
                      key={ep.id}
                      className="rounded-2xl p-1 bg-white/[0.03] border border-white/10 hover:border-white/20 transition-all duration-200 shadow-premium"
                    >
                      <div className="rounded-[calc(1rem-2px)] bg-[#0c0d12] overflow-hidden">
                        
                        {/* Endpoint Header Bar */}
                        <div className="p-4 sm:p-5 border-b border-white/10 flex flex-wrap items-center justify-between gap-3 bg-white/[0.02]">
                          <div className="flex items-center gap-3">
                            <Badge 
                              className={`font-mono text-xs font-bold px-2.5 py-0.5 rounded-md ${
                                ep.method === "POST"
                                  ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                                  : ep.method === "GET"
                                  ? "bg-blue-500/20 text-blue-400 border border-blue-500/30"
                                  : "bg-rose-500/20 text-rose-400 border border-rose-500/30"
                              }`}
                            >
                              {ep.method}
                            </Badge>
                            <span className="font-mono text-xs sm:text-sm font-bold text-white">
                              {ep.path}
                            </span>
                          </div>

                          <span className="text-[11px] font-mono text-slate-500 uppercase tracking-wider">
                            {ep.category}
                          </span>
                        </div>

                        {/* Endpoint Content Body */}
                        <div className="p-4 sm:p-6 space-y-4">
                          <div>
                            <h3 className="text-sm font-display font-bold text-white mb-1">
                              {ep.title}
                            </h3>
                            <p className="text-xs text-slate-400 font-sans leading-relaxed">
                              {ep.description}
                            </p>
                          </div>

                          {/* Interactive Language & Snippet Tabs */}
                          <Tabs defaultValue="curl" className="w-full">
                            <div className="flex items-center justify-between pb-2">
                              <TabsList className="bg-black/40 border border-white/10 p-0.5 h-8 rounded-lg">
                                <TabsTrigger value="curl" className="text-[11px] font-mono h-7 px-2.5 data-[state=active]:bg-white/10 data-[state=active]:text-white">
                                  cURL
                                </TabsTrigger>
                                <TabsTrigger value="python" className="text-[11px] font-mono h-7 px-2.5 data-[state=active]:bg-white/10 data-[state=active]:text-white">
                                  Python
                                </TabsTrigger>
                                <TabsTrigger value="ts" className="text-[11px] font-mono h-7 px-2.5 data-[state=active]:bg-white/10 data-[state=active]:text-white">
                                  TypeScript
                                </TabsTrigger>
                                <TabsTrigger value="response" className="text-[11px] font-mono h-7 px-2.5 data-[state=active]:bg-white/10 data-[state=active]:text-white">
                                  JSON Schema
                                </TabsTrigger>
                              </TabsList>
                            </div>

                            <TabsContent value="curl" className="m-0 relative">
                              <div className="absolute right-3 top-3 z-10">
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => handleCopy(ep.curlExample, `${ep.id}_curl`, "cURL Command")}
                                  className="h-7 px-2 font-mono text-[10px] gap-1 text-slate-400 hover:text-white bg-white/5 hover:bg-white/10 rounded"
                                >
                                  {copiedKey === `${ep.id}_curl` ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
                                  <span>{copiedKey === `${ep.id}_curl` ? "Copied" : "Copy"}</span>
                                </Button>
                              </div>
                              <pre className="p-4 bg-black/80 rounded-xl border border-white/5 text-[11px] font-mono text-slate-300 overflow-x-auto leading-relaxed">
                                <code>{ep.curlExample}</code>
                              </pre>
                            </TabsContent>

                            <TabsContent value="python" className="m-0 relative">
                              <div className="absolute right-3 top-3 z-10">
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => handleCopy(ep.pythonExample, `${ep.id}_py`, "Python Snippet")}
                                  className="h-7 px-2 font-mono text-[10px] gap-1 text-slate-400 hover:text-white bg-white/5 hover:bg-white/10 rounded"
                                >
                                  {copiedKey === `${ep.id}_py` ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
                                  <span>{copiedKey === `${ep.id}_py` ? "Copied" : "Copy"}</span>
                                </Button>
                              </div>
                              <pre className="p-4 bg-black/80 rounded-xl border border-white/5 text-[11px] font-mono text-emerald-300 overflow-x-auto leading-relaxed">
                                <code>{ep.pythonExample}</code>
                              </pre>
                            </TabsContent>

                            <TabsContent value="ts" className="m-0 relative">
                              <div className="absolute right-3 top-3 z-10">
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => handleCopy(ep.tsExample, `${ep.id}_ts`, "TypeScript Snippet")}
                                  className="h-7 px-2 font-mono text-[10px] gap-1 text-slate-400 hover:text-white bg-white/5 hover:bg-white/10 rounded"
                                >
                                  {copiedKey === `${ep.id}_ts` ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
                                  <span>{copiedKey === `${ep.id}_ts` ? "Copied" : "Copy"}</span>
                                </Button>
                              </div>
                              <pre className="p-4 bg-black/80 rounded-xl border border-white/5 text-[11px] font-mono text-blue-300 overflow-x-auto leading-relaxed">
                                <code>{ep.tsExample}</code>
                              </pre>
                            </TabsContent>

                            <TabsContent value="response" className="m-0 relative">
                              <div className="absolute right-3 top-3 z-10">
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => handleCopy(ep.responseBody, `${ep.id}_resp`, "Response JSON")}
                                  className="h-7 px-2 font-mono text-[10px] gap-1 text-slate-400 hover:text-white bg-white/5 hover:bg-white/10 rounded"
                                >
                                  {copiedKey === `${ep.id}_resp` ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
                                  <span>{copiedKey === `${ep.id}_resp` ? "Copied" : "Copy"}</span>
                                </Button>
                              </div>
                              <pre className="p-4 bg-black/80 rounded-xl border border-white/5 text-[11px] font-mono text-amber-300 overflow-x-auto leading-relaxed">
                                <code>{ep.responseBody}</code>
                              </pre>
                            </TabsContent>
                          </Tabs>

                        </div>

                      </div>
                    </div>
                  ))}
                </div>
              </section>

              {/* SECTION 6: Architecture Specs & Security Invariants */}
              <section id="security" className="space-y-5 scroll-mt-28">
                <div className="flex items-center gap-3 pb-3 border-b border-white/10">
                  <div className="h-10 w-10 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 flex items-center justify-center shrink-0">
                    <ShieldCheck className="h-5 w-5" />
                  </div>
                  <div>
                    <h2 className="text-xl sm:text-2xl font-display font-bold text-white tracking-tight">
                      6. Architecture Invariants &amp; Security Compliance
                    </h2>
                    <p className="text-xs sm:text-sm text-slate-400 font-sans">
                      Zero-retention ephemeral execution, AST dunder isolation, and privacy boundaries.
                    </p>
                  </div>
                </div>

                <div className="rounded-2xl p-1 bg-white/[0.03] border border-white/10">
                  <div className="rounded-[calc(1rem-2px)] bg-[#0c0d12] overflow-hidden divide-y divide-white/5 text-xs font-mono">
                    {[
                      { param: "Storage Architecture", detail: "100% In-Memory RAM Buffers. Zero bytes persisted to permanent disk drives." },
                      { param: "Session Lifetime", detail: "Automatic garbage collection and memory un-registration after 60 minutes of inactivity." },
                      { param: "C++ Engine Isolation", detail: "DuckDB configured with SET enable_external_access = false blocking all socket and local file IO." },
                      { param: "Python AST Sandboxing", detail: "Enforces strict AST parsing blocking all dunder attributes (__class__, __subclasses__, __globals__)." },
                      { param: "Transit Encryption", detail: "TLS 1.3 enforced with strict HSTS (Strict-Transport-Security: max-age=63072000)." },
                      { param: "Content Security Policy", detail: "Strict CSP with script-src 'self' and frame-ancestors 'none' preventing iframe clickjacking." },
                      { param: "Rate Limiting", detail: "Token-bucket sliding window: 60 uploads/minute per source IP address with 429 throttling." },
                      { param: "LLM Privacy Boundary", detail: "Raw data records strictly excluded from external prompts. Only statistical schemas are analyzed." }
                    ].map((row) => (
                      <div key={row.param} className="p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-1.5 hover:bg-white/[0.02] transition-colors">
                        <span className="font-bold text-white font-sans text-xs">{row.param}:</span>
                        <span className="text-slate-400 text-[11px] sm:text-right max-w-md">{row.detail}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </section>

            </div>

          </div>
        </div>

        {/* Bottom CTA Banner */}
        <div className="mt-20">
          <CtaBanner />
        </div>

      </main>

      <Footer />
    </div>
  );
};

export default Documentation;
