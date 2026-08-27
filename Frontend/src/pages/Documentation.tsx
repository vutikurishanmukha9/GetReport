import { useState } from "react";
import { Link } from "react-router-dom";
import { 
  FileSpreadsheet, Gauge, ShieldCheck, Code2, Copy, 
  Check, ArrowRight, BookOpen, Layers
} from "lucide-react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";
import { useToast } from "@/hooks/use-toast";

interface EndpointSpec {
  method: "POST" | "GET";
  path: string;
  title: string;
  description: string;
  requestBody?: string;
  responseBody: string;
}

const apiEndpoints: EndpointSpec[] = [
  {
    method: "POST",
    path: "/api/upload",
    title: "Ingest Tabular File",
    description: "Stream a CSV, Excel, Parquet, or JSONL file into ephemeral memory. Returns inferred schema, preview rows, and task_id.",
    requestBody: `// Multipart Form Data
file: File (e.g. transactions_q1.csv)`,
    responseBody: `{
  "task_id": "job_9481ad2",
  "filename": "transactions_q1.csv",
  "info": {
    "rows": 12500,
    "columns": ["id", "amount", "created_at", "status"],
    "dtypes": { "id": "Int64", "amount": "Float64" }
  },
  "cleaning_report": { "duplicate_rows_removed": 14 }
}`
  },
  {
    method: "POST",
    path: "/api/process",
    title: "Execute Issue Ledger Remediation",
    description: "Submit approved transformation actions. Computes confidence scores, bivariate correlations, and initiates PDF compilation.",
    requestBody: `{
  "task_id": "job_9481ad2",
  "decisions": {
    "issue_101": "approve",
    "issue_102": "reject"
  }
}`,
    responseBody: `{
  "status": "completed",
  "dataset_confidence_score": 94.2,
  "download_url": "/api/download/pdf/job_9481ad2",
  "columns_audited": 4
}`
  },
  {
    method: "POST",
    path: "/api/rag/chat",
    title: "Query Ephemeral RAG Companion",
    description: "Ask natural language questions regarding dataset health, correlation insights, or anomaly causes.",
    requestBody: `{
  "task_id": "job_9481ad2",
  "query": "Which columns have extreme skewness or missing values?"
}`,
    responseBody: `{
  "response": "The 'referral_code' column exhibits 81.5% missing values. The 'amount' variable has positive skewness (gamma = 2.4).",
  "sources": [
    { "type": "column_confidence", "name": "referral_code", "score": "18.5%" }
  ]
}`
  }
];

export const Documentation = () => {
  const { toast } = useToast();
  const [copiedPath, setCopiedPath] = useState<string | null>(null);

  const handleCopy = (text: string, path: string) => {
    navigator.clipboard.writeText(text);
    setCopiedPath(path);
    toast({ title: "Copied to Clipboard", description: `Endpoint ${path} schema copied.` });
    setTimeout(() => setCopiedPath(null), 2000);
  };

  return (
    <div className="min-h-screen flex flex-col bg-background animate-in fade-in duration-500">
      <Header onReset={() => {}} showReset={false} />

      <main className="flex-1 pt-16 sm:pt-20">
        {/* Header */}
        <div className="border-b border-border/60 bg-gradient-to-b from-muted/20 via-background to-background py-8 sm:py-12">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-5xl text-center space-y-3 sm:space-y-4">
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-primary/10 text-primary text-xs font-semibold uppercase tracking-wider font-mono border border-primary/20 t-badge-shimmer">
              <BookOpen className="h-3.5 w-3.5" />
              <span>Technical Documentation & Architecture</span>
            </div>

            <h1 className="text-3xl sm:text-4xl md:text-5xl font-display font-extrabold text-foreground tracking-tight uppercase leading-[1.08]">
              GetReport Technical Hub.
            </h1>
            
            <p className="text-sm sm:text-base text-muted-foreground max-w-2xl mx-auto leading-relaxed font-sans">
              Comprehensive specifications for Polars zero-copy ingestion, A-F column confidence algorithms, WeasyPrint compilation, and headless REST endpoints.
            </p>
          </div>
        </div>

        {/* Section: Main Content Grid with Left Nav */}
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-10 max-w-7xl">
          {/* Mobile Quick-Jump Nav Bar (< lg) */}
          <div className="lg:hidden sticky top-20 z-20 bg-background/95 backdrop-blur-md border-b border-border/60 py-2.5 -mx-4 px-4 sm:-mx-6 sm:px-6 mb-5">
            <div className="flex items-center gap-2 overflow-x-auto scrollbar-none">
              <a href="#supported-formats" className="px-3 py-1.5 rounded-xl bg-muted/40 text-foreground text-xs font-mono whitespace-nowrap border border-border shrink-0 hover:bg-muted">
                1. Formats
              </a>
              <a href="#confidence-scores" className="px-3 py-1.5 rounded-xl bg-muted/40 text-foreground text-xs font-mono whitespace-nowrap border border-border shrink-0 hover:bg-muted">
                2. Confidence
              </a>
              <a href="#issue-ledger" className="px-3 py-1.5 rounded-xl bg-muted/40 text-foreground text-xs font-mono whitespace-nowrap border border-border shrink-0 hover:bg-muted">
                3. Ledger DAG
              </a>
              <a href="#api-endpoints" className="px-3 py-1.5 rounded-xl bg-muted/40 text-foreground text-xs font-mono whitespace-nowrap border border-border shrink-0 hover:bg-muted">
                4. REST API
              </a>
              <a href="#security" className="px-3 py-1.5 rounded-xl bg-muted/40 text-foreground text-xs font-mono whitespace-nowrap border border-border shrink-0 hover:bg-muted">
                5. Security
              </a>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8">
            
            {/* Left Sticky Navigation Column (Desktop lg:block) */}
            <div className="hidden lg:block lg:col-span-4">
              <div className="sticky top-24 space-y-3.5">
                <Card className="border border-border bg-card rounded-2xl p-5 shadow-premium space-y-3">
                  <span className="font-mono text-xs uppercase tracking-wider font-bold text-muted-foreground block border-b border-border/60 pb-2">
                    Documentation Index
                  </span>
                  <nav className="space-y-1 text-xs font-mono">
                    <a 
                      href="#supported-formats" 
                      className="flex items-center gap-2 px-3 py-2 rounded-xl text-muted-foreground hover:text-foreground hover:bg-muted/40 transition-colors"
                    >
                      <FileSpreadsheet className="h-4 w-4 text-emerald-600 shrink-0" />
                      <span>1. Supported File Formats</span>
                    </a>
                    <a 
                      href="#confidence-scores" 
                      className="flex items-center gap-2 px-3 py-2 rounded-xl text-muted-foreground hover:text-foreground hover:bg-muted/40 transition-colors"
                    >
                      <Gauge className="h-4 w-4 text-primary shrink-0" />
                      <span>2. Confidence Scoring Algorithm</span>
                    </a>
                    <a 
                      href="#issue-ledger" 
                      className="flex items-center gap-2 px-3 py-2 rounded-xl text-muted-foreground hover:text-foreground hover:bg-muted/40 transition-colors"
                    >
                      <Layers className="h-4 w-4 text-blue-600 shrink-0" />
                      <span>3. Issue Ledger DAG Categories</span>
                    </a>
                    <a 
                      href="#api-endpoints" 
                      className="flex items-center gap-2 px-3 py-2 rounded-xl text-muted-foreground hover:text-foreground hover:bg-muted/40 transition-colors"
                    >
                      <Code2 className="h-4 w-4 text-purple-600 shrink-0" />
                      <span>4. REST API Endpoint Specs</span>
                    </a>
                    <a 
                      href="#security" 
                      className="flex items-center gap-2 px-3 py-2 rounded-xl text-muted-foreground hover:text-foreground hover:bg-muted/40 transition-colors"
                    >
                      <ShieldCheck className="h-4 w-4 text-rose-600 shrink-0" />
                      <span>5. Memory Security & Lifetimes</span>
                    </a>
                  </nav>
                </Card>

                {/* Quick GitHub Badge Card */}
                <Card className="border border-border bg-muted/20 rounded-2xl p-4 text-xs font-mono space-y-1.5">
                  <div className="flex items-center justify-between text-muted-foreground">
                    <span>Engine Release:</span>
                    <span className="font-bold text-foreground">v2.4.0 (Polars Rust)</span>
                  </div>
                  <div className="flex items-center justify-between text-muted-foreground">
                    <span>License:</span>
                    <span className="font-bold text-emerald-600">Apache 2.0 Open Source</span>
                  </div>
                </Card>
              </div>
            </div>

            {/* Right Documentation Sections (Col 5-12) */}
            <div className="lg:col-span-8 space-y-10 sm:space-y-12">
              
              {/* Section 1: Supported File Formats */}
              <section id="supported-formats" className="space-y-4 scroll-mt-28">
                <div className="flex items-center gap-3 border-b border-border/80 pb-2.5">
                  <div className="p-2 bg-emerald-500/10 text-emerald-700 rounded-xl">
                    <FileSpreadsheet className="h-5 w-5" />
                  </div>
                  <div>
                    <h2 className="text-xl sm:text-2xl font-display font-bold text-foreground uppercase tracking-tight">
                      1. Supported Ingestion Formats
                    </h2>
                    <p className="text-xs text-muted-foreground font-sans">
                      Zero-copy memory streaming specifications across tabular formats.
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 font-mono text-xs">
                  <Card className="p-4 sm:p-5 border border-border bg-card rounded-2xl space-y-1.5 shadow-xs t-card-lift">
                    <div className="flex justify-between items-center">
                      <span className="font-bold text-foreground text-sm font-sans">CSV & TSV</span>
                      <Badge variant="secondary" className="text-[10px]">Polars Stream</Badge>
                    </div>
                    <p className="text-[11px] text-muted-foreground font-sans leading-relaxed">
                      Auto-detects delimiters (<code>,</code>, <code>;</code>, <code>\t</code>, <code>|</code>) and decodes UTF-8, Latin-1, CP1252 with no BOM errors.
                    </p>
                  </Card>

                  <Card className="p-4 sm:p-5 border border-border bg-card rounded-2xl space-y-1.5 shadow-xs t-card-lift">
                    <div className="flex justify-between items-center">
                      <span className="font-bold text-foreground text-sm font-sans">Excel (.xlsx, .xls)</span>
                      <Badge variant="secondary" className="text-[10px]">calamine Rust</Badge>
                    </div>
                    <p className="text-[11px] text-muted-foreground font-sans leading-relaxed">
                      Extracts active worksheet cells with 10x faster parsing than openpyxl while enforcing decompression zip-bomb limits.
                    </p>
                  </Card>

                  <Card className="p-4 sm:p-5 border border-border bg-card rounded-2xl space-y-1.5 shadow-xs t-card-lift">
                    <div className="flex justify-between items-center">
                      <span className="font-bold text-foreground text-sm font-sans">Apache Parquet</span>
                      <Badge variant="secondary" className="text-[10px]">Native Arrow</Badge>
                    </div>
                    <p className="text-[11px] text-muted-foreground font-sans leading-relaxed">
                      Sub-millisecond columnar projection pushdown reading only requested schema fields directly into RAM.
                    </p>
                  </Card>

                  <Card className="p-4 sm:p-5 border border-border bg-card rounded-2xl space-y-1.5 shadow-xs t-card-lift">
                    <div className="flex justify-between items-center">
                      <span className="font-bold text-foreground text-sm font-sans">JSON Lines (.jsonl)</span>
                      <Badge variant="secondary" className="text-[10px]">Schema Flatten</Badge>
                    </div>
                    <p className="text-[11px] text-muted-foreground font-sans leading-relaxed">
                      Iterative ndjson streaming with automatic schema unification and nested object flattening.
                    </p>
                  </Card>
                </div>
              </section>

              {/* Section 2: Confidence Scoring */}
              <section id="confidence-scores" className="space-y-4 scroll-mt-28">
                <div className="flex items-center gap-3 border-b border-border/80 pb-2.5">
                  <div className="p-2 bg-primary/10 text-primary rounded-xl">
                    <Gauge className="h-5 w-5" />
                  </div>
                  <div>
                    <h2 className="text-xl sm:text-2xl font-display font-bold text-foreground uppercase tracking-tight">
                      2. Column Confidence Scoring
                    </h2>
                    <p className="text-xs text-muted-foreground font-sans">
                      Mathematical weighting formulation across 4 quality dimensions.
                    </p>
                  </div>
                </div>

                <Card className="border border-border bg-card rounded-2xl p-5 sm:p-6 shadow-premium space-y-3.5 font-mono text-xs">
                  <div className="p-3 bg-muted/40 rounded-xl border border-border/60 text-emerald-800 text-xs">
                    <code>Score = (Completeness × 0.35) + (Consistency × 0.25) + (Validity × 0.25) + (Stability × 0.15)</code>
                  </div>

                  <div className="space-y-2.5 font-sans text-xs text-muted-foreground leading-relaxed">
                    <div>
                      <strong className="text-foreground font-mono block">1. Completeness (Weight: 35%)</strong>
                      <span>Penalizes null rates, empty whitespace strings, and masked NaN values (e.g. &quot;N/A&quot;, &quot;null&quot;, &quot;-999&quot;).</span>
                    </div>
                    <div>
                      <strong className="text-foreground font-mono block">2. Consistency (Weight: 25%)</strong>
                      <span>Measures data type homogeneity, datetime format uniformity, and string case variance.</span>
                    </div>
                    <div>
                      <strong className="text-foreground font-mono block">3. Validity (Weight: 25%)</strong>
                      <span>Validates domain boundaries (e.g. positive prices, standard postal codes, valid emails).</span>
                    </div>
                    <div>
                      <strong className="text-foreground font-mono block">4. Stability (Weight: 15%)</strong>
                      <span>Checks statistical distribution stability, kurtosis tails, and chronological segment drift.</span>
                    </div>
                  </div>
                </Card>
              </section>

              {/* Section 3: Issue Ledger & DAG Categories */}
              <section id="issue-ledger" className="space-y-4 scroll-mt-28">
                <div className="flex items-center gap-3 border-b border-border/80 pb-2.5">
                  <div className="p-2 bg-blue-500/10 text-blue-700 rounded-xl">
                    <Layers className="h-5 w-5" />
                  </div>
                  <div>
                    <h2 className="text-xl sm:text-2xl font-display font-bold text-foreground uppercase tracking-tight">
                      3. Issue Ledger & Transformation DAG
                    </h2>
                    <p className="text-xs text-muted-foreground font-sans">
                      9 discrete structural issue categories audited and remediated by the engine.
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 font-mono text-xs">
                  {[
                    { name: "Duplicate Rows", fix: "Exact & Primary Key deduplication via Polars unique()" },
                    { name: "Currency Strings", fix: "Regex strip symbols ($, €, £) & cast to Float64" },
                    { name: "Extreme Outliers", fix: "Winsorization capping at 1st / 99th percentiles" },
                    { name: "Missing Values", fix: "Domain-aware median/mode imputation or sentinel fill" },
                    { name: "Constant Columns", fix: "Zero-variance variable isolation & feature pruning" },
                    { name: "Mixed Date Formats", fix: "Chronological ISO-8601 normalization to Date32" }
                  ].map((issue) => (
                    <Card key={issue.name} className="p-3.5 border border-border bg-card rounded-xl space-y-1 shadow-2xs">
                      <span className="font-bold text-foreground font-sans text-xs">{issue.name}</span>
                      <p className="text-[11px] text-muted-foreground font-mono">{issue.fix}</p>
                    </Card>
                  ))}
                </div>
              </section>

              {/* Section 4: REST API Endpoints */}
              <section id="api-endpoints" className="space-y-4 scroll-mt-28">
                <div className="flex items-center gap-3 border-b border-border/80 pb-2.5">
                  <div className="p-2 bg-purple-500/10 text-purple-700 rounded-xl">
                    <Code2 className="h-5 w-5" />
                  </div>
                  <div>
                    <h2 className="text-xl sm:text-2xl font-display font-bold text-foreground uppercase tracking-tight">
                      4. REST API Endpoint Specifications
                    </h2>
                    <p className="text-xs text-muted-foreground font-sans">
                      Headless JSON endpoints for programmatic data audits and CI/CD pipelines.
                    </p>
                  </div>
                </div>

                <div className="space-y-4 sm:space-y-5">
                  {apiEndpoints.map((ep) => (
                    <Card key={ep.path} className="border border-border bg-card rounded-2xl shadow-premium overflow-hidden font-mono text-xs">
                      <CardHeader className="bg-muted/30 border-b border-border/60 p-3.5 sm:p-4 flex flex-row items-center justify-between">
                        <div className="flex items-center gap-2.5">
                          <Badge className="bg-primary text-primary-foreground font-bold px-2 py-0.5 rounded">
                            {ep.method}
                          </Badge>
                          <span className="font-bold text-foreground text-xs sm:text-sm font-mono">{ep.path}</span>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleCopy(ep.responseBody, ep.path)}
                          className="h-7 px-2.5 font-mono text-[11px] gap-1.5 rounded-lg"
                        >
                          {copiedPath === ep.path ? <Check className="h-3.5 w-3.5 text-emerald-600" /> : <Copy className="h-3.5 w-3.5" />}
                          <span>{copiedPath === ep.path ? "Copied" : "Copy Schema"}</span>
                        </Button>
                      </CardHeader>
                      
                      <CardContent className="p-4 sm:p-5 space-y-3">
                        <p className="text-xs text-muted-foreground font-sans">{ep.description}</p>
                        
                        {ep.requestBody && (
                          <div className="space-y-1">
                            <span className="text-[10px] text-muted-foreground uppercase tracking-wider font-bold block">
                              Request Payload:
                            </span>
                            <pre className="p-2.5 bg-muted/40 rounded-xl border border-border/40 text-[11px] text-foreground overflow-x-auto leading-relaxed">
                              <code>{ep.requestBody}</code>
                            </pre>
                          </div>
                        )}

                        <div className="space-y-1">
                          <span className="text-[10px] text-muted-foreground uppercase tracking-wider font-bold block">
                            Response JSON:
                          </span>
                          <pre className="p-2.5 bg-zinc-950 text-emerald-400 rounded-xl border border-zinc-800 text-[11px] overflow-x-auto leading-relaxed">
                            <code>{ep.responseBody}</code>
                          </pre>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </section>

              {/* Section 5: Architecture Specs & Security */}
              <section id="security" className="space-y-4 scroll-mt-28">
                <div className="flex items-center gap-3 border-b border-border/80 pb-2.5">
                  <div className="p-2 bg-blue-500/10 text-blue-700 rounded-xl">
                    <ShieldCheck className="h-5 w-5" />
                  </div>
                  <div>
                    <h2 className="text-xl sm:text-2xl font-display font-bold text-foreground uppercase tracking-tight">
                      5. Architecture Specs & Security Invariants
                    </h2>
                    <p className="text-xs text-muted-foreground font-sans">
                      In-memory execution, encryption protocols, and zero-retention compliance.
                    </p>
                  </div>
                </div>

                <div className="border border-border/80 bg-card rounded-2xl shadow-premium overflow-hidden font-mono text-xs">
                  <div className="divide-y divide-border/40">
                    {[
                      { param: "Storage Model", detail: "100% In-Memory RAM (0 bytes written to permanent disks)" },
                      { param: "Session Lifetime", detail: "Automatic garbage collection after 60 minutes of inactivity" },
                      { param: "Encryption in Transit", detail: "TLS 1.3 enforced with strict HSTS (Strict-Transport-Security)" },
                      { param: "Content Security Policy", detail: "Strict CSP with script-src 'self' and frame-ancestors 'none'" },
                      { param: "Rate Limiting", detail: "Token-bucket limiter: 60 upload requests/min per IP address" },
                      { param: "AI Privacy Boundary", detail: "Raw customer/patient rows strictly omitted from external prompts" }
                    ].map((row, rIdx) => (
                      <div key={rIdx} className="px-4 sm:px-5 py-3 flex flex-col sm:flex-row sm:items-center justify-between gap-1.5 hover:bg-muted/10">
                        <span className="font-bold text-foreground font-sans">{row.param}:</span>
                        <span className="text-muted-foreground text-[11px]">{row.detail}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </section>

            </div>

          </div>
        </div>

        {/* Section: Bottom CTA */}
        <div className="border-t border-border/60 bg-muted/20 py-10 sm:py-12">
          <div className="container mx-auto px-4 text-center space-y-4 max-w-3xl">
            <h2 className="text-xl sm:text-2xl font-display font-bold text-foreground uppercase tracking-tight">
              Ready to audit a dataset?
            </h2>
            <p className="text-xs sm:text-sm text-muted-foreground font-sans max-w-lg mx-auto">
              Launch the interactive workspace and see these algorithms in action.
            </p>
            <div className="pt-1 flex flex-wrap items-center justify-center gap-3">
              <Link to="/workspace">
                <Button size="lg" className="h-11 px-7 rounded-xl shadow-premium t-card-lift t-spring-press font-display font-semibold text-sm">
                  <span>Launch Workspace</span>
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
            </div>
          </div>
        </div>

      </main>

      <Footer />
    </div>
  );
};

export default Documentation;
