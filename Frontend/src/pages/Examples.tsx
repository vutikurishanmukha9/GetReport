import { useState } from "react";
import { 
  BarChart3, GraduationCap, Users, DollarSign, Activity, ArrowRight, 
  AlertTriangle, ChevronDown, ChevronUp, Eye, FileText
} from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";
import { CtaBanner } from "@/components/CtaBanner";

interface DatasetExample {
  id: string;
  icon: typeof BarChart3;
  title: string;
  domain: string;
  description: string;
  grade: string;
  gradeScore: number;
  format: string;
  rows: string;
  cols: number;
  detectedIssues: string[];
  sampleColumns: { name: string; type: string; quality: string }[];
  keyInsight: string;
}

const examplesList: DatasetExample[] = [
  {
    id: "sales_retail",
    icon: BarChart3,
    title: "Global E-Commerce Sales Ledger",
    domain: "Sales & E-Commerce",
    description: "Multinational retail transaction records covering product categories, discounts, customer loyalty tiers, and shipping lead times.",
    grade: "A",
    gradeScore: 94.8,
    format: "CSV",
    rows: "48,500",
    cols: 14,
    detectedIssues: ["42 currency symbol strings", "14 duplicate transaction IDs", "2.1% missing shipping dates"],
    sampleColumns: [
      { name: "order_id", type: "Int64", quality: "100% Valid" },
      { name: "revenue_usd", type: "Float64", quality: "Coerced from string" },
      { name: "customer_tier", type: "Categorical", quality: "4 Clean Categories" },
      { name: "order_date", type: "Date32", quality: "ISO-8601 Validated" }
    ],
    keyInsight: "Strong positive correlation between loyalty tier and average order value (r = 0.78). Returns spike in November."
  },
  {
    id: "education_perf",
    icon: GraduationCap,
    title: "University Academic Cohort Ledger",
    domain: "Education",
    description: "Multi-semester student enrollment metrics, GPA distributions, prerequisite completions, and graduation velocity indicators.",
    grade: "B",
    gradeScore: 84.2,
    format: "Parquet",
    rows: "12,400",
    cols: 11,
    detectedIssues: ["8.4% null prerequisite grades", "Extreme outlier in credits earned (>180)", "Mixed letter vs numeric scoring"],
    sampleColumns: [
      { name: "student_id", type: "Int64", quality: "100% Unique" },
      { name: "term_gpa", type: "Float64", quality: "Winsorized [0.0 - 4.0]" },
      { name: "credits_earned", type: "Int32", quality: "Outlier Capped" },
      { name: "status", type: "Utf8", quality: "Remediated Case" }
    ],
    keyInsight: "Attendance rates below 75% exhibit exponential drop in course completion rates."
  },
  {
    id: "saas_churn",
    icon: Users,
    title: "SaaS Retention & Churn Analytics",
    domain: "SaaS & Retention",
    description: "B2B SaaS telemetry including monthly recurring revenue (MRR), active seats, support ticket volume, and renewal flags.",
    grade: "A",
    gradeScore: 96.1,
    format: "XLSX",
    rows: "28,200",
    cols: 16,
    detectedIssues: ["Leading whitespace in account names", "3 negative MRR values corrected", "Empty department strings"],
    sampleColumns: [
      { name: "account_id", type: "Int64", quality: "100% Valid" },
      { name: "mrr_usd", type: "Float64", quality: "Range Validated" },
      { name: "active_seats", type: "Int32", quality: "Clean Integers" },
      { name: "churn_risk", type: "Float64", quality: "Calibrated 0-1" }
    ],
    keyInsight: "Support ticket resolution time > 48h strongly drives churn risk in Enterprise tier."
  },
  {
    id: "finance_budget",
    icon: DollarSign,
    title: "Quarterly Departmental Budget Variance",
    domain: "Finance & Banking",
    description: "Corporate cost center ledger with projected allocations, vendor payment invoices, currency conversions, and cost variance.",
    grade: "A",
    gradeScore: 95.4,
    format: "CSV",
    rows: "18,900",
    cols: 12,
    detectedIssues: ["Mixed EUR/USD currency strings", "9 trailing space vendor names", "0 duplicate invoice IDs"],
    sampleColumns: [
      { name: "cost_center", type: "Utf8", quality: "Standardized Taxon" },
      { name: "budget_allocated", type: "Float64", quality: "Validated Currency" },
      { name: "variance_pct", type: "Float64", quality: "Derived Formula" },
      { name: "fiscal_quarter", type: "Utf8", quality: "Q1-Q4 Uniform" }
    ],
    keyInsight: "R&D cost variance remained within 3.2% of target budget while Marketing variance exceeded 14.5%."
  },
  {
    id: "clinical_vitals",
    icon: Activity,
    title: "Clinical Trial Patient Vitals Audit",
    domain: "Healthcare",
    description: "De-identified biometric trial records tracking systolic blood pressure, dosage frequency, and adverse symptom logs.",
    grade: "B",
    gradeScore: 82.7,
    format: "Parquet",
    rows: "8,500",
    cols: 15,
    detectedIssues: ["12.5% missing BMI measurements", "Non-standard systolic blood pressure strings", "2 duplicate patient IDs dropped"],
    sampleColumns: [
      { name: "subject_id", type: "Int64", quality: "De-identified Key" },
      { name: "systolic_bp", type: "Int32", quality: "Parsed from '120/80'" },
      { name: "dosage_mg", type: "Float64", quality: "Unit Normalized" },
      { name: "adverse_event", type: "Boolean", quality: "Clean Binary" }
    ],
    keyInsight: "Zero critical drug interaction anomalies detected in Cohort A. Mean systolic pressure reduced by 8.4 mmHg."
  },
  {
    id: "hr_attrition",
    icon: Users,
    title: "Workforce Compensation & Attrition Data",
    domain: "HR Analytics",
    description: "Enterprise headcount metrics with compensation salary bands, tenure duration, performance ratings, and department turnover.",
    grade: "A",
    gradeScore: 93.9,
    format: "CSV",
    rows: "16,400",
    cols: 13,
    detectedIssues: ["4 empty job titles imputed", "Salary skewness detected (log transformed)", "0 duplicate employee IDs"],
    sampleColumns: [
      { name: "employee_id", type: "Int64", quality: "Unique Key" },
      { name: "salary_band", type: "Categorical", quality: "8 Uniform Levels" },
      { name: "tenure_months", type: "Int32", quality: "Clean Range" },
      { name: "performance_score", type: "Float64", quality: "1.0 - 5.0 Standard" }
    ],
    keyInsight: "Salary band equity ratio shows strong correlation with voluntary retention across Engineering roles."
  }
];

const DOMAINS = ["All", "Sales & E-Commerce", "Education", "SaaS & Retention", "Finance & Banking", "Healthcare", "HR Analytics"] as const;

export const Examples = () => {
  const [selectedDomain, setSelectedDomain] = useState<string>("All");
  const [expandedId, setExpandedId] = useState<string | null>("sales_retail");

  const filteredExamples = selectedDomain === "All"
    ? examplesList
    : examplesList.filter(e => e.domain === selectedDomain);

  const toggleExpand = (id: string) => {
    setExpandedId(prev => prev === id ? null : id);
  };

  return (
    <div className="min-h-screen flex flex-col bg-background text-foreground transition-colors duration-200 animate-in fade-in duration-500 relative">
      {/* Technical Background Grid & Film Grain */}
      <div className="absolute inset-0 bg-[radial-gradient(rgba(0,0,0,0.04)_1px,transparent_1px)] dark:bg-[radial-gradient(rgba(255,255,255,0.06)_1px,transparent_1px)] [background-size:24px_24px] pointer-events-none opacity-50 dark:opacity-40" />
      <div className="hero-grain-overlay" />

      <Header onReset={() => {}} showReset={false} />

      <main className="flex-1 pt-24 sm:pt-28 relative z-10">
        {/* Header */}
        <div className="border-b border-border bg-background/90 py-8 sm:py-14 relative z-10">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-5xl text-center space-y-3 sm:space-y-4">

            <h1 className="text-3xl sm:text-4xl md:text-5xl font-display font-black text-foreground tracking-[-0.03em] leading-[1.12]">
              Interactive audit examples.
            </h1>
            
            <p className="text-sm sm:text-base text-muted-foreground max-w-2xl mx-auto leading-relaxed font-sans">
              Explore how GetReport automatically infers schemas, scores column quality, isolates anomalies, and synthesizes executive takeaways across multiple industries.
            </p>
          </div>
        </div>

        {/* Section: Dataset Catalog */}
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 max-w-7xl space-y-6 sm:space-y-8">
          
          {/* Domain Filter Bar */}
          <div className="flex overflow-x-auto gap-2 pb-2 scrollbar-none justify-start md:justify-center border-b border-border">
            {DOMAINS.map((domain) => (
              <button
                key={domain}
                type="button"
                onClick={() => setSelectedDomain(domain)}
                className={`px-3.5 py-1.5 rounded-xl text-xs font-mono transition-all shrink-0 cursor-pointer border ${
                  selectedDomain === domain
                    ? "bg-slate-900 text-white dark:bg-white dark:text-black border-slate-900 dark:border-white shadow-xs font-bold"
                    : "bg-slate-100 text-slate-700 hover:text-slate-950 hover:bg-slate-200 border-slate-200 dark:bg-card dark:text-muted-foreground dark:hover:text-foreground dark:border-white/10 font-semibold"
                }`}
              >
                {domain}
              </button>
            ))}
          </div>

          {/* Dataset Cards Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5 lg:gap-6 items-start">
            {filteredExamples.map((item) => {
              const Icon = item.icon;
              const isExpanded = expandedId === item.id;

              return (
                <Card 
                  key={item.id}
                  className="border border-slate-200 dark:border-white/10 bg-white dark:bg-[#0c0d12]/90 backdrop-blur-md rounded-2xl shadow-sm hover:shadow-md dark:shadow-none hover:border-slate-300 dark:hover:border-white/20 transition-all duration-200 flex flex-col justify-between text-card-foreground"
                >
                  <CardHeader className="p-5 sm:p-6 pb-3 sm:pb-4 space-y-3.5 border-b border-slate-200 dark:border-white/10">
                    <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-2.5 sm:gap-4">
                      <div className="flex items-center gap-3">
                        <div className="h-9 w-9 rounded-xl bg-slate-100 text-slate-800 dark:bg-white/[0.06] dark:text-white border border-slate-200 dark:border-white/10 flex items-center justify-center shrink-0">
                          <Icon className="h-4 w-4" />
                        </div>
                        <div>
                          <Badge variant="outline" className="text-[10px] font-mono font-bold bg-slate-100 text-slate-700 border-slate-200 dark:bg-white/[0.04] dark:text-muted-foreground dark:border-white/10">
                            {item.domain}
                          </Badge>
                          <h3 className="text-base sm:text-lg font-display font-bold text-foreground mt-0.5">
                            {item.title}
                          </h3>
                        </div>
                      </div>

                      {/* Grade Badge */}
                      <div className="self-start sm:self-auto sm:text-right shrink-0">
                        <Badge className={`font-mono font-bold text-xs px-2.5 py-0.5 rounded-lg shadow-2xs ${
                          item.grade === "A" 
                            ? "bg-emerald-100 text-emerald-950 border border-emerald-300 dark:bg-emerald-500/10 dark:text-emerald-400 dark:border-emerald-500/30" 
                            : "bg-blue-100 text-blue-950 border border-blue-300 dark:bg-blue-500/10 dark:text-blue-400 dark:border-blue-500/30"
                        }`}>
                          Grade {item.grade} • {item.gradeScore}%
                        </Badge>
                      </div>
                    </div>

                    <p className="text-xs text-slate-600 dark:text-muted-foreground font-sans leading-relaxed">
                      {item.description}
                    </p>

                    {/* Dataset Metrics Bar */}
                    <div className="flex items-center gap-3 font-mono text-[11px] text-slate-600 dark:text-muted-foreground pt-0.5">
                      <span><strong>Format:</strong> <span className="text-foreground">{item.format}</span></span>
                      <span>•</span>
                      <span><strong>Rows:</strong> <span className="text-foreground">{item.rows}</span></span>
                      <span>•</span>
                      <span><strong>Cols:</strong> <span className="text-foreground">{item.cols}</span></span>
                    </div>
                  </CardHeader>

                  <CardContent className="p-5 sm:p-6 pt-3.5 sm:pt-4 space-y-3.5 flex-1 flex flex-col justify-between">
                    {/* Identified Quality Flags - Adaptive High-Contrast Container */}
                    <div className="rounded-2xl bg-amber-50/70 dark:bg-gradient-to-b dark:from-amber-500/[0.05] dark:via-zinc-900/60 dark:to-zinc-950/80 border border-amber-200 dark:border-amber-500/30 p-4 space-y-3 shadow-2xs dark:shadow-[0_2px_12px_rgba(0,0,0,0.35)] backdrop-blur-md">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2.5">
                          <span className="w-6 h-6 rounded-lg bg-amber-100 text-amber-800 dark:bg-amber-500/20 dark:text-amber-300 border border-amber-300 dark:border-amber-500/40 flex items-center justify-center shrink-0 shadow-xs">
                            <AlertTriangle className="h-3.5 w-3.5 stroke-[2.4]" />
                          </span>
                          <span className="text-xs font-bold text-amber-950 dark:text-zinc-100 font-sans tracking-tight">Identified Quality Flags</span>
                        </div>
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-mono font-bold bg-amber-100 dark:bg-amber-500/20 text-amber-900 dark:text-amber-300 border border-amber-300 dark:border-amber-500/40 shadow-2xs">
                          <span className="w-1.5 h-1.5 rounded-full bg-amber-500 dark:bg-amber-400 shadow-[0_0_6px_rgba(245,158,11,0.9)] animate-pulse" />
                          {item.detectedIssues.length} Detected
                        </span>
                      </div>

                      <div className="flex flex-wrap gap-2 pt-0.5">
                        {item.detectedIssues.map((issue, iIdx) => {
                          const numMatch = issue.match(/^([0-9.]+%?)\s+(.*)$/);
                          return (
                            <span
                              key={iIdx}
                              className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-medium text-slate-800 dark:text-zinc-100 bg-white dark:bg-zinc-900/90 hover:bg-slate-50 dark:hover:bg-zinc-800/90 border border-slate-200 dark:border-white/10 hover:border-amber-400/50 shadow-2xs hover:shadow-xs transition-all duration-200 select-none group/flag cursor-default"
                            >
                              {numMatch ? (
                                <span className="inline-flex items-center justify-center px-1.5 py-0.5 rounded-md text-[11px] font-mono font-bold bg-amber-100 dark:bg-amber-500/20 text-amber-900 dark:text-amber-300 border border-amber-300 dark:border-amber-500/40 group-hover/flag:bg-amber-200 dark:group-hover/flag:bg-amber-500/30 transition-colors">
                                  {numMatch[1]}
                                </span>
                              ) : (
                                <span className="w-4 h-4 rounded-md bg-amber-100 text-amber-900 dark:bg-amber-500/20 dark:text-amber-300 flex items-center justify-center shrink-0 border border-amber-300 dark:border-amber-500/30">
                                  <AlertTriangle className="w-2.5 h-2.5 stroke-[2.4]" />
                                </span>
                              )}
                              <span className="font-sans tracking-tight text-slate-700 dark:text-zinc-200 group-hover/flag:text-slate-950 dark:group-hover/flag:text-white font-medium">
                                {numMatch ? numMatch[2] : issue}
                              </span>
                            </span>
                          );
                        })}
                      </div>
                    </div>

                    {/* Executive Key Insight Takeaway */}
                    <div className="p-3.5 bg-slate-50 dark:bg-white/[0.03] rounded-xl border border-slate-200 dark:border-white/[0.08] text-xs font-sans space-y-1.5 shadow-2xs">
                      <div className="flex items-center gap-1.5 text-[10px] font-mono font-bold uppercase tracking-wider text-slate-700 dark:text-primary">
                        <FileText className="h-3.5 w-3.5 text-primary shrink-0" />
                        <span>RAG Executive Takeaway</span>
                      </div>
                      <p className="italic text-slate-800 dark:text-zinc-200 text-xs leading-relaxed font-sans font-medium">&ldquo;{item.keyInsight}&rdquo;</p>
                    </div>

                    {/* Expandable Schema Preview Drawer */}
                    {isExpanded && (
                      <div className="border border-slate-200 dark:border-white/[0.08] bg-slate-50/90 dark:bg-zinc-950/60 rounded-xl p-3.5 space-y-2 font-mono text-[11px] shadow-2xs animate-in slide-in-from-top-1 duration-200">
                        <span className="text-[10px] font-bold text-slate-700 dark:text-zinc-400 uppercase tracking-wider block">
                          Column Remediation Sample:
                        </span>
                        <div className="divide-y divide-slate-200 dark:divide-white/[0.05]">
                          {item.sampleColumns.map((col) => (
                            <div key={col.name} className="py-1.5 flex justify-between items-center text-xs">
                              <span className="font-semibold text-slate-900 dark:text-zinc-200">{col.name} <span className="text-[10px] text-slate-500 font-normal">({col.type})</span></span>
                              <span className="text-emerald-900 dark:text-emerald-400 font-bold text-[11px] px-2 py-0.5 rounded-md bg-emerald-100 dark:bg-emerald-500/10 border border-emerald-300 dark:border-emerald-500/20 shadow-2xs">{col.quality}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Card Actions */}
                    <div className="flex items-center justify-between pt-3 border-t border-slate-200 dark:border-white/[0.06]">
                      <button
                        type="button"
                        onClick={() => toggleExpand(item.id)}
                        className="text-xs font-semibold text-slate-600 hover:text-slate-950 dark:text-zinc-400 dark:hover:text-white flex items-center gap-1.5 transition-colors cursor-pointer py-1.5 px-2.5 rounded-lg hover:bg-slate-100 dark:hover:bg-white/[0.06]"
                      >
                        <Eye className="h-3.5 w-3.5 text-slate-500 dark:text-zinc-400" />
                        <span>{isExpanded ? "Hide Schema" : "Inspect Schema"}</span>
                        {isExpanded ? <ChevronUp className="h-3.5 w-3.5 text-slate-500 dark:text-zinc-500" /> : <ChevronDown className="h-3.5 w-3.5 text-slate-500 dark:text-zinc-500" />}
                      </button>

                      <Link to={`/workspace?sample=${item.id}`}>
                        <Button size="sm" variant="hero" className="rounded-xl text-xs gap-1.5 h-8 px-3.5 font-bold shadow-xs cursor-pointer">
                          <span>Audit in Workspace</span>
                          <ArrowRight className="h-3.5 w-3.5 stroke-[2.5]" />
                        </Button>
                      </Link>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </div>

        {/* Bottom CTA */}
        <CtaBanner
          title="Have your own dataset to analyze?"
          description="GetReport is 100% free and processes data securely in ephemeral RAM with zero disk retention."
          buttonText="Start Free Audit"
          buttonLink="/workspace"
        />

      </main>

      <Footer />
    </div>
  );
};

export default Examples;
