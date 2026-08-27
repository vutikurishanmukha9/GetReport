import { useState } from "react";
import { 
  BarChart3, GraduationCap, Users, DollarSign, Heart, ArrowRight, 
  AlertTriangle, ChevronDown, ChevronUp, Eye
} from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";

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
    icon: Heart,
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
    <div className="min-h-screen flex flex-col bg-background animate-in fade-in duration-500">
      <Header onReset={() => {}} showReset={false} />

      <main className="flex-1 pt-16 sm:pt-20">
        {/* Header */}
        <div className="border-b border-border/60 bg-gradient-to-b from-muted/20 via-background to-background py-8 sm:py-12">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-5xl text-center space-y-3 sm:space-y-4">
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-primary/10 text-primary text-xs font-semibold uppercase tracking-wider font-mono border border-primary/20 t-badge-shimmer">
              <BarChart3 className="h-3.5 w-3.5" />
              <span>Real-World Sample Datasets</span>
            </div>

            <h1 className="text-3xl sm:text-4xl md:text-5xl font-display font-extrabold text-foreground tracking-tight uppercase leading-[1.08]">
              Interactive Audit Examples.
            </h1>
            
            <p className="text-sm sm:text-base text-muted-foreground max-w-2xl mx-auto leading-relaxed font-sans">
              Explore how GetReport automatically infers schemas, scores column quality, isolates anomalies, and synthesizes executive takeaways across multiple industries.
            </p>
          </div>
        </div>

        {/* Section: Dataset Catalog */}
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-10 max-w-7xl space-y-6 sm:space-y-8">
          
          {/* Domain Filter Bar */}
          <div className="flex overflow-x-auto gap-2 pb-1.5 scrollbar-none justify-start md:justify-center border-b border-border/60">
            {DOMAINS.map((domain) => (
              <button
                key={domain}
                type="button"
                onClick={() => setSelectedDomain(domain)}
                className={`px-3.5 py-1.5 rounded-xl text-xs font-mono transition-all shrink-0 cursor-pointer border ${
                  selectedDomain === domain
                    ? "bg-primary text-primary-foreground border-primary shadow-xs font-bold"
                    : "bg-card text-muted-foreground border-border hover:text-foreground hover:bg-muted/40"
                }`}
              >
                {domain}
              </button>
            ))}
          </div>

          {/* Dataset Cards Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5 lg:gap-6">
            {filteredExamples.map((item) => {
              const Icon = item.icon;
              const isExpanded = expandedId === item.id;

              return (
                <Card 
                  key={item.id}
                  className="border border-border bg-card rounded-2xl shadow-premium overflow-hidden flex flex-col justify-between t-card-lift"
                >
                  <CardHeader className="p-5 sm:p-6 pb-3 sm:pb-4 space-y-3.5 border-b border-border/40">
                    <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-2.5 sm:gap-4">
                      <div className="flex items-center gap-3">
                        <div className="h-9 w-9 rounded-xl bg-primary/10 text-primary flex items-center justify-center shrink-0">
                          <Icon className="h-4 w-4" />
                        </div>
                        <div>
                          <Badge variant="outline" className="text-[10px] font-mono font-semibold bg-muted/40 text-muted-foreground border-border/60">
                            {item.domain}
                          </Badge>
                          <h3 className="text-base sm:text-lg font-display font-bold text-foreground mt-0.5">
                            {item.title}
                          </h3>
                        </div>
                      </div>

                      {/* Grade Badge */}
                      <div className="self-start sm:self-auto sm:text-right shrink-0">
                        <Badge className={`font-mono font-bold text-xs px-2.5 py-0.5 rounded-lg ${
                          item.grade === "A" ? "bg-emerald-500/10 text-emerald-700 border-emerald-500/30" : "bg-blue-500/10 text-blue-700 border-blue-500/30"
                        }`}>
                          Grade {item.grade} • {item.gradeScore}%
                        </Badge>
                      </div>
                    </div>

                    <p className="text-xs text-muted-foreground font-sans leading-relaxed">
                      {item.description}
                    </p>

                    {/* Dataset Metrics Bar */}
                    <div className="flex items-center gap-3 font-mono text-[11px] text-muted-foreground pt-0.5">
                      <span><strong>Format:</strong> {item.format}</span>
                      <span>•</span>
                      <span><strong>Rows:</strong> {item.rows}</span>
                      <span>•</span>
                      <span><strong>Cols:</strong> {item.cols}</span>
                    </div>
                  </CardHeader>

                  <CardContent className="p-5 sm:p-6 pt-3.5 sm:pt-4 space-y-3.5 flex-1 flex flex-col justify-between">
                    {/* Quality Flags */}
                    <div className="space-y-1.5">
                      <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-muted-foreground block">
                        Identified Quality Flags:
                      </span>
                      <div className="flex flex-wrap gap-1.5 font-mono text-[11px]">
                        {item.detectedIssues.map((issue, iIdx) => (
                          <span key={iIdx} className="inline-flex items-center gap-1 bg-amber-500/10 text-amber-800 border border-amber-500/20 px-2 py-0.5 rounded-md text-[10px] sm:text-[11px]">
                            <AlertTriangle className="h-3 w-3 shrink-0" />
                            <span>{issue}</span>
                          </span>
                        ))}
                      </div>
                    </div>

                    {/* Executive Key Insight Quote */}
                    <div className="p-3 bg-muted/30 rounded-xl border border-border/40 text-xs font-sans text-foreground">
                      <strong className="text-primary font-mono text-[10px] uppercase tracking-wider block mb-0.5">RAG Executive Insight:</strong>
                      <p className="italic text-muted-foreground leading-relaxed">&ldquo;{item.keyInsight}&rdquo;</p>
                    </div>

                    {/* Expandable Schema Preview Drawer */}
                    {isExpanded && (
                      <div className="border border-border/60 bg-muted/20 rounded-xl p-3 space-y-1.5 font-mono text-[11px] animate-in slide-in-from-top-1 duration-200">
                        <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider block">
                          Column Remediation Sample:
                        </span>
                        <div className="divide-y divide-border/40">
                          {item.sampleColumns.map((col) => (
                            <div key={col.name} className="py-1 flex justify-between items-center text-xs">
                              <span className="font-semibold text-foreground">{col.name} <span className="text-[10px] text-muted-foreground">({col.type})</span></span>
                              <span className="text-emerald-700 font-medium text-[11px]">{col.quality}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Card Actions */}
                    <div className="flex items-center justify-between pt-2 border-t border-border/40">
                      <button
                        type="button"
                        onClick={() => toggleExpand(item.id)}
                        className="text-xs font-mono text-primary hover:underline flex items-center gap-1 cursor-pointer"
                      >
                        <Eye className="h-3.5 w-3.5" />
                        <span>{isExpanded ? "Hide Schema" : "Inspect Schema"}</span>
                        {isExpanded ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                      </button>

                      <Link to="/workspace">
                        <Button size="sm" className="rounded-xl font-display font-semibold text-xs gap-1 shadow-xs h-8 px-3">
                          <span>Audit in Workspace</span>
                          <ArrowRight className="h-3.5 w-3.5" />
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
        <div className="border-t border-border/60 bg-muted/20 py-10 sm:py-12">
          <div className="container mx-auto px-4 text-center space-y-4 max-w-3xl">
            <h2 className="text-xl sm:text-2xl font-display font-bold text-foreground uppercase tracking-tight">
              Have your own dataset to analyze?
            </h2>
            <p className="text-xs sm:text-sm text-muted-foreground font-sans max-w-lg mx-auto">
              GetReport is 100% free and processes data securely in ephemeral RAM with zero disk retention.
            </p>
            <div className="pt-1 flex flex-wrap items-center justify-center gap-3">
              <Link to="/workspace">
                <Button size="lg" className="h-11 px-7 rounded-xl shadow-premium t-card-lift t-spring-press font-display font-semibold text-sm">
                  <span>Start Free Audit</span>
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

export default Examples;
