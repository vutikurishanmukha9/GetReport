import { BarChart, GraduationCap, Users, DollarSign, Heart, ArrowRight, CheckCircle2 } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";
import { useState } from "react";

const examples = [
  {
    icon: BarChart,
    title: "Sales & E-Commerce Audit",
    description: "Retail transaction dataset with Q1-Q4 revenue, customer acquisition channels, and returns.",
    domain: "Sales & E-Commerce",
    grade: "Grade A",
    features: ["Trend detection", "Seasonality analysis", "Revenue correlation", "Returns anomaly check"],
  },
  {
    icon: GraduationCap,
    title: "Academic Performance Ledger",
    description: "Student test scores, attendance records, demographic distribution, and graduation risk.",
    domain: "Education",
    grade: "Grade B",
    features: ["Score distributions", "Attendance impact", "Grade predictions", "Outlier capping"],
  },
  {
    icon: Users,
    title: "SaaS Customer Churn Dataset",
    description: "Subscription tier metrics, monthly active usage, support ticket count, and churn indicators.",
    domain: "SaaS & Retention",
    grade: "Grade A",
    features: ["Churn indicators", "Engagement scores", "Cohort analysis", "Missing value fill"],
  },
  {
    icon: DollarSign,
    title: "Financial Ledger & Budget Variance",
    description: "Quarterly expense tracking, department budgets, vendor payments, and cost variances.",
    domain: "Finance & Banking",
    grade: "Grade A",
    features: ["Budget variance", "Category breakdown", "Trend forecasting", "Currency coercion"],
  },
  {
    icon: Heart,
    title: "Clinical Trial Vitals Audit",
    description: "Patient vitals, dosage frequency, treatment outcomes, and adverse response logs.",
    domain: "Healthcare",
    grade: "Grade B",
    features: ["Vital correlations", "Outcome analysis", "Risk stratification", "Missing cell imputation"],
  },
  {
    icon: Users,
    title: "HR Workforce & Attrition Data",
    description: "Employee tenure, salary bands, performance reviews, and department turnover risks.",
    domain: "HR & People Analytics",
    grade: "Grade A",
    features: ["Tenure analysis", "Salary skewness", "Turnover risk", "Duplicate row drop"],
  },
];

const DOMAINS = ["All", "Sales & E-Commerce", "Education", "SaaS & Retention", "Finance & Banking", "Healthcare", "HR & People Analytics"] as const;

export const Examples = () => {
  const [selectedDomain, setSelectedDomain] = useState<string>("All");

  const filteredExamples = selectedDomain === "All"
    ? examples
    : examples.filter(e => e.domain === selectedDomain);

  return (
    <div className="min-h-screen flex flex-col bg-background animate-in fade-in duration-500">
      <Header onReset={() => {}} showReset={false} />

      <main className="flex-1 pt-20">
        {/* Header */}
        <div className="border-b border-border/60 bg-gradient-to-b from-muted/30 to-background py-16 md:py-24">
          <div className="container mx-auto px-4 text-center space-y-4 max-w-4xl">
            <Badge variant="outline" className="font-mono text-xs uppercase tracking-wider text-primary border-primary/30 px-3 py-1">
              Sample Datasets & Domain Intelligence
            </Badge>
            <h1 className="text-4xl sm:text-5xl md:text-6xl font-display font-extrabold text-foreground tracking-tight uppercase leading-[1.05]">
              Interactive Audit Examples.
            </h1>
            <p className="text-base sm:text-lg text-muted-foreground max-w-2xl mx-auto leading-relaxed">
              Explore how GetReport's semantic domain detection and automated issue ledger audit real-world datasets across industries.
            </p>
          </div>
        </div>

        {/* Content */}
        <div className="container mx-auto px-4 py-16 max-w-7xl space-y-12">
          
          {/* Domain Filter Pills */}
          <div className="flex flex-wrap items-center justify-center gap-2">
            {DOMAINS.map((domain) => (
              <button
                key={domain}
                onClick={() => setSelectedDomain(domain)}
                className={`px-4 py-2 rounded-full text-xs font-mono transition-all duration-150 ${
                  selectedDomain === domain
                    ? "bg-primary text-primary-foreground font-bold shadow-md"
                    : "bg-muted/60 text-muted-foreground hover:bg-muted hover:text-foreground border border-border/60"
                }`}
              >
                {domain}
              </button>
            ))}
          </div>

          {/* Examples Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredExamples.map((example) => {
              const Icon = example.icon;
              return (
                <Card key={example.title} className="hover:shadow-xl transition-all duration-300 group rounded-2xl border border-border/80 bg-card p-6 flex flex-col justify-between hover:border-primary/30">
                  <div>
                    <div className="flex items-center justify-between mb-4">
                      <div className="h-12 w-12 rounded-xl bg-primary/10 text-primary flex items-center justify-center group-hover:scale-105 transition-transform">
                        <Icon className="h-6 w-6" />
                      </div>
                      <Badge className="bg-emerald-100 text-emerald-800 border-emerald-300 font-mono text-[10px] font-bold">
                        {example.grade}
                      </Badge>
                    </div>

                    <div className="space-y-2 mb-4">
                      <span className="text-[10px] font-mono font-bold text-primary uppercase tracking-wider">
                        {example.domain}
                      </span>
                      <h3 className="font-display font-bold text-lg text-foreground uppercase tracking-tight">
                        {example.title}
                      </h3>
                      <p className="text-xs text-muted-foreground leading-relaxed">
                        {example.description}
                      </p>
                    </div>

                    <div className="space-y-2 pt-3 border-t border-border/60 mb-6">
                      <span className="text-[9px] font-mono text-muted-foreground uppercase tracking-wider font-bold">Detected Features</span>
                      <div className="flex flex-wrap gap-1.5">
                        {example.features.map((feature, fIdx) => (
                          <span key={fIdx} className="text-[10px] font-mono bg-muted/60 text-muted-foreground px-2 py-0.5 rounded border border-border/40 flex items-center gap-1">
                            <CheckCircle2 className="h-3 w-3 text-emerald-600 shrink-0" />
                            {feature}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>

                  <Link to="/workspace">
                    <Button variant="outline" size="sm" className="w-full rounded-xl font-display text-xs font-semibold hover:bg-primary hover:text-primary-foreground transition-all">
                      <span>Test in Workspace</span>
                      <ArrowRight className="ml-1.5 h-3.5 w-3.5" />
                    </Button>
                  </Link>
                </Card>
              );
            })}
          </div>

        </div>

      </main>

      <Footer />
    </div>
  );
};

export default Examples;
