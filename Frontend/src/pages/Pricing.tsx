import { Check, ArrowRight, Zap, Shield, HelpCircle, Sparkles, Building2, UserCheck } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";

const plans = [
  {
    name: "Free Audit",
    price: "$0",
    period: "forever",
    description: "Ideal for data analysts and individual researchers needing instant quality audits.",
    badge: null,
    features: [
      "Up to 50MB file ingestion per job",
      "Polars Rust multi-threaded execution",
      "Full Issue Ledger (9 quality categories)",
      "A-F Column Confidence Scores",
      "Board-ready PDF report downloads",
      "In-memory zero data retention",
    ],
    cta: "Start Free Audit",
    link: "/workspace",
    variant: "outline" as const,
  },
  {
    name: "Team Pro",
    price: "$49",
    period: "per seat / month",
    description: "Designed for data teams needing multi-dataset joins, custom threshold controls, and team DAG exports.",
    badge: "MOST POPULAR",
    features: [
      "Everything in Free Audit",
      "Multi-dataset relational joins (up to 5 files)",
      "Custom outlier threshold overrides (IQR / Z-Score)",
      "Deterministic RAG chat assistant for datasets",
      "Multi-format streaming exports (CSV, Parquet, HTML)",
      "Priority processing worker queue",
    ],
    cta: "Start 14-Day Free Trial",
    link: "/contact",
    variant: "default" as const,
  },
  {
    name: "Enterprise",
    price: "Custom",
    period: "annual billing",
    description: "For enterprise governance, custom on-premise deployments, and dedicated SLA support.",
    badge: "ENTERPRISE",
    features: [
      "Everything in Team Pro",
      "Custom file size limits (> 500MB streaming)",
      "Dedicated isolated compute workers",
      "On-premise Docker / Kubernetes deployment",
      "SSO & Custom SAML / OAuth authentication",
      "Dedicated 24/7 SLA & compliance support",
    ],
    cta: "Contact Enterprise Sales",
    link: "/contact",
    variant: "outline" as const,
  },
];

const faqs = [
  {
    question: "Is my data safe during processing?",
    answer: "Yes. All data processing occurs entirely in ephemeral server memory. Raw files are never written to permanent disk storage and are automatically purged after report synthesis.",
  },
  {
    question: "What file formats are supported?",
    answer: "GetReport supports CSV (.csv), Excel (.xls, .xlsx), Parquet (.parquet), TSV (.tsv), JSON/JSONL, Feather, Arrow, and compressed archives (.gz).",
  },
  {
    question: "How does the Issue Ledger work?",
    answer: "Our engine scans your dataset for 9 quality issue categories (missing values, duplicates, outliers, type mismatches, empty columns, etc.) and suggests automated Polars code fixes that you can approve or reject before execution.",
  },
  {
    question: "Can I join multiple datasets?",
    answer: "Yes! Team Pro and Enterprise plans support multi-dataset relational joins (inner, left, right, outer, semi, anti) on primary key columns with automatic column collision handling.",
  },
];

export const Pricing = () => {
  return (
    <div className="min-h-screen flex flex-col bg-background animate-in fade-in duration-500">
      <Header onReset={() => {}} showReset={false} />

      <main className="flex-1 pt-20">
        {/* Header */}
        <div className="border-b border-border/60 bg-gradient-to-b from-muted/30 to-background py-16 md:py-24">
          <div className="container mx-auto px-4 text-center space-y-4 max-w-4xl">
            <Badge variant="outline" className="font-mono text-xs uppercase tracking-wider text-primary border-primary/30 px-3 py-1">
              Simple & Transparent Pricing
            </Badge>
            <h1 className="text-4xl sm:text-5xl md:text-6xl font-display font-extrabold text-foreground tracking-tight uppercase leading-[1.05]">
              Predictable Plans for Every Team.
            </h1>
            <p className="text-base sm:text-lg text-muted-foreground max-w-2xl mx-auto leading-relaxed">
              Audit datasets instantly with our free tier, or upgrade for multi-dataset relational joins, custom threshold controls, and enterprise support.
            </p>
          </div>
        </div>

        {/* Pricing Cards */}
        <div className="container mx-auto px-4 py-16 md:py-24 max-w-7xl">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 items-stretch">
            {plans.map((plan, idx) => (
              <Card
                key={plan.name}
                className={`border bg-card rounded-2xl p-8 flex flex-col justify-between shadow-premium transition-all duration-300 relative ${
                  plan.badge ? "border-primary shadow-xl scale-105 z-10" : "border-border/80 hover:border-primary/30"
                }`}
              >
                <div>
                  {plan.badge && (
                    <Badge className="absolute -top-3.5 right-6 bg-primary text-primary-foreground text-[10px] font-mono font-bold uppercase tracking-wider px-3 py-1 rounded-full shadow-sm">
                      {plan.badge}
                    </Badge>
                  )}

                  <div className="space-y-3 mb-6">
                    <h3 className="text-xl font-display font-bold text-foreground uppercase tracking-tight">
                      {plan.name}
                    </h3>
                    <div className="flex items-baseline gap-1">
                      <span className="text-4xl font-display font-extrabold text-foreground">{plan.price}</span>
                      <span className="text-xs text-muted-foreground font-mono">{plan.period}</span>
                    </div>
                    <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                      {plan.description}
                    </p>
                  </div>

                  <div className="space-y-3 pt-4 border-t border-border/60 mb-8">
                    <span className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground font-bold">Included Features</span>
                    <ul className="space-y-2.5 text-xs text-foreground font-sans">
                      {plan.features.map((feature, fIdx) => (
                        <li key={fIdx} className="flex items-start gap-2.5">
                          <Check className="h-4 w-4 text-emerald-600 shrink-0 mt-0.5" />
                          <span>{feature}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>

                <div className="pt-4">
                  <Link to={plan.link}>
                    <Button
                      size="lg"
                      variant={plan.variant}
                      className={`w-full rounded-xl h-11 font-display font-semibold text-sm ${
                        plan.badge ? "bg-primary text-primary-foreground hover:bg-primary/90 shadow-md" : ""
                      }`}
                    >
                      <span>{plan.cta}</span>
                      <ArrowRight className="ml-2 h-4 w-4" />
                    </Button>
                  </Link>
                </div>
              </Card>
            ))}
          </div>
        </div>

        {/* FAQ Section */}
        <div className="border-t border-border/60 bg-muted/20 py-16 md:py-24">
          <div className="container mx-auto px-4 max-w-4xl">
            <div className="text-center mb-12 space-y-2">
              <h2 className="text-2xl sm:text-3xl font-display font-bold text-foreground uppercase tracking-tight">Frequently Asked Questions</h2>
              <p className="text-xs sm:text-sm text-muted-foreground">Everything you need to know about GetReport security, limits, and pricing.</p>
            </div>

            <Accordion type="single" collapsible className="space-y-4">
              {faqs.map((faq, idx) => (
                <AccordionItem key={idx} value={`item-${idx}`} className="border border-border/80 bg-card rounded-2xl px-6 shadow-xs">
                  <AccordionTrigger className="text-sm font-display font-bold text-foreground py-4 hover:no-underline hover:text-primary">
                    {faq.question}
                  </AccordionTrigger>
                  <AccordionContent className="text-xs sm:text-sm text-muted-foreground leading-relaxed pb-4">
                    {faq.answer}
                  </AccordionContent>
                </AccordionItem>
              ))}
            </Accordion>
          </div>
        </div>

      </main>

      <Footer />
    </div>
  );
};

export default Pricing;
