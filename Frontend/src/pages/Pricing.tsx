import { Check, ArrowRight, Zap, Shield, HelpCircle, Sparkles, Building2, UserCheck, Code2, Globe, Heart } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";

const plans = [
  {
    name: "Browser & Cloud Audit",
    price: "$0",
    period: "100% FREE FOREVER",
    description: "Instant in-memory dataset audits directly in your web browser with zero registration or paywalls.",
    badge: "RECOMMENDED",
    features: [
      "Full file ingestion (CSV, XLSX, Parquet, TSV, JSONL)",
      "Interactive Issue Ledger (9 quality issue categories)",
      "A-F Column Confidence Scoring System",
      "Multi-dataset relational joins (up to 5 files)",
      "Deterministic RAG AI Companion for dataset chat",
      "WeasyPrint executive PDF report downloads",
      "Multi-format streaming export (CSV, Parquet, HTML)",
      "100% ephemeral in-memory processing security",
    ],
    cta: "Start Free Audit",
    link: "/workspace",
    variant: "default" as const,
  },
  {
    name: "Self-Hosted Open Source",
    price: "$0",
    period: "OPEN SOURCE",
    description: "Deploy GetReport locally on your workstation or company servers with Docker or Python virtualenv.",
    badge: "DEVELOPERS",
    features: [
      "100% full source code access on GitHub",
      "Local Python & Celery task execution",
      "Custom memory & file size limits (unrestricted RAM)",
      "Zero data leaving your local machine or network",
      "Custom LLM API key integration (OpenAI / OpenRouter)",
      "Full API routing & custom backend service extendability",
    ],
    cta: "View GitHub Repository",
    link: "https://github.com/vutikurishanmukha9/GetReport",
    external: true,
    variant: "outline" as const,
  },
  {
    name: "Enterprise Self-Hosted",
    price: "$0",
    period: "COMMUNITY EDITION",
    description: "Deploy in private Kubernetes clusters or AWS/GCP VPC environments with custom compliance controls.",
    badge: "ENTERPRISE READY",
    features: [
      "Complete deployment guides for Docker & Render",
      "Isolated celery worker pool configuration",
      "PostgreSQL asyncpg connection pool tuning",
      "Security header defenses (HSTS, CSP, banner masking)",
      "Automated quality remediation DAG execution",
      "Zero vendor lock-in or subscription fees",
    ],
    cta: "Explore Architecture Specs",
    link: "/documentation#security",
    variant: "outline" as const,
  },
];

const faqs = [
  {
    question: "Is GetReport really 100% free?",
    answer: "Yes! GetReport is completely free and open source. All features—including multi-dataset relational joins, Issue Ledger remediation, AI companion chat, and board-ready PDF generation—are unlocked for everyone with zero fees or paywalls.",
  },
  {
    question: "Is my uploaded data safe and private?",
    answer: "Absolutely. All data processing occurs entirely in ephemeral server memory. Raw files are never written to permanent disk storage, are never used to train AI models, and are automatically purged from memory after report generation or 1 hour of session inactivity.",
  },
  {
    question: "What file formats can I audit for free?",
    answer: "GetReport supports CSV (.csv), Excel (.xls, .xlsx), Parquet (.parquet), TSV (.tsv), JSON/JSONL, Feather, Arrow, and compressed archives (.gz).",
  },
  {
    question: "How does the Issue Ledger work?",
    answer: "Our engine scans your dataset for 9 quality issue categories (missing values, duplicates, outliers, type mismatches, empty columns, etc.) and suggests automated Polars code fixes that you can approve or reject before execution.",
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
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-emerald-500/10 text-emerald-700 text-xs font-semibold uppercase tracking-wider font-mono border border-emerald-500/20">
              <Heart className="h-3.5 w-3.5 fill-emerald-500" />
              <span>100% Free & Open Source for Everyone</span>
            </div>

            <h1 className="text-4xl sm:text-5xl md:text-6xl font-display font-extrabold text-foreground tracking-tight uppercase leading-[1.05]">
              All Features Unlocked. $0 Forever.
            </h1>
            <p className="text-base sm:text-lg text-muted-foreground max-w-2xl mx-auto leading-relaxed">
              No hidden fees, no credit cards required, and no feature paywalls. Experience full automated data quality audits, multi-dataset joins, and executive PDF reports for free.
            </p>
          </div>
        </div>

        {/* Pricing Cards */}
        <div className="container mx-auto px-4 py-16 md:py-24 max-w-7xl">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 items-stretch">
            {plans.map((plan, idx) => (
              <Card
                key={plan.name}
                className={`border bg-card rounded-2xl p-6 sm:p-8 flex flex-col justify-between shadow-sm hover:shadow-md transition-all duration-300 relative ${
                  plan.badge === "RECOMMENDED" ? "border-primary shadow-lg ring-1 ring-primary/20" : "border-border/80 hover:border-primary/30"
                }`}
              >
                <div>
                  <div className="flex items-center justify-between gap-2 mb-4">
                    <h3 className="text-lg sm:text-xl font-display font-bold text-foreground uppercase tracking-tight">
                      {plan.name}
                    </h3>
                    {plan.badge && (
                      <Badge className="bg-primary text-primary-foreground text-[9px] font-mono font-bold uppercase tracking-wider px-2.5 py-0.5 rounded-md whitespace-nowrap shrink-0">
                        {plan.badge}
                      </Badge>
                    )}
                  </div>

                  <div className="space-y-3 mb-6">
                    <div className="flex items-baseline gap-1">
                      <span className="text-4xl font-display font-extrabold text-foreground">{plan.price}</span>
                      <span className="text-xs text-emerald-700 font-mono font-bold">{plan.period}</span>
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
                  {plan.external ? (
                    <a href={plan.link} target="_blank" rel="noopener noreferrer">
                      <Button
                        size="lg"
                        variant={plan.variant}
                        className="w-full rounded-xl h-11 font-display font-semibold text-sm"
                      >
                        <span>{plan.cta}</span>
                        <ArrowRight className="ml-2 h-4 w-4" />
                      </Button>
                    </a>
                  ) : (
                    <Link to={plan.link}>
                      <Button
                        size="lg"
                        variant={plan.variant}
                        className={`w-full rounded-xl h-11 font-display font-semibold text-sm ${
                          plan.badge === "RECOMMENDED" ? "bg-primary text-primary-foreground hover:bg-primary/90 shadow-md" : ""
                        }`}
                      >
                        <span>{plan.cta}</span>
                        <ArrowRight className="ml-2 h-4 w-4" />
                      </Button>
                    </Link>
                  )}
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
              <p className="text-xs sm:text-sm text-muted-foreground">Everything you need to know about GetReport's free platform and security guarantees.</p>
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
