import { Check, ArrowRight, Shield, Heart } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";

const currentOfferingFeatures = [
  "Multi-format file ingestion (CSV, XLSX, XLS, Parquet, TSV, JSONL)",
  "Interactive Issue Ledger with 1-click quality remediation",
  "A-F Column Confidence Scoring System",
  "Multi-dataset relational joins (up to 5 datasets on primary keys)",
  "Ephemeral RAG AI Companion for interactive dataset Q&A",
  "WeasyPrint board-ready PDF report downloads",
  "Multi-format streaming export (CSV, Parquet, HTML)",
  "100% ephemeral in-memory execution (zero permanent disk storage)",
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
          <div className="container mx-auto px-4 text-center space-y-4 max-w-3xl">
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-emerald-500/10 text-emerald-700 text-xs font-semibold uppercase tracking-wider font-mono border border-emerald-500/20">
              <Heart className="h-3.5 w-3.5 fill-emerald-500" />
              <span>100% Free & Open Source</span>
            </div>

            <h1 className="text-4xl sm:text-5xl md:text-6xl font-display font-extrabold text-foreground tracking-tight uppercase leading-[1.05]">
              GetReport Free Edition.
            </h1>
            <p className="text-base sm:text-lg text-muted-foreground max-w-2xl mx-auto leading-relaxed">
              No subscriptions, no hidden fees, and no feature locks. Complete automated data quality audits, multi-dataset joins, and executive PDF reports for everyone.
            </p>
          </div>
        </div>

        {/* Single Focused Offering Card */}
        <div className="container mx-auto px-4 py-16 max-w-3xl">
          <Card className="border-2 border-primary/30 bg-card rounded-3xl p-8 sm:p-12 shadow-premium relative overflow-hidden space-y-8">
            
            {/* Card Header & Price */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-6 border-b border-border/60 pb-8">
              <div className="space-y-2">
                <Badge className="bg-primary text-primary-foreground text-[10px] font-mono font-bold uppercase tracking-wider px-3 py-1 rounded-md whitespace-nowrap w-fit">
                  FULL SUITE INCLUDED
                </Badge>
                <h2 className="text-2xl sm:text-3xl font-display font-extrabold text-foreground uppercase tracking-tight">
                  GetReport Full Access
                </h2>
                <p className="text-xs sm:text-sm text-muted-foreground">
                  Instant in-memory dataset audits with zero registration required.
                </p>
              </div>

              <div className="flex flex-col items-start sm:items-end shrink-0">
                <div className="flex items-baseline gap-1">
                  <span className="text-5xl font-display font-extrabold text-foreground">$0</span>
                  <span className="text-xs text-emerald-700 font-mono font-bold">100% FREE</span>
                </div>
                <span className="text-[11px] text-muted-foreground font-mono">No credit card required</span>
              </div>
            </div>

            {/* Included Features List */}
            <div className="space-y-4">
              <span className="text-xs font-mono uppercase tracking-wider text-primary font-bold block">
                Everything Included In Current Version:
              </span>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs sm:text-sm text-foreground font-sans">
                {currentOfferingFeatures.map((feature, idx) => (
                  <div key={idx} className="flex items-start gap-2.5 p-2.5 rounded-xl bg-muted/40 border border-border/40">
                    <Check className="h-4 w-4 text-emerald-600 shrink-0 mt-0.5" />
                    <span className="leading-snug">{feature}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Action CTA */}
            <div className="pt-4 border-t border-border/60 flex flex-col sm:flex-row items-center justify-between gap-4">
              <div className="flex items-center gap-2 text-xs text-muted-foreground font-mono">
                <Shield className="h-4 w-4 text-emerald-600" />
                <span>Session-scoped processing</span>
              </div>

              <Link to="/workspace" className="w-full sm:w-auto">
                <Button size="lg" className="w-full sm:w-auto h-12 px-8 rounded-xl font-display font-semibold text-sm shadow-premium">
                  <span>Start Free Audit Now</span>
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
            </div>

          </Card>
        </div>

        {/* FAQ Section */}
        <div className="border-t border-border/60 bg-muted/20 py-16 md:py-24">
          <div className="container mx-auto px-4 max-w-3xl">
            <div className="text-center mb-12 space-y-2">
              <h2 className="text-2xl sm:text-3xl font-display font-bold text-foreground uppercase tracking-tight">Frequently Asked Questions</h2>
              <p className="text-xs sm:text-sm text-muted-foreground">Everything you need to know about GetReport's free platform.</p>
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
