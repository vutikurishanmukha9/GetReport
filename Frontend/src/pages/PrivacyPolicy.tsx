import { Lock, Trash2, EyeOff, Server, ShieldCheck } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";

export const PrivacyPolicy = () => {
  return (
    <div className="min-h-screen flex flex-col bg-background animate-in fade-in duration-500">
      <Header onReset={() => {}} showReset={false} />

      <main className="flex-1 pt-16 sm:pt-20">
        {/* Header */}
        <div className="border-b border-border/60 bg-gradient-to-b from-muted/20 via-background to-background py-8 sm:py-12">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-5xl text-center space-y-3 sm:space-y-4">
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-emerald-500/10 text-emerald-700 text-xs font-semibold uppercase tracking-wider font-mono border border-emerald-500/20 t-badge-shimmer">
              <ShieldCheck className="h-3.5 w-3.5" />
              <span>Zero Retention & Privacy Charter</span>
            </div>

            <h1 className="text-3xl sm:text-4xl md:text-5xl font-display font-extrabold text-foreground tracking-tight uppercase leading-[1.08]">
              Privacy & In-Memory Guarantees.
            </h1>
            
            <p className="text-sm sm:text-base text-muted-foreground max-w-2xl mx-auto leading-relaxed font-sans">
              Your raw tabular datasets are executed strictly in isolated ephemeral memory. We never sell, retain on disk, or train AI models on your proprietary records.
            </p>
          </div>
        </div>

        {/* Section: 4 Key Security Pillars */}
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-10 max-w-5xl space-y-6 sm:space-y-8">
          
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card className="p-4 sm:p-5 border border-border bg-card shadow-premium space-y-2 rounded-2xl t-card-lift">
              <div className="h-9 w-9 rounded-xl bg-emerald-500/10 text-emerald-700 flex items-center justify-center">
                <Lock className="h-4 w-4" />
              </div>
              <h3 className="font-display font-bold text-foreground text-xs sm:text-sm">100% In-Memory RAM</h3>
              <p className="text-[11px] text-muted-foreground font-sans leading-relaxed">
                Dataframes load directly into RAM for Polars Rust execution. Zero permanent database disk writes occur for raw row records.
              </p>
            </Card>

            <Card className="p-4 sm:p-5 border border-border bg-card shadow-premium space-y-2 rounded-2xl t-card-lift">
              <div className="h-9 w-9 rounded-xl bg-primary/10 text-primary flex items-center justify-center">
                <Trash2 className="h-4 w-4" />
              </div>
              <h3 className="font-display font-bold text-foreground text-xs sm:text-sm">60-Minute Auto Purge</h3>
              <p className="text-[11px] text-muted-foreground font-sans leading-relaxed">
                All uploaded files, intermediate buffers, and generated PDF reports are automatically wiped from memory within 60 minutes of inactivity.
              </p>
            </Card>

            <Card className="p-4 sm:p-5 border border-border bg-card shadow-premium space-y-2 rounded-2xl t-card-lift">
              <div className="h-9 w-9 rounded-xl bg-blue-500/10 text-blue-700 flex items-center justify-center">
                <EyeOff className="h-4 w-4" />
              </div>
              <h3 className="font-display font-bold text-foreground text-xs sm:text-sm">Zero Model Training</h3>
              <p className="text-[11px] text-muted-foreground font-sans leading-relaxed">
                We never use your raw datasets to train public or private LLMs. Only aggregated mathematical summaries are analyzed.
              </p>
            </Card>

            <Card className="p-4 sm:p-5 border border-border bg-card shadow-premium space-y-2 rounded-2xl t-card-lift">
              <div className="h-9 w-9 rounded-xl bg-purple-500/10 text-purple-700 flex items-center justify-center">
                <Server className="h-4 w-4" />
              </div>
              <h3 className="font-display font-bold text-foreground text-xs sm:text-sm">TLS 1.3 Encryption</h3>
              <p className="text-[11px] text-muted-foreground font-sans leading-relaxed">
                All data in transit is encrypted using modern TLS 1.3 standards with HSTS enforcement and strict Content Security Policies.
              </p>
            </Card>
          </div>

          {/* Detailed Legal Breakdown */}
          <div className="border border-border/80 bg-card rounded-2xl sm:rounded-3xl p-5 sm:p-8 shadow-premium space-y-6 text-xs sm:text-sm text-muted-foreground leading-relaxed font-sans">
            
            <section className="space-y-2">
              <div className="flex items-center gap-2 border-b border-border/60 pb-2">
                <span className="font-mono font-bold text-primary text-xs">SECTION 01</span>
                <h2 className="text-base sm:text-lg font-display font-bold text-foreground uppercase tracking-tight">
                  Information Collection & Data Lifecycle
                </h2>
              </div>
              <p>
                GetReport collects minimal metadata strictly required to perform automated data quality scoring and report generation: temporary session identifiers (<code>task_id</code>), user-configured Issue Ledger approvals, and dataset shape metadata (row count, column names, inferred data types). Raw records are kept transiently in server memory solely during the active processing lifecycle.
              </p>
              <div className="p-3 bg-muted/40 rounded-xl border border-border/60 font-mono text-xs text-foreground">
                <strong>Key Takeaway:</strong> Raw CSV or Excel rows are never persisted to disk, relational databases, or external storage buckets.
              </div>
            </section>

            <section className="space-y-2">
              <div className="flex items-center gap-2 border-b border-border/60 pb-2">
                <span className="font-mono font-bold text-primary text-xs">SECTION 02</span>
                <h2 className="text-base sm:text-lg font-display font-bold text-foreground uppercase tracking-tight">
                  Third-Party Subprocessors & LLM Safety Boundary
                </h2>
              </div>
              <p>
                For executive insight generation and natural language Q&A, GetReport sends strictly aggregated, non-identifying statistical metrics (e.g. column mean, standard deviation, domain taxonomy, anomaly count) to provider endpoints (such as Google Gemini, OpenAI, or OpenRouter). Individual customer, patient, or employee rows are programmatically stripped from prompt payloads prior to dispatch.
              </p>
            </section>

            <section className="space-y-2">
              <div className="flex items-center gap-2 border-b border-border/60 pb-2">
                <span className="font-mono font-bold text-primary text-xs">SECTION 03</span>
                <h2 className="text-base sm:text-lg font-display font-bold text-foreground uppercase tracking-tight">
                  User Ownership & Immediate Memory Deletion
                </h2>
              </div>
              <p>
                Users retain 100% intellectual property and commercial ownership over all uploaded data and compiled report artifacts. You may trigger an immediate purge of your active workspace session at any time by clicking the &quot;Reset Workspace&quot; button in the application header.
              </p>
            </section>

          </div>

        </div>

      </main>

      <Footer />
    </div>
  );
};

export default PrivacyPolicy;
