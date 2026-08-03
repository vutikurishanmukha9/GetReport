import { Shield, Lock, Trash2, EyeOff, Server, FileText, CheckCircle2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";

export const PrivacyPolicy = () => {
  return (
    <div className="min-h-screen flex flex-col bg-background animate-in fade-in duration-500">
      <Header onReset={() => {}} showReset={false} />

      <main className="flex-1 pt-20">
        {/* Header */}
        <div className="border-b border-border/60 bg-gradient-to-b from-muted/30 to-background py-16 md:py-24">
          <div className="container mx-auto px-4 text-center space-y-4 max-w-4xl">
            <Badge variant="outline" className="font-mono text-xs uppercase tracking-wider text-emerald-700 border-emerald-500/30 px-3 py-1 bg-emerald-50">
              Data Privacy & Memory Security Policy
            </Badge>
            <h1 className="text-4xl sm:text-5xl md:text-6xl font-display font-extrabold text-foreground tracking-tight uppercase leading-[1.05]">
              Privacy & Zero Retention Guarantee.
            </h1>
            <p className="text-base sm:text-lg text-muted-foreground max-w-2xl mx-auto leading-relaxed">
              Your raw tabular datasets are executed strictly in isolated ephemeral memory. We never sell, store, or train AI models on your proprietary records.
            </p>
          </div>
        </div>

        {/* Content */}
        <div className="container mx-auto px-4 py-16 max-w-4xl space-y-12">
          
          {/* Key Principles Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="p-6 border border-border bg-card shadow-premium space-y-3 rounded-2xl">
              <div className="h-10 w-10 rounded-xl bg-emerald-500/10 text-emerald-700 flex items-center justify-center">
                <Lock className="h-5 w-5" />
              </div>
              <h3 className="font-display font-bold text-foreground text-base">In-Memory Calculation</h3>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Dataframes are loaded directly into RAM for Polars Rust execution. No persistent database disk writes occur for raw row records.
              </p>
            </Card>

            <Card className="p-6 border border-border bg-card shadow-premium space-y-3 rounded-2xl">
              <div className="h-10 w-10 rounded-xl bg-primary/10 text-primary flex items-center justify-center">
                <Trash2 className="h-5 w-5" />
              </div>
              <h3 className="font-display font-bold text-foreground text-base">Automated Session Purge</h3>
              <p className="text-xs text-muted-foreground leading-relaxed">
                All uploaded files, intermediate buffers, and generated PDF reports are automatically purged from memory within 60 minutes of inactivity.
              </p>
            </Card>

            <Card className="p-6 border border-border bg-card shadow-premium space-y-3 rounded-2xl">
              <div className="h-10 w-10 rounded-xl bg-blue-500/10 text-blue-700 flex items-center justify-center">
                <EyeOff className="h-5 w-5" />
              </div>
              <h3 className="font-display font-bold text-foreground text-base">Zero Model Training</h3>
              <p className="text-xs text-muted-foreground leading-relaxed">
                We never use your raw datasets to train public or private LLMs. Only aggregated statistical summaries are evaluated by our insight engine.
              </p>
            </Card>

            <Card className="p-6 border border-border bg-card shadow-premium space-y-3 rounded-2xl">
              <div className="h-10 w-10 rounded-xl bg-purple-500/10 text-purple-700 flex items-center justify-center">
                <Server className="h-5 w-5" />
              </div>
              <h3 className="font-display font-bold text-foreground text-base">TLS 1.3 Encryption</h3>
              <p className="text-xs text-muted-foreground leading-relaxed">
                All data in transit is encrypted using modern TLS 1.3 standards with HSTS enforcement and strict Content Security Policies.
              </p>
            </Card>
          </div>

          {/* Policy Breakdown */}
          <div className="space-y-8 border-t border-border/60 pt-12 text-xs sm:text-sm text-muted-foreground leading-relaxed">
            
            <section className="space-y-3">
              <h2 className="text-xl font-display font-bold text-foreground uppercase tracking-tight">1. Information Collection & Usage</h2>
              <p>
                GetReport collects minimal metadata required to deliver data auditing services: temporary job identifiers (`task_id`), user-configured issue ledger decisions, and anonymized file metadata (row count, column count, schema headers). Raw row contents are held transiently in server memory solely during the active processing lifecycle.
              </p>
            </section>

            <section className="space-y-3">
              <h2 className="text-xl font-display font-bold text-foreground uppercase tracking-tight">2. Third-Party Subprocessors</h2>
              <p>
                For executive insight generation, GetReport passes non-identifying aggregated statistical metrics (e.g. column mean, standard deviation, domain classification) to provider endpoints (such as OpenRouter or OpenAI). Raw rows or personal identifiable information (PII) are filtered out prior to API payload dispatch.
              </p>
            </section>

            <section className="space-y-3">
              <h2 className="text-xl font-display font-bold text-foreground uppercase tracking-tight">3. User Rights & Data Deletion</h2>
              <p>
                Users maintain 100% ownership of all uploaded datasets and remediated outputs. You may clear your active workspace session at any time by clicking "Reset Workspace", triggering an immediate deletion of in-memory data structures.
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
