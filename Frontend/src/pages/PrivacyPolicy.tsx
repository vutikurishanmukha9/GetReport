import { ShieldCheck, Lock, Trash2, EyeOff, Server } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";

export const PrivacyPolicy = () => {
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
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-border bg-muted/50 text-muted-foreground text-xs font-mono">
              <ShieldCheck className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />
              <span>Zero Cloud Persistence Policy</span>
            </div>

            <h1 className="text-3xl sm:text-4xl md:text-5xl font-display font-black text-foreground tracking-[-0.03em] leading-[1.12]">
              Security & privacy architecture.
            </h1>
            
            <p className="text-sm sm:text-base text-muted-foreground max-w-2xl mx-auto leading-relaxed font-sans">
              Learn how GetReport protects your proprietary spreadsheets, enforces in-memory execution boundaries, and maintains zero permanent records of your data.
            </p>
          </div>
        </div>

        {/* Content Container */}
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 max-w-4xl space-y-8">
          
          {/* Top Security Highlights Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card className="p-4 sm:p-5 border border-border dark:border-white/10 bg-card dark:bg-[#0c0d12]/80 backdrop-blur-md shadow-2xs space-y-2.5 rounded-2xl text-card-foreground">
              <div className="h-9 w-9 rounded-xl bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 flex items-center justify-center border border-emerald-500/20">
                <Lock className="h-4 w-4" />
              </div>
              <h3 className="font-display font-bold text-foreground text-xs sm:text-sm">100% In-Memory RAM</h3>
              <p className="text-[11px] text-muted-foreground font-sans leading-relaxed">
                Dataframes load directly into RAM for Polars execution. Zero permanent database disk writes occur for raw row records.
              </p>
            </Card>

            <Card className="p-4 sm:p-5 border border-border dark:border-white/10 bg-card dark:bg-[#0c0d12]/80 backdrop-blur-md shadow-2xs space-y-2.5 rounded-2xl text-card-foreground">
              <div className="h-9 w-9 rounded-xl bg-muted dark:bg-white/[0.06] text-foreground dark:text-white flex items-center justify-center border border-border dark:border-white/10">
                <Trash2 className="h-4 w-4" />
              </div>
              <h3 className="font-display font-bold text-foreground text-xs sm:text-sm">60-Minute Auto Purge</h3>
              <p className="text-[11px] text-muted-foreground font-sans leading-relaxed">
                All uploaded files, intermediate buffers, and generated PDF reports are automatically wiped from memory within 60 minutes of inactivity.
              </p>
            </Card>

            <Card className="p-4 sm:p-5 border border-border dark:border-white/10 bg-card dark:bg-[#0c0d12]/80 backdrop-blur-md shadow-2xs space-y-2.5 rounded-2xl text-card-foreground">
              <div className="h-9 w-9 rounded-xl bg-amber-500/10 text-amber-600 dark:text-amber-400 flex items-center justify-center border border-amber-500/20">
                <EyeOff className="h-4 w-4" />
              </div>
              <h3 className="font-display font-bold text-foreground text-xs sm:text-sm">Zero Model Training</h3>
              <p className="text-[11px] text-muted-foreground font-sans leading-relaxed">
                We never use your raw datasets to train public or private LLMs. Only aggregated mathematical summaries are analyzed.
              </p>
            </Card>

            <Card className="p-4 sm:p-5 border border-border dark:border-white/10 bg-card dark:bg-[#0c0d12]/80 backdrop-blur-md shadow-2xs space-y-2.5 rounded-2xl text-card-foreground">
              <div className="h-9 w-9 rounded-xl bg-sky-500/10 text-sky-600 dark:text-sky-400 flex items-center justify-center border border-sky-500/20">
                <Server className="h-4 w-4" />
              </div>
              <h3 className="font-display font-bold text-foreground text-xs sm:text-sm">Local Docker Support</h3>
              <p className="text-[11px] text-muted-foreground font-sans leading-relaxed">
                Need air-gapped security? Clone the repository and run GetReport inside your private VPC or local workstation.
              </p>
            </Card>
          </div>

          {/* Legal Prose Content */}
          <div className="border border-border dark:border-white/10 bg-card dark:bg-[#0c0d12]/70 rounded-2xl p-6 sm:p-8 space-y-6 text-xs sm:text-sm text-foreground/90 font-sans leading-relaxed shadow-2xs">
            <section className="space-y-2">
              <h2 className="text-base font-display font-bold text-foreground">1. Scope and Architecture</h2>
              <p className="text-muted-foreground">
                GetReport is designed from first principles as an ephemeral data intelligence engine. When you upload a dataset (CSV, Excel, Parquet), the binary payload is ingested into memory ChunkedArrays via Polars. No raw rows or cells are written to permanent database disks or third-party storage buckets.
              </p>
            </section>

            <section className="space-y-2">
              <h2 className="text-base font-display font-bold text-foreground">2. Ephemeral RAG Companion AI Inferences</h2>
              <p className="text-muted-foreground">
                When using the interactive AI companion, prompt contexts are constructed strictly from high-level mathematical aggregations (column names, inferred dtypes, null frequencies, outlier z-scores, and correlation coefficients). Raw table rows and personally identifiable information (PII) are scrubbed before prompt dispatch.
              </p>
            </section>

            <section className="space-y-2">
              <h2 className="text-base font-display font-bold text-foreground">3. Cryptographic Session Lifetime</h2>
              <p className="text-muted-foreground">
                Each ingestion workflow creates an ephemeral Task ID. Once you download your audit report or click &ldquo;Start Over&rdquo;, session buffers are marked for immediate garbage collection. All unreferenced memory allocations are automatically purged within 60 minutes.
              </p>
            </section>

            <section className="space-y-2">
              <h2 className="text-base font-display font-bold text-foreground">4. Snowflake Warehouse Telemetry</h2>
              <p className="text-muted-foreground">
                Warehouse credentials supplied for Snowflake synchronization are encrypted in memory using AES-256-GCM. Credentials are never written to disk, never logged to server consoles, and are immediately discarded upon query execution completion.
              </p>
            </section>

            <section className="space-y-2">
              <h2 className="text-base font-display font-bold text-foreground">5. Self-Hosting and Open Source Auditability</h2>
              <p className="text-muted-foreground">
                Because GetReport is fully open source under the Apache 2.0 license, enterprise engineering teams can inspect every line of the Polars ingestion pipeline and run the entire stack in an isolated, air-gapped on-premises environment.
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
