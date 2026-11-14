import { Lock, Trash2, EyeOff, Server, ShieldCheck } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";

export const PrivacyPolicy = () => {
  return (
    <div className="min-h-screen flex flex-col bg-[#F5F2EB] text-stone-900 dark:bg-[#07080a] dark:text-zinc-100 animate-in fade-in duration-500 relative">
      {/* Technical Background Grid & Film Grain */}
      <div className="absolute inset-0 bg-[radial-gradient(rgba(0,0,0,0.05)_1px,transparent_1px)] dark:bg-[radial-gradient(rgba(255,255,255,0.06)_1px,transparent_1px)] [background-size:24px_24px] pointer-events-none opacity-40" />
      <div className="hero-grain-overlay" />

      <Header onReset={() => { }} showReset={false} />

      <main className="flex-1 pt-24 sm:pt-28 relative z-10">
        {/* Header */}
        <div className="border-b border-stone-300 bg-[#FAF8F5]/90 dark:border-white/[0.08] dark:bg-[#07080a]/90 py-8 sm:py-14 relative z-10">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-5xl text-center space-y-3 sm:space-y-4">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-stone-300 bg-[#FAF8F5] text-stone-700 dark:border-white/10 dark:bg-white/[0.04] dark:text-zinc-300 text-xs font-mono shadow-xs">
              <ShieldCheck className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />
              <span>Zero-Persistence Privacy Guarantee</span>
            </div>

            <h1 className="text-3xl sm:text-4xl md:text-5xl font-display font-black text-stone-900 dark:text-white tracking-[-0.03em] leading-[1.12]">
              Privacy &amp; in-memory guarantees.
            </h1>

            <p className="text-sm sm:text-base text-stone-600 dark:text-zinc-400 max-w-2xl mx-auto leading-relaxed font-sans">
              Your raw tabular datasets are executed strictly in isolated ephemeral memory. We never sell, retain on disk, or train AI models on your proprietary records.
            </p>
          </div>
        </div>

        {/* Section: 4 Key Security Pillars */}
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 max-w-5xl space-y-6 sm:space-y-8">

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card className="p-4 sm:p-5 border border-stone-300 bg-[#FAF8F5] dark:border-white/10 dark:bg-[#0c0d12]/80 dark:backdrop-blur-md shadow-xs space-y-2.5 rounded-2xl">
              <div className="h-9 w-9 rounded-xl bg-emerald-100 text-emerald-900 dark:bg-emerald-500/10 dark:text-emerald-400 flex items-center justify-center border border-emerald-300 dark:border-emerald-500/20">
                <Lock className="h-4 w-4" />
              </div>
              <h3 className="font-display font-bold text-stone-900 dark:text-white text-xs sm:text-sm">100% In-Memory RAM</h3>
              <p className="text-[11px] text-stone-600 dark:text-zinc-400 font-sans leading-relaxed">
                Dataframes load directly into RAM for Polars execution. Zero permanent database disk writes occur for raw row records.
              </p>
            </Card>

            <Card className="p-4 sm:p-5 border border-stone-300 bg-[#FAF8F5] dark:border-white/10 dark:bg-[#0c0d12]/80 dark:backdrop-blur-md shadow-xs space-y-2.5 rounded-2xl">
              <div className="h-9 w-9 rounded-xl bg-stone-200 text-stone-900 dark:bg-white/[0.06] dark:text-white flex items-center justify-center border border-stone-300 dark:border-white/10">
                <Trash2 className="h-4 w-4" />
              </div>
              <h3 className="font-display font-bold text-stone-900 dark:text-white text-xs sm:text-sm">60-Minute Auto Purge</h3>
              <p className="text-[11px] text-stone-600 dark:text-zinc-400 font-sans leading-relaxed">
                All uploaded files, intermediate buffers, and generated PDF reports are automatically wiped from memory within 60 minutes of inactivity.
              </p>
            </Card>

            <Card className="p-4 sm:p-5 border border-stone-300 bg-[#FAF8F5] dark:border-white/10 dark:bg-[#0c0d12]/80 dark:backdrop-blur-md shadow-xs space-y-2.5 rounded-2xl">
              <div className="h-9 w-9 rounded-xl bg-amber-100 text-amber-900 dark:bg-amber-500/10 dark:text-amber-400 flex items-center justify-center border border-amber-300 dark:border-amber-500/20">
                <EyeOff className="h-4 w-4" />
              </div>
              <h3 className="font-display font-bold text-stone-900 dark:text-white text-xs sm:text-sm">Zero Model Training</h3>
              <p className="text-[11px] text-stone-600 dark:text-zinc-400 font-sans leading-relaxed">
                We never use your raw datasets to train public or private LLMs. Only aggregated mathematical summaries are analyzed.
              </p>
            </Card>

            <Card className="p-4 sm:p-5 border border-stone-300 bg-[#FAF8F5] dark:border-white/10 dark:bg-[#0c0d12]/80 dark:backdrop-blur-md shadow-xs space-y-2.5 rounded-2xl">
              <div className="h-9 w-9 rounded-xl bg-sky-100 text-sky-900 dark:bg-sky-500/10 dark:text-sky-400 flex items-center justify-center border border-sky-300 dark:border-sky-500/20">
                <Server className="h-4 w-4" />
              </div>
              <h3 className="font-display font-bold text-stone-900 dark:text-white text-xs sm:text-sm">Local Docker Support</h3>
              <p className="text-[11px] text-stone-600 dark:text-zinc-400 font-sans leading-relaxed">
                Need air-gapped security? Clone the repository and run GetReport inside your private VPC or local workstation.
              </p>
            </Card>
          </div>

          {/* Legal Prose Content */}
          <div className="border border-stone-300 bg-[#FAF8F5] rounded-2xl p-6 sm:p-8 space-y-6 text-xs sm:text-sm text-stone-700 dark:text-zinc-300 font-sans leading-relaxed dark:border-white/10 dark:bg-[#0c0d12]/70 shadow-xs">
            <section className="space-y-2">
              <h2 className="text-base font-display font-bold text-stone-900 dark:text-white">1. Scope and Architecture</h2>
              <p>
                GetReport is designed from first principles as an ephemeral data intelligence engine. When you upload a dataset (CSV, Excel, Parquet), the binary payload is ingested into memory ChunkedArrays via Polars. No raw rows or cells are written to permanent database disks or third-party storage buckets.
              </p>
            </section>

            <section className="space-y-2">
              <h2 className="text-base font-display font-bold text-stone-900 dark:text-white">2. Ephemeral RAG Companion AI Inferences</h2>
              <p>
                When using the interactive AI companion, prompt contexts are constructed strictly from high-level mathematical aggregations (column names, inferred dtypes, null frequencies, outlier z-scores, and correlation coefficients). Raw table rows and personally identifiable information (PII) are scrubbed before prompt dispatch.
              </p>
            </section>

            <section className="space-y-2">
              <h2 className="text-base font-display font-bold text-stone-900 dark:text-white">3. Cryptographic Session Lifetime</h2>
              <p>
                Each ingestion workflow creates an ephemeral Task ID. Once you download your audit report or click &ldquo;Start Over&rdquo;, session buffers are marked for immediate garbage collection. All unreferenced memory allocations are automatically purged within 60 minutes.
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
