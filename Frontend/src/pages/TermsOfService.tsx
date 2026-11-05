import { FileText, Shield, Scale, AlertCircle, CheckCircle2, ShieldCheck } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";

export const TermsOfService = () => {
  return (
    <div className="min-h-screen flex flex-col bg-[#07080a] text-zinc-100 animate-in fade-in duration-500 relative">
      {/* Technical Background Grid & Film Grain */}
      <div className="absolute inset-0 bg-[radial-gradient(rgba(255,255,255,0.06)_1px,transparent_1px)] [background-size:24px_24px] pointer-events-none opacity-40" />
      <div className="hero-grain-overlay" />

      <Header onReset={() => {}} showReset={false} />

      <main className="flex-1 pt-24 sm:pt-28 relative z-10">
        {/* Header */}
        <div className="border-b border-white/[0.08] bg-[#07080a]/90 py-8 sm:py-14 relative z-10">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-5xl text-center space-y-3 sm:space-y-4">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-white/10 bg-white/[0.04] text-zinc-300 text-xs font-mono">
              <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
              <span>Apache 2.0 Open Source Platform Terms</span>
            </div>

            <h1 className="text-3xl sm:text-4xl md:text-5xl font-display font-black text-white tracking-[-0.03em] leading-[1.12]">
              Terms of service.
            </h1>
            
            <p className="text-sm sm:text-base text-zinc-400 max-w-2xl mx-auto leading-relaxed font-sans">
              Clear guidelines and operating terms governing your access to and use of GetReport&apos;s automated data intelligence platform.
            </p>
          </div>
        </div>

        {/* Content */}
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 max-w-4xl space-y-6">
          
          <div className="space-y-5 text-xs sm:text-sm text-zinc-300 leading-relaxed font-sans">
            
            {/* Section 1: Acceptance & Open Source */}
            <Card className="p-5 sm:p-7 border border-white/10 bg-[#0c0d12]/80 backdrop-blur-md shadow-sm space-y-3 rounded-2xl">
              <div className="flex items-center gap-3 border-b border-white/10 pb-3">
                <div className="p-2 bg-white/[0.06] text-white rounded-xl border border-white/10">
                  <Scale className="h-4 w-4" />
                </div>
                <div>
                  <span className="font-mono text-[11px] text-zinc-400 font-bold block">CLAUSE 01</span>
                  <h2 className="text-base sm:text-lg font-display font-bold text-white tracking-tight">
                    1. Acceptance of Terms & Open Source Framework
                  </h2>
                </div>
              </div>
              <p className="text-zinc-400 leading-relaxed">
                By accessing, uploading datasets to, or self-hosting GetReport, you agree to be bound by these Terms of Service and the accompanying Apache 2.0 open source license. If you do not agree with any portion of these terms, you must refrain from using the service.
              </p>
              <div className="p-3 bg-emerald-500/10 rounded-xl border border-emerald-500/20 font-mono text-xs text-emerald-300 flex items-center gap-2.5">
                <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0" />
                <span>GetReport is completely free to use for both personal and commercial data audits.</span>
              </div>
            </Card>

            {/* Section 2: Acceptable Use Policy */}
            <Card className="p-5 sm:p-7 border border-white/10 bg-[#0c0d12]/80 backdrop-blur-md shadow-sm space-y-3 rounded-2xl">
              <div className="flex items-center gap-3 border-b border-white/10 pb-3">
                <div className="p-2 bg-rose-500/10 text-rose-400 rounded-xl border border-rose-500/20">
                  <FileText className="h-4 w-4" />
                </div>
                <div>
                  <span className="font-mono text-[11px] text-rose-400 font-bold block">CLAUSE 02</span>
                  <h2 className="text-base sm:text-lg font-display font-bold text-white tracking-tight">
                    2. Acceptable Use & Payload Safety
                  </h2>
                </div>
              </div>
              <p className="text-zinc-400 leading-relaxed">
                You agree not to upload malicious binaries, weaponized spreadsheets containing macro exploits, or decompression zip-bomb archives intended to exhaust server memory. GetReport reserves the right to rate-limit, terminate, or drop client connections exhibiting automated scanning or denial-of-service behaviors.
              </p>
            </Card>

            {/* Section 3: Data Ownership & Intellectual Property */}
            <Card className="p-5 sm:p-7 border border-white/10 bg-[#0c0d12]/80 backdrop-blur-md shadow-sm space-y-3 rounded-2xl">
              <div className="flex items-center gap-3 border-b border-white/10 pb-3">
                <div className="p-2 bg-emerald-500/10 text-emerald-400 rounded-xl border border-emerald-500/20">
                  <Shield className="h-4 w-4" />
                </div>
                <div>
                  <span className="font-mono text-[11px] text-emerald-400 font-bold block">CLAUSE 03</span>
                  <h2 className="text-base sm:text-lg font-display font-bold text-white tracking-tight">
                    3. Complete Data Ownership & Commercial Rights
                  </h2>
                </div>
              </div>
              <p className="text-zinc-400 leading-relaxed">
                You retain 100% intellectual property, commercial licensing, and ownership rights over all uploaded tabular datasets, remediated output files (CSV, Parquet), and compiled PDF reports. GetReport claims zero ownership over your proprietary data.
              </p>
            </Card>

            {/* Section 4: Limitation of Liability */}
            <Card className="p-5 sm:p-7 border border-white/10 bg-[#0c0d12]/80 backdrop-blur-md shadow-sm space-y-3 rounded-2xl">
              <div className="flex items-center gap-3 border-b border-white/10 pb-3">
                <div className="p-2 bg-amber-500/10 text-amber-400 rounded-xl border border-amber-500/20">
                  <AlertCircle className="h-4 w-4" />
                </div>
                <div>
                  <span className="font-mono text-[11px] text-amber-400 font-bold block">CLAUSE 04</span>
                  <h2 className="text-base sm:text-lg font-display font-bold text-white tracking-tight">
                    4. Limitation of Liability & Audit Disclaimer
                  </h2>
                </div>
              </div>
              <p className="text-zinc-400 leading-relaxed">
                GetReport is provided &ldquo;as is&rdquo; without warranty of any kind. While automated hygiene algorithms and confidence scores are engineered with high statistical rigor, audit reports should not replace formal legal, regulatory compliance, or financial audits.
              </p>
            </Card>

          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default TermsOfService;
