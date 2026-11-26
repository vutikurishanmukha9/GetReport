import { FileText, Shield, Scale, AlertCircle, CheckCircle2 } from "lucide-react";
import { Card } from "@/components/ui/card";
import { LegalPageLayout } from "@/components/LegalPageLayout";

export const TermsOfService = () => {
  return (
    <LegalPageLayout
      badgeText="Apache 2.0 Open Source Platform Terms"
      title="Terms of service."
      subtitle="Clear guidelines and operating terms governing your access to and use of GetReport's automated data intelligence platform."
      contentClassName="max-w-4xl space-y-6"
    >
          
          <div className="space-y-5 text-xs sm:text-sm text-stone-700 dark:text-zinc-300 leading-relaxed font-sans">
            
            {/* Section 1: Acceptance & Open Source */}
            <Card className="p-5 sm:p-7 border border-stone-300 bg-[#FAF8F5] dark:border-white/10 dark:bg-[#0c0d12]/80 dark:backdrop-blur-md shadow-xs space-y-3 rounded-2xl">
              <div className="flex items-center gap-3 border-b border-stone-200 dark:border-white/10 pb-3">
                <div className="p-2 bg-stone-200 text-stone-900 rounded-xl border border-stone-300 dark:bg-white/[0.06] dark:text-white dark:border-white/10">
                  <Scale className="h-4 w-4" />
                </div>
                <div>
                  <span className="font-mono text-[11px] text-stone-500 dark:text-zinc-400 font-bold block">CLAUSE 01</span>
                  <h2 className="text-base sm:text-lg font-display font-bold text-stone-900 dark:text-white tracking-tight">
                    1. Acceptance of Terms &amp; Open Source Framework
                  </h2>
                </div>
              </div>
              <p className="text-stone-600 dark:text-zinc-400 leading-relaxed">
                By accessing, uploading datasets to, or self-hosting GetReport, you agree to be bound by these Terms of Service and the accompanying Apache 2.0 open source license. If you do not agree with any portion of these terms, you must refrain from using the service.
              </p>
              <div className="p-3 bg-emerald-50 rounded-xl border border-emerald-300 font-mono text-xs text-emerald-900 dark:bg-emerald-500/10 dark:border-emerald-500/20 dark:text-emerald-300 flex items-center gap-2.5">
                <CheckCircle2 className="h-4 w-4 text-emerald-600 dark:text-emerald-400 shrink-0" />
                <span>GetReport is completely free to use for both personal and commercial data audits.</span>
              </div>
            </Card>

            {/* Section 2: Acceptable Use Policy */}
            <Card className="p-5 sm:p-7 border border-stone-300 bg-[#FAF8F5] dark:border-white/10 dark:bg-[#0c0d12]/80 dark:backdrop-blur-md shadow-xs space-y-3 rounded-2xl">
              <div className="flex items-center gap-3 border-b border-stone-200 dark:border-white/10 pb-3">
                <div className="p-2 bg-rose-100 text-rose-900 rounded-xl border border-rose-300 dark:bg-rose-500/10 dark:text-rose-400 dark:border-rose-500/20">
                  <FileText className="h-4 w-4" />
                </div>
                <div>
                  <span className="font-mono text-[11px] text-rose-800 dark:text-rose-400 font-bold block">CLAUSE 02</span>
                  <h2 className="text-base sm:text-lg font-display font-bold text-stone-900 dark:text-white tracking-tight">
                    2. Acceptable Use &amp; Payload Safety
                  </h2>
                </div>
              </div>
              <p className="text-stone-600 dark:text-zinc-400 leading-relaxed">
                You agree not to upload malicious binaries, weaponized spreadsheets containing macro exploits, or decompression zip-bomb archives intended to exhaust server memory. GetReport reserves the right to rate-limit, terminate, or drop client connections exhibiting automated scanning or denial-of-service behaviors.
              </p>
            </Card>

            {/* Section 3: Data Ownership & Intellectual Property */}
            <Card className="p-5 sm:p-7 border border-stone-300 bg-[#FAF8F5] dark:border-white/10 dark:bg-[#0c0d12]/80 dark:backdrop-blur-md shadow-xs space-y-3 rounded-2xl">
              <div className="flex items-center gap-3 border-b border-stone-200 dark:border-white/10 pb-3">
                <div className="p-2 bg-emerald-100 text-emerald-900 rounded-xl border border-emerald-300 dark:bg-emerald-500/10 dark:text-emerald-400 dark:border-emerald-500/20">
                  <Shield className="h-4 w-4" />
                </div>
                <div>
                  <span className="font-mono text-[11px] text-emerald-800 dark:text-emerald-400 font-bold block">CLAUSE 03</span>
                  <h2 className="text-base sm:text-lg font-display font-bold text-stone-900 dark:text-white tracking-tight">
                    3. Complete Data Ownership &amp; Commercial Rights
                  </h2>
                </div>
              </div>
              <p className="text-stone-600 dark:text-zinc-400 leading-relaxed">
                You retain 100% intellectual property, commercial licensing, and ownership rights over all uploaded tabular datasets, remediated output files (CSV, Parquet), and compiled PDF reports. GetReport claims zero ownership over your proprietary data.
              </p>
            </Card>

            {/* Section 4: Limitation of Liability */}
            <Card className="p-5 sm:p-7 border border-stone-300 bg-[#FAF8F5] dark:border-white/10 dark:bg-[#0c0d12]/80 dark:backdrop-blur-md shadow-xs space-y-3 rounded-2xl">
              <div className="flex items-center gap-3 border-b border-stone-200 dark:border-white/10 pb-3">
                <div className="p-2 bg-amber-100 text-amber-900 rounded-xl border border-amber-300 dark:bg-amber-500/10 dark:text-amber-400 dark:border-amber-500/20">
                  <AlertCircle className="h-4 w-4" />
                </div>
                <div>
                  <span className="font-mono text-[11px] text-amber-800 dark:text-amber-400 font-bold block">CLAUSE 04</span>
                  <h2 className="text-base sm:text-lg font-display font-bold text-stone-900 dark:text-white tracking-tight">
                    4. Limitation of Liability &amp; Audit Disclaimer
                  </h2>
                </div>
              </div>
              <p className="text-stone-600 dark:text-zinc-400 leading-relaxed">
                GetReport is provided &ldquo;as is&rdquo; without warranty of any kind. While automated hygiene algorithms and confidence scores are engineered with high statistical rigor, audit reports should not replace formal legal, regulatory compliance, or financial audits.
              </p>
            </Card>

          </div>
    </LegalPageLayout>
  );
};

export default TermsOfService;
