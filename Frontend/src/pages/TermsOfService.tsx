import { FileText, Shield, Scale, AlertCircle, CheckCircle2 } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";

export const TermsOfService = () => {
  return (
    <div className="min-h-screen flex flex-col bg-background animate-in fade-in duration-500">
      <Header onReset={() => {}} showReset={false} />

      <main className="flex-1 pt-16 sm:pt-20">
        {/* Header */}
        <div className="border-b border-border/60 bg-gradient-to-b from-muted/20 via-background to-background py-8 sm:py-12">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-5xl text-center space-y-3 sm:space-y-4">
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-primary/10 text-primary text-xs font-semibold uppercase tracking-wider font-mono border border-primary/20 t-badge-shimmer">
              <Scale className="h-3.5 w-3.5" />
              <span>Legal Operating Agreement</span>
            </div>

            <h1 className="text-3xl sm:text-4xl md:text-5xl font-display font-extrabold text-foreground tracking-tight uppercase leading-[1.08]">
              Terms of Service.
            </h1>
            
            <p className="text-sm sm:text-base text-muted-foreground max-w-2xl mx-auto leading-relaxed font-sans">
              Clear guidelines and operating terms governing your access to and use of GetReport&apos;s automated data intelligence platform.
            </p>
          </div>
        </div>

        {/* Content */}
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-10 max-w-4xl space-y-5">
          
          <div className="space-y-4 text-xs sm:text-sm text-muted-foreground leading-relaxed font-sans">
            
            {/* Section 1: Acceptance & Open Source */}
            <Card className="p-4 sm:p-6 border border-border bg-card shadow-premium space-y-3 rounded-2xl t-card-lift">
              <div className="flex items-center gap-2.5 border-b border-border/60 pb-2.5">
                <div className="p-2 bg-primary/10 text-primary rounded-xl">
                  <Scale className="h-4 w-4" />
                </div>
                <div>
                  <span className="font-mono text-xs text-primary font-bold">CLAUSE 01</span>
                  <h2 className="text-base sm:text-lg font-display font-bold text-foreground uppercase tracking-tight">
                    1. Acceptance of Terms & Open Source Framework
                  </h2>
                </div>
              </div>
              <p>
                By accessing, uploading datasets to, or self-hosting GetReport, you agree to be bound by these Terms of Service and the accompanying Apache 2.0 open source license. If you do not agree with any portion of these terms, you must refrain from using the service.
              </p>
              <div className="p-2.5 bg-muted/40 rounded-xl border border-border/60 font-mono text-xs text-foreground flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0" />
                <span>GetReport is completely free to use for both personal and commercial data audits.</span>
              </div>
            </Card>

            {/* Section 2: Acceptable Use Policy */}
            <Card className="p-4 sm:p-6 border border-border bg-card shadow-premium space-y-3 rounded-2xl t-card-lift">
              <div className="flex items-center gap-2.5 border-b border-border/60 pb-2.5">
                <div className="p-2 bg-rose-500/10 text-rose-700 rounded-xl">
                  <FileText className="h-4 w-4" />
                </div>
                <div>
                  <span className="font-mono text-xs text-rose-600 font-bold">CLAUSE 02</span>
                  <h2 className="text-base sm:text-lg font-display font-bold text-foreground uppercase tracking-tight">
                    2. Acceptable Use & Payload Safety
                  </h2>
                </div>
              </div>
              <p>
                You agree not to upload malicious binaries, weaponized spreadsheets containing macro exploits, or decompression zip-bomb archives intended to exhaust server memory. GetReport reserves the right to rate-limit, terminate, or drop client connections exhibiting automated scanning or denial-of-service behaviors.
              </p>
            </Card>

            {/* Section 3: Data Ownership & Intellectual Property */}
            <Card className="p-4 sm:p-6 border border-border bg-card shadow-premium space-y-3 rounded-2xl t-card-lift">
              <div className="flex items-center gap-2.5 border-b border-border/60 pb-2.5">
                <div className="p-2 bg-emerald-500/10 text-emerald-700 rounded-xl">
                  <Shield className="h-4 w-4" />
                </div>
                <div>
                  <span className="font-mono text-xs text-emerald-600 font-bold">CLAUSE 03</span>
                  <h2 className="text-base sm:text-lg font-display font-bold text-foreground uppercase tracking-tight">
                    3. Complete Data Ownership & Commercial Rights
                  </h2>
                </div>
              </div>
              <p>
                You retain 100% intellectual property, commercial licensing, and ownership rights over all uploaded tabular datasets, remediated output files (CSV, Parquet), and compiled WeasyPrint PDF reports. GetReport claims zero ownership over your proprietary data.
              </p>
            </Card>

            {/* Section 4: Limitation of Liability */}
            <Card className="p-4 sm:p-6 border border-border bg-card shadow-premium space-y-3 rounded-2xl t-card-lift">
              <div className="flex items-center gap-2.5 border-b border-border/60 pb-2.5">
                <div className="p-2 bg-amber-500/10 text-amber-700 rounded-xl">
                  <AlertCircle className="h-4 w-4" />
                </div>
                <div>
                  <span className="font-mono text-xs text-amber-600 font-bold">CLAUSE 04</span>
                  <h2 className="text-base sm:text-lg font-display font-bold text-foreground uppercase tracking-tight">
                    4. Limitation of Liability & Audit Disclaimer
                  </h2>
                </div>
              </div>
              <p>
                GetReport delivers statistical scoring, data cleaning recommendations, and executive summaries on an &quot;AS IS&quot; basis. While our Polars algorithms provide rigorous mathematical evaluations, final business, financial, and clinical decisions based on generated audit reports remain the sole responsibility of the user.
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
