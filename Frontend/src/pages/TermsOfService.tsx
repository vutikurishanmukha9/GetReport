import { FileText, Shield, Scale, AlertCircle, CheckCircle2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";

export const TermsOfService = () => {
  return (
    <div className="min-h-screen flex flex-col bg-background animate-in fade-in duration-500">
      <Header onReset={() => {}} showReset={false} />

      <main className="flex-1 pt-20">
        {/* Header */}
        <div className="border-b border-border/60 bg-gradient-to-b from-muted/30 to-background py-16 md:py-24">
          <div className="container mx-auto px-4 text-center space-y-4 max-w-4xl">
            <Badge variant="outline" className="font-mono text-xs uppercase tracking-wider text-primary border-primary/30 px-3 py-1">
              Legal Framework & Operating Agreement
            </Badge>
            <h1 className="text-4xl sm:text-5xl md:text-6xl font-display font-extrabold text-foreground tracking-tight uppercase leading-[1.05]">
              Terms of Service.
            </h1>
            <p className="text-base sm:text-lg text-muted-foreground max-w-2xl mx-auto leading-relaxed">
              Guidelines and terms governing your access to and use of GetReport's automated data intelligence platform.
            </p>
          </div>
        </div>

        {/* Content */}
        <div className="container mx-auto px-4 py-16 max-w-4xl space-y-12">
          
          <div className="space-y-8 text-xs sm:text-sm text-muted-foreground leading-relaxed">
            
            <Card className="p-6 border border-border bg-card shadow-sm space-y-3 rounded-2xl">
              <h2 className="text-lg font-display font-bold text-foreground uppercase tracking-tight flex items-center gap-2">
                <Scale className="h-5 w-5 text-primary" />
                <span>1. Acceptance of Terms</span>
              </h2>
              <p>
                By accessing or using GetReport, you agree to be bound by these Terms of Service. If you do not agree with any portion of these terms, you must refrain from uploading datasets or utilizing our automated reporting engine.
              </p>
            </Card>

            <Card className="p-6 border border-border bg-card shadow-sm space-y-3 rounded-2xl">
              <h2 className="text-lg font-display font-bold text-foreground uppercase tracking-tight flex items-center gap-2">
                <FileText className="h-5 w-5 text-primary" />
                <span>2. Acceptable Use Policy</span>
              </h2>
              <p>
                You agree not to upload malicious binaries, weaponized spreadsheets, or zip-bomb archives intended to disrupt server operations. GetReport reserves the right to rate-limit or drop connections exhibiting automated scanning or denial-of-service behaviors.
              </p>
            </Card>

            <Card className="p-6 border border-border bg-card shadow-sm space-y-3 rounded-2xl">
              <h2 className="text-lg font-display font-bold text-foreground uppercase tracking-tight flex items-center gap-2">
                <Shield className="h-5 w-5 text-emerald-600" />
                <span>3. Data Ownership & Intellectual Property</span>
              </h2>
              <p>
                You retain complete ownership of all uploaded datasets, remediated output files, and generated PDF reports. GetReport claims zero ownership or licensing rights over your proprietary input data or synthesized audit artifacts.
              </p>
            </Card>

            <Card className="p-6 border border-border bg-card shadow-sm space-y-3 rounded-2xl">
              <h2 className="text-lg font-display font-bold text-foreground uppercase tracking-tight flex items-center gap-2">
                <AlertCircle className="h-5 w-5 text-amber-600" />
                <span>4. Limitation of Liability</span>
              </h2>
              <p>
                GetReport delivers data quality audits and recommendations on an "AS IS" basis. While our algorithms provide rigorous statistical scoring and Polars execution, final business decisions based on generated audit reports remain the user's responsibility.
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
