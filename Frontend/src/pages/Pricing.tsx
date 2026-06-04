import { Check, Zap, HelpCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Link } from "react-router-dom";
import { Badge } from "@/components/ui/badge";

export const Pricing = () => {
  return (
    <div className="min-h-screen bg-background animate-in fade-in duration-500 pb-20">
      
      {/* Title Header */}
      <div className="border-b border-border/60 bg-background py-20">
        <div className="container mx-auto px-4 text-center space-y-4">
          <Badge variant="outline" className="font-mono text-xs uppercase tracking-wider text-primary">
            Early Adopter Program
          </Badge>
          <h1 className="text-4xl sm:text-5xl font-display font-extrabold text-foreground tracking-tight uppercase">
            Simple, Transparent Value.
          </h1>
          <p className="text-sm sm:text-base text-muted-foreground max-w-xl mx-auto leading-relaxed">
            Gain full access to our Polars engine, Why-I-Did-X transparency logs, and high-DPI reports.
          </p>
        </div>
      </div>

      <div className="container mx-auto px-4 py-16">
        
        {/* Spotlight Card */}
        <div className="max-w-xl mx-auto">
          <Card className="border border-primary/30 bg-card shadow-premium relative overflow-hidden rounded-2xl flex flex-col justify-between p-8">
            <div className="absolute top-0 right-0 p-4">
              <Badge className="bg-primary text-primary-foreground font-mono text-[9px] tracking-widest font-bold uppercase rounded-md shadow-sm">
                Early Access
              </Badge>
            </div>

            <CardHeader className="text-center pb-6 pt-4 space-y-2">
              <CardTitle className="text-2xl font-display font-black text-foreground uppercase tracking-tight">Early Access Tier</CardTitle>
              <CardDescription className="text-xs max-w-xs mx-auto">Everything you need to audit, clean, and analyze datasets</CardDescription>
              <div className="pt-4 flex items-baseline justify-center gap-1">
                <span className="text-5xl font-display font-black text-foreground tracking-tighter">$0</span>
                <span className="text-xs font-mono text-muted-foreground uppercase tracking-wider">/ month</span>
              </div>
            </CardHeader>

            <CardContent className="space-y-6">
              <div className="bg-primary/5 border border-primary/20 p-3.5 rounded-xl text-center">
                <p className="text-xs font-semibold text-primary flex items-center justify-center gap-2">
                  <Zap className="h-3.5 w-3.5 fill-primary" />
                  Currently, we are offering all premium features completely FREE.
                </p>
              </div>

              <div className="h-[1px] w-full bg-border/60" />

              <ul className="space-y-3.5 text-xs sm:text-sm">
                {[
                  "Unlimited Dataset Uploads (CSV/XLSX)",
                  "Column Confidence Scores (A-F Grades)",
                  "“Why I Did X” Decision Transparency logs",
                  "Semantic Domain Detection & mapping",
                  "Pearson correlations & VIF multicollinearity",
                  "AI-Powered RAG Insights & Chat companion",
                  "PDF Report Compilation (15+ Sections)",
                  "Polars-powered cleanup overrides",
                  "Matplotlib high-DPI visual charts",
                ].map((feature) => (
                  <li key={feature} className="flex items-start gap-3 text-muted-foreground font-sans">
                    <div className="h-4.5 w-4.5 rounded-full bg-emerald-500/10 text-emerald-600 flex items-center justify-center shrink-0 mt-0.5 text-[10px] font-bold">
                      ✓
                    </div>
                    <span>{feature}</span>
                  </li>
                ))}
              </ul>
            </CardContent>

            <CardFooter className="pt-6">
              <Link to="/workspace" className="w-full">
                <Button className="w-full h-11 text-sm font-semibold shadow-premium rounded-xl hover:-translate-y-0.5 active:scale-95 transition-all">
                  Get Started For Free
                </Button>
              </Link>
            </CardFooter>
          </Card>
        </div>

        {/* FAQs */}
        <div className="mt-24 max-w-2xl mx-auto space-y-8">
          <div className="text-center space-y-2">
            <h2 className="text-xl sm:text-2xl font-display font-bold text-foreground">Frequently Asked Questions</h2>
            <p className="text-xs text-muted-foreground">Clear explanations for pricing policies and data guidelines.</p>
          </div>

          <div className="grid gap-4">
            <Card className="border border-border bg-card shadow-premium rounded-xl p-5 space-y-2">
              <h3 className="font-display font-bold text-sm sm:text-base text-foreground flex items-center gap-2">
                <HelpCircle className="h-4 w-4 text-primary shrink-0" />
                <span>Is it really free?</span>
              </h3>
              <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed pl-6">
                Yes. During our Early Access release, we are opening all diagnostic, clean-up, and report features to everyone for free. We appreciate design and feature feedback.
              </p>
            </Card>

            <Card className="border border-border bg-card shadow-premium rounded-xl p-5 space-y-2">
              <h3 className="font-display font-bold text-sm sm:text-base text-foreground flex items-center gap-2">
                <HelpCircle className="h-4 w-4 text-primary shrink-0" />
                <span>What happens to my uploaded data?</span>
              </h3>
              <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed pl-6">
                We enforce ephemeral memory processing. Your uploaded dataset is loaded in volatile RAM and is completely wiped immediately after report compilation. We do not persist raw values or train models on user inputs.
              </p>
            </Card>

            <Card className="border border-border bg-card shadow-premium rounded-xl p-5 space-y-2">
              <h3 className="font-display font-bold text-sm sm:text-base text-foreground flex items-center gap-2">
                <HelpCircle className="h-4 w-4 text-primary shrink-0" />
                <span>Are there file limit caps?</span>
              </h3>
              <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed pl-6">
                We support files up to 50MB during Early Access. If you require processing larger enterprise records or custom DB adapters, reach out to our team.
              </p>
            </Card>
          </div>
        </div>

        {/* Footnote */}
        <div className="mt-16 text-center space-y-4">
          <p className="text-xs text-muted-foreground">
            No credit card registration required. Ephermeral calculations. Free forever.
          </p>
          <Link to="/workspace">
            <Button size="sm" variant="outline" className="rounded-xl border-border bg-card hover:bg-muted/10 transition-all hover:-translate-y-0.5">
              Start Audit Now
            </Button>
          </Link>
        </div>

      </div>
    </div>
  );
};

export default Pricing;
