import React from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import {
  AlertTriangle,
  CheckCircle2,
  HelpCircle,
  XCircle,
  Cpu,
  Layers,
  ShieldCheck,
  Check,
} from "lucide-react";
import { MLReadiness } from "@/types/api";

interface MLReadinessCardProps {
  mlReadiness?: MLReadiness;
}

export const MLReadinessCard: React.FC<MLReadinessCardProps> = ({ mlReadiness }) => {
  if (!mlReadiness) return null;

  const { score, status, reasons, recommendation, column_context } = mlReadiness;
  const roundedScore = Math.max(0, Math.min(100, Math.round(score)));

  // Status-specific themes & telemetry
  const statusTheme = {
    Ready: {
      accentColor: "#10b981",
      barColor: "bg-emerald-500 dark:bg-emerald-400",
      pillClass: "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20",
      dotClass: "bg-emerald-500 dark:bg-emerald-400",
      statusLabel: "PRODUCTION READY",
      icon: <CheckCircle2 className="h-4 w-4 text-emerald-500 dark:text-emerald-400 shrink-0" />,
      subtext: "Feature distributions and cardinality profiles are compliant with scikit-learn, XGBoost, and PyTorch training loops.",
      treeStatus: "100% Compatible",
      treeStatusClass: "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20",
      linearStatus: "Verified",
      linearStatusClass: "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20",
      nnStatus: "Input Ready",
      nnStatusClass: "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20",
    },
    "Needs Cleaning": {
      accentColor: "#f59e0b",
      barColor: "bg-amber-500 dark:bg-amber-400",
      pillClass: "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20",
      dotClass: "bg-amber-500 dark:bg-amber-400",
      statusLabel: "NEEDS PREPROCESSING",
      icon: <AlertTriangle className="h-4 w-4 text-amber-500 dark:text-amber-400 shrink-0" />,
      subtext: "Apply recommended transforms or imputation rules before splitting train/test validation cohorts.",
      treeStatus: "Usable with Imputation",
      treeStatusClass: "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20",
      linearStatus: "Scaling Advised",
      linearStatusClass: "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20",
      nnStatus: "Pre-processing Req.",
      nnStatusClass: "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20",
    },
    "Not Ready": {
      accentColor: "#f43f5e",
      barColor: "bg-rose-500 dark:bg-rose-400",
      pillClass: "bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-500/20",
      dotClass: "bg-rose-500 dark:bg-rose-400",
      statusLabel: "UNFIT FOR MODELING",
      icon: <XCircle className="h-4 w-4 text-rose-500 dark:text-rose-400 shrink-0" />,
      subtext: "High-null or zero-variance columns must be pruned or imputed to prevent gradient breakdown during training.",
      treeStatus: "Pruning Required",
      treeStatusClass: "bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-500/20",
      linearStatus: "Convergence Risk",
      linearStatusClass: "bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-500/20",
      nnStatus: "Pipeline Blocked",
      nnStatusClass: "bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-500/20",
    },
  }[status] || {
    accentColor: "#64748b",
    barColor: "bg-slate-500 dark:bg-slate-400",
    pillClass: "bg-slate-500/10 text-slate-600 dark:text-slate-400 border-slate-500/20",
    dotClass: "bg-slate-500 dark:bg-slate-400",
    statusLabel: "AUDIT PENDING",
    icon: <HelpCircle className="h-4 w-4 text-slate-500 dark:text-slate-400 shrink-0" />,
    subtext: "Upload a complete dataset to compute full machine learning suitability metrics.",
    treeStatus: "Pending",
    treeStatusClass: "bg-muted/40 text-muted-foreground border-border/40",
    linearStatus: "Pending",
    linearStatusClass: "bg-muted/40 text-muted-foreground border-border/40",
    nnStatus: "Pending",
    nnStatusClass: "bg-muted/40 text-muted-foreground border-border/40",
  };

  return (
    <Card className="relative border border-border/80 dark:border-white/10 bg-card dark:bg-[#090a10] shadow-premium rounded-3xl overflow-hidden mt-6 text-foreground">
      {/* Header Section */}
      <CardHeader className="border-b border-border/80 dark:border-white/10 bg-muted/20 dark:bg-white/[0.02] px-6 py-5 relative z-10">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-start sm:items-center gap-3.5">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-primary/10 border border-primary/20 text-primary shadow-2xs">
              <Cpu className="h-5 w-5" />
            </div>
            <div>
              <div className="flex items-center gap-2.5">
                <CardTitle className="text-base sm:text-lg font-display font-bold text-foreground tracking-tight">
                  Machine Learning Readiness Assessment
                </CardTitle>
              </div>
              <p className="text-xs text-muted-foreground font-sans mt-0.5">
                Automated algorithmic audit for predictive modeling suitability & feature pipeline robustness
              </p>
            </div>
          </div>

          {/* Status Indicator Pill */}
          <div className="self-start sm:self-center">
            <span
              className={`inline-flex items-center gap-2 px-3 py-1 rounded-full border text-[11px] font-mono font-semibold tracking-wider shadow-2xs ${statusTheme.pillClass}`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${statusTheme.dotClass}`} />
              {statusTheme.statusLabel}
            </span>
          </div>
        </div>
      </CardHeader>

      <CardContent className="p-6 sm:p-7 space-y-6 relative z-10">
        {/* Top Hero Grid: Precision Telemetry Console & Architecture Advisory */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
          {/* Left: Precision Telemetry & Score Console */}
          <div className="lg:col-span-5 p-5 rounded-2xl bg-muted/20 dark:bg-white/[0.02] border border-border/80 dark:border-white/10 shadow-2xs flex flex-col justify-between space-y-4">
            {/* Console Header */}
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground font-semibold flex items-center gap-1.5">
                <ShieldCheck className="h-3.5 w-3.5 text-primary" />
                Readiness Telemetry
              </span>
              <span className={`text-[10px] font-mono font-semibold px-2 py-0.5 rounded border uppercase tracking-wider ${statusTheme.pillClass}`}>
                {status}
              </span>
            </div>

            {/* Score & Health Metric Block */}
            <div className="space-y-3">
              <div className="flex items-baseline justify-between">
                <div className="flex items-baseline gap-1.5">
                  <span className="text-3xl sm:text-4xl font-mono font-bold tracking-tight text-foreground">
                    {roundedScore}
                  </span>
                  <span className="text-xs font-mono font-semibold text-muted-foreground">/ 100</span>
                </div>
                <div className="text-right">
                  <span className="text-xs font-mono font-bold text-foreground block">
                    {roundedScore >= 85 ? "Optimal Matrix" : roundedScore >= 70 ? "Preprocessing Advised" : "Drift Detected"}
                  </span>
                  <span className="text-[10px] font-mono text-muted-foreground">
                    Algorithm Suitability
                  </span>
                </div>
              </div>

              {/* Segmented 16-Bar Hardware Precision Meter */}
              <div className="space-y-1.5">
                <div className="flex gap-1 h-2 sm:h-2.5">
                  {Array.from({ length: 16 }).map((_, idx) => {
                    const threshold = ((idx + 1) / 16) * 100;
                    const isFilled = roundedScore >= threshold;
                    return (
                      <div
                        key={`telemetry-bar-${idx}`}
                        className={`flex-1 h-full rounded-[2px] transition-all duration-300 ${
                          isFilled
                            ? `${statusTheme.barColor} shadow-[0_0_6px_rgba(16,185,129,0.25)]`
                            : "bg-muted/50 dark:bg-white/[0.06] border border-border/40 dark:border-white/5"
                        }`}
                      />
                    );
                  })}
                </div>
                <div className="flex justify-between text-[9px] font-mono text-muted-foreground/80 px-0.5">
                  <span>0 MIN</span>
                  <span>50 MID</span>
                  <span>100 MAX</span>
                </div>
              </div>
            </div>

            {/* Pipeline Context Footer */}
            <div className="pt-3 border-t border-border/60 dark:border-white/10 flex flex-wrap items-center justify-between gap-2 text-xs font-mono text-muted-foreground">
              <div className="flex items-center gap-1.5 min-w-0 max-w-[260px] truncate">
                <span className="truncate">{column_context}</span>
              </div>
              <div className="flex items-center gap-1.5 shrink-0">
                <span className="text-[10px] px-2 py-0.5 rounded bg-muted/40 dark:bg-white/[0.04] border border-border/40 dark:border-white/10 font-medium">
                  Polars 2.0
                </span>
                <span className="text-[10px] px-2 py-0.5 rounded bg-muted/40 dark:bg-white/[0.04] border border-border/40 dark:border-white/10 font-medium">
                  Zero Null Gates
                </span>
              </div>
            </div>
          </div>

          {/* Right: Expert ML Architect Advisory */}
          <div className="lg:col-span-7 p-5 rounded-2xl bg-muted/20 dark:bg-white/[0.02] border border-border/80 dark:border-white/10 shadow-2xs flex flex-col justify-between">
            <div>
              <div className="flex items-center gap-2 text-xs font-mono font-semibold uppercase tracking-wider text-primary mb-2">
                <Cpu className="h-3.5 w-3.5 text-primary" />
                <span>Algorithmic Architecture Advisory</span>
              </div>
              <p className="text-xs sm:text-sm font-sans font-medium text-foreground leading-relaxed">
                {recommendation}
              </p>
            </div>

            <p className="text-[11px] font-sans text-muted-foreground mt-3 pt-3 border-t border-border/60 dark:border-white/10">
              {statusTheme.subtext}
            </p>
          </div>
        </div>

        {/* Model Architecture Compatibility Matrix */}
        <div className="pt-2">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-xs font-mono uppercase tracking-wider text-muted-foreground font-semibold flex items-center gap-1.5">
              <Layers className="h-3.5 w-3.5 text-primary" />
              <span>Model Architecture Suitability Matrix</span>
            </h4>
            <span className="text-[10px] font-mono text-muted-foreground">Pre-training compatibility</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {/* Tree Models */}
            <div className="p-3.5 rounded-xl bg-muted/20 dark:bg-white/[0.02] border border-border/70 dark:border-white/10 flex flex-col justify-between space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-display font-bold text-foreground">Tree Ensembles</span>
                <span className={`text-[9px] font-mono font-semibold px-2 py-0.5 rounded border uppercase tracking-wider ${statusTheme.treeStatusClass}`}>
                  {statusTheme.treeStatus}
                </span>
              </div>
              <p className="text-[11px] text-muted-foreground leading-snug">
                XGBoost, LightGBM, CatBoost. Highly robust to non-linear splits.
              </p>
            </div>

            {/* Linear Models */}
            <div className="p-3.5 rounded-xl bg-muted/20 dark:bg-white/[0.02] border border-border/70 dark:border-white/10 flex flex-col justify-between space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-display font-bold text-foreground">Linear Models</span>
                <span className={`text-[9px] font-mono font-semibold px-2 py-0.5 rounded border uppercase tracking-wider ${statusTheme.linearStatusClass}`}>
                  {statusTheme.linearStatus}
                </span>
              </div>
              <p className="text-[11px] text-muted-foreground leading-snug">
                Ridge, Lasso, Logistic. Sensitive to multicollinearity & feature scales.
              </p>
            </div>

            {/* Neural Networks */}
            <div className="p-3.5 rounded-xl bg-muted/20 dark:bg-white/[0.02] border border-border/70 dark:border-white/10 flex flex-col justify-between space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-display font-bold text-foreground">Deep Learning</span>
                <span className={`text-[9px] font-mono font-semibold px-2 py-0.5 rounded border uppercase tracking-wider ${statusTheme.nnStatusClass}`}>
                  {statusTheme.nnStatus}
                </span>
              </div>
              <p className="text-[11px] text-muted-foreground leading-snug">
                MLP / PyTorch Tensors. Requires dense standard numeric normalization.
              </p>
            </div>
          </div>
        </div>

        {/* Diagnostics & Guardrails Section */}
        {reasons && reasons.length > 0 ? (
          <div className="pt-2">
            <h4 className="text-xs font-mono uppercase tracking-wider text-muted-foreground font-semibold mb-3 flex items-center gap-1.5">
              <AlertTriangle className="h-3.5 w-3.5 text-amber-500" />
              <span>Detected Pipeline Constraints & Issues</span>
            </h4>
            <div className="grid gap-2">
              {reasons.map((reason, idx) => (
                <div
                  key={`ml-reason-${idx}`}
                  className="p-3 rounded-xl bg-amber-500/[0.05] border border-amber-500/25 flex items-start gap-3 text-xs font-sans text-amber-700 dark:text-amber-200/90"
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-500 mt-1.5 shrink-0" />
                  <span className="leading-relaxed">{reason}</span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="pt-2">
            <h4 className="text-xs font-mono uppercase tracking-wider text-muted-foreground font-semibold mb-3 flex items-center gap-1.5">
              <Check className="h-3.5 w-3.5 text-emerald-500" />
              <span>Production Pipeline Verification Checklist</span>
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div className="p-3 rounded-xl bg-emerald-500/[0.05] border border-emerald-500/25 flex items-start gap-2.5">
                <CheckCircle2 className="h-4 w-4 text-emerald-500 shrink-0 mt-0.5" />
                <div>
                  <h5 className="text-xs font-semibold text-foreground">Completeness Guardrail</h5>
                  <p className="text-[11px] text-muted-foreground mt-0.5">Zero high-null gates (&gt;70%) tripped</p>
                </div>
              </div>
              <div className="p-3 rounded-xl bg-emerald-500/[0.05] border border-emerald-500/25 flex items-start gap-2.5">
                <CheckCircle2 className="h-4 w-4 text-emerald-500 shrink-0 mt-0.5" />
                <div>
                  <h5 className="text-xs font-semibold text-foreground">Variance Integrity</h5>
                  <p className="text-[11px] text-muted-foreground mt-0.5">Zero degenerate or constant features</p>
                </div>
              </div>
              <div className="p-3 rounded-xl bg-emerald-500/[0.05] border border-emerald-500/25 flex items-start gap-2.5">
                <CheckCircle2 className="h-4 w-4 text-emerald-500 shrink-0 mt-0.5" />
                <div>
                  <h5 className="text-xs font-semibold text-foreground">Label Balance</h5>
                  <p className="text-[11px] text-muted-foreground mt-0.5">Distribution free of extreme class skews</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
