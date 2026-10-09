import React from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import {
  AlertTriangle,
  CheckCircle2,
  HelpCircle,
  XCircle,
  Cpu,
  Layers,
  TrendingUp,
  ShieldCheck,
  Check,
  Binary
} from "lucide-react";
import { MLReadiness } from "@/types/api";

interface MLReadinessCardProps {
  mlReadiness?: MLReadiness;
}

export const MLReadinessCard: React.FC<MLReadinessCardProps> = ({ mlReadiness }) => {
  if (!mlReadiness) return null;

  const { score, status, reasons, recommendation, column_context } = mlReadiness;
  const roundedScore = Math.max(0, Math.min(100, Math.round(score)));

  // SVG Gauge calculations (radius = 38, circumference ≈ 238.76)
  const radius = 38;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (roundedScore / 100) * circumference;

  // Status-specific themes & telemetry
  const statusTheme = {
    Ready: {
      accentColor: "#10b981",
      glowBg: "bg-emerald-500/10",
      pillClass: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
      dotClass: "bg-emerald-400",
      statusLabel: "PRODUCTION READY",
      icon: <CheckCircle2 className="h-5 w-5 text-emerald-400" />,
      subtext: "Feature distributions and cardinality profiles are compliant with scikit-learn, XGBoost, and PyTorch training loops.",
      treeStatus: "100% Compatible",
      treeStatusClass: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
      linearStatus: "Verified",
      linearStatusClass: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
      nnStatus: "Input Ready",
      nnStatusClass: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
    },
    "Needs Cleaning": {
      accentColor: "#f59e0b",
      glowBg: "bg-amber-500/10",
      pillClass: "bg-amber-500/15 text-amber-300 border-amber-500/30",
      dotClass: "bg-amber-400",
      statusLabel: "NEEDS PREPROCESSING",
      icon: <AlertTriangle className="h-5 w-5 text-amber-400" />,
      subtext: "Apply recommended transforms or imputation rules before splitting train/test validation cohorts.",
      treeStatus: "Usable with Imputation",
      treeStatusClass: "bg-amber-500/15 text-amber-300 border-amber-500/30",
      linearStatus: "Scaling Advised",
      linearStatusClass: "bg-amber-500/15 text-amber-300 border-amber-500/30",
      nnStatus: "Pre-processing Req.",
      nnStatusClass: "bg-amber-500/15 text-amber-300 border-amber-500/30",
    },
    "Not Ready": {
      accentColor: "#f43f5e",
      glowBg: "bg-rose-500/10",
      pillClass: "bg-rose-500/15 text-rose-300 border-rose-500/30",
      dotClass: "bg-rose-400",
      statusLabel: "UNFIT FOR MODELING",
      icon: <XCircle className="h-5 w-5 text-rose-400" />,
      subtext: "High-null or zero-variance columns must be pruned or imputed to prevent gradient breakdown during training.",
      treeStatus: "Pruning Required",
      treeStatusClass: "bg-rose-500/15 text-rose-300 border-rose-500/30",
      linearStatus: "Convergence Risk",
      linearStatusClass: "bg-rose-500/15 text-rose-300 border-rose-500/30",
      nnStatus: "Pipeline Blocked",
      nnStatusClass: "bg-rose-500/15 text-rose-300 border-rose-500/30",
    },
  }[status] || {
    accentColor: "#64748b",
    glowBg: "bg-slate-500/10",
    pillClass: "bg-slate-500/15 text-slate-300 border-slate-500/30",
    dotClass: "bg-slate-400",
    statusLabel: "AUDIT PENDING",
    icon: <HelpCircle className="h-5 w-5 text-slate-400" />,
    subtext: "Upload a complete dataset to compute full machine learning suitability metrics.",
    treeStatus: "Pending",
    treeStatusClass: "bg-muted text-muted-foreground border-border",
    linearStatus: "Pending",
    linearStatusClass: "bg-muted text-muted-foreground border-border",
    nnStatus: "Pending",
    nnStatusClass: "bg-muted text-muted-foreground border-border",
  };

  return (
    <Card className="relative border border-border/80 bg-card/95 backdrop-blur-xl shadow-premium rounded-3xl overflow-hidden mt-6">
      {/* Ambient Radial Telemetry Aura */}
      <div
        className={`absolute -top-24 -right-24 w-80 h-80 rounded-full ${statusTheme.glowBg} blur-3xl pointer-events-none transition-all duration-700`}
      />

      {/* Header Section */}
      <CardHeader className="border-b border-border/60 bg-muted/15 px-6 py-5 relative z-10">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-start sm:items-center gap-3.5">
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-primary/10 border border-primary/20 text-primary shadow-2xs">
              <Cpu className="h-5 w-5 text-primary" />
            </div>
            <div>
              <div className="flex items-center gap-2.5">
                <CardTitle className="text-lg sm:text-xl font-display font-bold text-foreground tracking-tight">
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
              className={`inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full border text-xs font-mono font-bold tracking-wider shadow-xs ${statusTheme.pillClass}`}
            >
              <span className={`w-2 h-2 rounded-full ${statusTheme.dotClass} animate-pulse`} />
              {statusTheme.statusLabel}
            </span>
          </div>
        </div>
      </CardHeader>

      <CardContent className="p-6 sm:p-7 space-y-6 relative z-10">
        {/* Top Hero Grid: Gauge Telemetry & Expert Advisory */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
          {/* Left: Telemetry Radial Gauge */}
          <div className="lg:col-span-5 p-5 rounded-2xl bg-secondary/50 border border-border/80 shadow-xs flex items-center gap-5">
            {/* SVG Circular Radial Gauge */}
            <div className="relative shrink-0 w-24 h-24 flex items-center justify-center">
              <svg className="w-full h-full -rotate-90 transform" viewBox="0 0 100 100">
                <circle
                  cx="50"
                  cy="50"
                  r={radius}
                  className="stroke-muted/30"
                  strokeWidth="8"
                  fill="transparent"
                />
                <circle
                  cx="50"
                  cy="50"
                  r={radius}
                  stroke={statusTheme.accentColor}
                  strokeWidth="8"
                  strokeDasharray={circumference}
                  strokeDashoffset={strokeDashoffset}
                  strokeLinecap="round"
                  fill="transparent"
                  className="transition-all duration-1000 ease-out"
                />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
                <span className="text-2xl font-display font-black text-foreground tracking-tight leading-none">
                  {roundedScore}%
                </span>
                <span className="text-[9px] font-mono uppercase tracking-widest text-muted-foreground mt-0.5 font-bold">
                  Score
                </span>
              </div>
            </div>

            {/* Status Breakdown & Context */}
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="text-base sm:text-lg font-display font-bold text-foreground truncate">
                  {status}
                </span>
                {statusTheme.icon}
              </div>
              <p className="text-xs font-mono text-muted-foreground mt-1 flex items-center gap-1.5 truncate">
                <ShieldCheck className="h-3.5 w-3.5 text-primary shrink-0" />
                <span>{column_context}</span>
              </p>

              <div className="flex flex-wrap gap-1.5 mt-2.5">
                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-md bg-background/60 border border-border/60 text-muted-foreground">
                  Polars 2.0 Engine
                </span>
                <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-md bg-background/60 border border-border/60 text-muted-foreground">
                  Zero Null Gates
                </span>
              </div>
            </div>
          </div>

          {/* Right: Expert ML Architect Advisory */}
          <div className="lg:col-span-7 p-5 rounded-2xl bg-gradient-to-br from-secondary/70 via-secondary/40 to-transparent border border-border/80 shadow-xs flex flex-col justify-between">
            <div>
              <div className="flex items-center gap-2 text-xs font-mono font-semibold uppercase tracking-wider text-primary mb-2">
                <Cpu className="h-3.5 w-3.5 text-primary" />
                <span>Algorithmic Architecture Advisory</span>
              </div>
              <p className="text-xs sm:text-sm font-sans font-medium text-foreground leading-relaxed">
                {recommendation}
              </p>
            </div>

            <p className="text-[11px] font-sans text-muted-foreground mt-3 pt-3 border-t border-border/50">
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
            <div className="p-3.5 rounded-xl bg-secondary/40 border border-border/70 flex flex-col justify-between space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-display font-bold text-foreground">Tree Ensembles</span>
                <span className={`text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full border ${statusTheme.treeStatusClass}`}>
                  {statusTheme.treeStatus}
                </span>
              </div>
              <p className="text-[11px] text-muted-foreground leading-snug">
                XGBoost, LightGBM, CatBoost. Highly robust to non-linear splits.
              </p>
            </div>

            {/* Linear Models */}
            <div className="p-3.5 rounded-xl bg-secondary/40 border border-border/70 flex flex-col justify-between space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-display font-bold text-foreground">Linear Models</span>
                <span className={`text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full border ${statusTheme.linearStatusClass}`}>
                  {statusTheme.linearStatus}
                </span>
              </div>
              <p className="text-[11px] text-muted-foreground leading-snug">
                Ridge, Lasso, Logistic. Sensitive to multicollinearity & feature scales.
              </p>
            </div>

            {/* Neural Networks */}
            <div className="p-3.5 rounded-xl bg-secondary/40 border border-border/70 flex flex-col justify-between space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-display font-bold text-foreground">Deep Learning</span>
                <span className={`text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full border ${statusTheme.nnStatusClass}`}>
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
              <AlertTriangle className="h-3.5 w-3.5 text-amber-400" />
              <span>Detected Pipeline Constraints & Issues</span>
            </h4>
            <div className="grid gap-2">
              {reasons.map((reason, idx) => (
                <div
                  key={`ml-reason-${idx}`}
                  className="p-3 rounded-xl bg-amber-500/[0.04] border border-amber-500/25 flex items-start gap-3 text-xs font-sans text-amber-200/90"
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-400 mt-1.5 shrink-0" />
                  <span className="leading-relaxed">{reason}</span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="pt-2">
            <h4 className="text-xs font-mono uppercase tracking-wider text-muted-foreground font-semibold mb-3 flex items-center gap-1.5">
              <Check className="h-3.5 w-3.5 text-emerald-400" />
              <span>Production Pipeline Verification Checklist</span>
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div className="p-3 rounded-xl bg-emerald-500/[0.04] border border-emerald-500/25 flex items-start gap-2.5">
                <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
                <div>
                  <h5 className="text-xs font-semibold text-foreground">Completeness Guardrail</h5>
                  <p className="text-[11px] text-muted-foreground mt-0.5">Zero high-null gates (&gt;70%) tripped</p>
                </div>
              </div>
              <div className="p-3 rounded-xl bg-emerald-500/[0.04] border border-emerald-500/25 flex items-start gap-2.5">
                <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
                <div>
                  <h5 className="text-xs font-semibold text-foreground">Variance Integrity</h5>
                  <p className="text-[11px] text-muted-foreground mt-0.5">Zero degenerate or constant features</p>
                </div>
              </div>
              <div className="p-3 rounded-xl bg-emerald-500/[0.04] border border-emerald-500/25 flex items-start gap-2.5">
                <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
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
