import { AlertTriangle, Database } from "lucide-react";
import { Card } from "@/components/ui/card";
import type { ConfidenceScores, DatasetInfo } from "@/types/api";
import { MLReadinessCard } from "../MLReadinessCard";

function parseCriticalIssue(ci: string) {
  const colonIdx = ci.indexOf(":");
  const col = colonIdx !== -1 ? ci.substring(0, colonIdx).trim() : "Dataset";
  const rest = colonIdx !== -1 ? ci.substring(colonIdx + 1).trim() : ci;
  const matchPct = rest.match(/(\d+(?:\.\d+)?)\s*%/);
  const pctStr = matchPct ? `${matchPct[1]}%` : null;
  const pctNum = matchPct ? parseFloat(matchPct[1]) : 0;
  const label = matchPct ? rest.replace(/\s*\(\d+(?:\.\d+)?%\)/, "").trim() : rest;
  return { col, pctStr, pctNum, label };
}

function CriticalIssueNoticeCard({ issue }: { issue: string }) {
  const { col, pctStr, pctNum, label } = parseCriticalIssue(issue);

  return (
    <div className="group relative rounded-2xl border border-stone-200/90 dark:border-white/10 bg-stone-50/50 dark:bg-[#181116] p-4 hover:border-rose-300 dark:hover:border-rose-800/60 hover:bg-white dark:hover:bg-[#1d141b] transition-all duration-300 flex flex-col justify-between shadow-2xs">
      <div>
        <div className="flex items-center justify-between gap-2">
          <span
            className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-white dark:bg-white/10 text-stone-900 dark:text-stone-100 font-mono text-xs font-semibold border border-stone-200/90 dark:border-white/10 truncate max-w-[170px]"
            title={col}
          >
            <Database className="h-3 w-3 text-stone-500 dark:text-zinc-400 shrink-0" />
            <span className="truncate">{col}</span>
          </span>
          <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold tracking-wider uppercase bg-rose-50 text-rose-700 border border-rose-200/80 dark:bg-rose-950/70 dark:text-rose-300 dark:border-rose-800/60 shrink-0">
            Severe
          </span>
        </div>

        <div className="mt-3.5 space-y-1.5">
          <div className="flex items-center justify-between text-xs">
            <span className="font-sans font-medium text-stone-600 dark:text-zinc-400 truncate">
              {label || "High missing rate"}
            </span>
            {pctStr && (
              <span className="font-mono font-bold text-rose-700 dark:text-rose-400 tabular-nums ml-2 shrink-0">
                {pctStr}
              </span>
            )}
          </div>

          {pctNum > 0 && (
            <div className="h-1.5 w-full rounded-full bg-stone-200/80 dark:bg-white/10 overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-rose-500 to-rose-600 dark:from-rose-500 dark:to-rose-400 transition-all duration-500"
                style={{ width: `${Math.min(pctNum, 100)}%` }}
              />
            </div>
          )}
        </div>
      </div>

      <div className="mt-4 pt-2.5 border-t border-stone-200/70 dark:border-white/[0.08] flex items-center justify-between">
        <span className="text-[11px] font-sans font-medium text-stone-500 dark:text-zinc-400">
          Suggested Action
        </span>
        <span className="font-mono font-semibold text-[10px] px-2 py-0.5 rounded-md bg-stone-100 dark:bg-white/5 text-stone-700 dark:text-zinc-300 border border-stone-200/90 dark:border-white/10">
          {pctNum >= 80 ? "Drop Column" : "Impute / Synthesize"}
        </span>
      </div>
    </div>
  );
}

export function OverviewTab({
  datasetConfidence,
  info,
  confidence,
}: {
  datasetConfidence: number;
  info: DatasetInfo;
  confidence: ConfidenceScores;
}) {
  const criticalIssues = confidence.critical_issues || [];

  return (
    <div className="space-y-8 animate-in fade-in duration-300">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <Card className="border border-border/80 bg-card shadow-premium rounded-2xl p-5">
          <span className="text-[10px] font-sans font-semibold uppercase tracking-[0.15em] text-muted-foreground block">
            Completeness
          </span>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-2xl sm:text-3xl font-sans font-bold text-foreground tracking-tight tabular-nums">
              {datasetConfidence.toFixed(1)}%
            </span>
            <span className="text-xs font-sans font-medium text-emerald-700 dark:text-emerald-400">High Quality</span>
          </div>
          <p className="text-xs font-sans text-muted-foreground/80 mt-1">Overall data confidence index</p>
        </Card>

        <Card className="border border-border/80 bg-card shadow-premium rounded-2xl p-5">
          <span className="text-[10px] font-sans font-semibold uppercase tracking-[0.15em] text-muted-foreground block">
            Dataset Dimensions
          </span>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-2xl sm:text-3xl font-sans font-bold text-foreground tracking-tight tabular-nums">
              {info.rows.toLocaleString()}
            </span>
            <span className="text-xs font-sans font-medium text-muted-foreground">x {info.columns.length} cols</span>
          </div>
          <p className="text-xs font-sans text-muted-foreground/80 mt-1">Processed without schema drift</p>
        </Card>

        <Card className="border border-border/80 bg-card shadow-premium rounded-2xl p-5">
          <span className="text-[10px] font-sans font-semibold uppercase tracking-[0.15em] text-muted-foreground block">
            Numeric vs Categorical
          </span>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-2xl sm:text-3xl font-sans font-bold text-foreground tracking-tight tabular-nums">
              {info.numeric_columns?.length || 0} / {info.categorical_columns?.length || 0}
            </span>
          </div>
          <p className="text-xs font-sans text-muted-foreground/80 mt-1">Continuous vs discrete variables</p>
        </Card>

        <Card className="border border-border/80 bg-card shadow-premium rounded-2xl p-5">
          <span className="text-[10px] font-sans font-semibold uppercase tracking-[0.15em] text-muted-foreground block">
            Memory Utilization
          </span>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-2xl sm:text-3xl font-sans font-bold text-foreground tracking-tight tabular-nums">
              {(info.memory_usage_mb || 0).toFixed(2)}
            </span>
            <span className="text-xs font-sans font-medium text-muted-foreground">MB</span>
          </div>
          <p className="text-xs font-sans text-muted-foreground/80 mt-1">Optimized columnar storage</p>
        </Card>
      </div>

      {criticalIssues.length > 0 && (
        <div className="rounded-3xl border border-rose-200/90 dark:border-rose-900/40 bg-stone-50/70 dark:bg-rose-950/10 p-1 sm:p-1.5 shadow-xs">
          <div className="rounded-[calc(1.5rem-2px)] sm:rounded-[calc(1.5rem-4px)] border border-rose-100 dark:border-rose-900/30 bg-white dark:bg-[#130d11] p-5 sm:p-6 shadow-xs">
            <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4">
              <div className="flex items-start gap-3.5">
                <div className="h-10 w-10 rounded-2xl bg-rose-50 dark:bg-rose-950/60 text-rose-600 dark:text-rose-400 border border-rose-200/80 dark:border-rose-800/50 flex items-center justify-center shrink-0 shadow-2xs mt-0.5">
                  <AlertTriangle className="h-5 w-5" />
                </div>
                <div className="space-y-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-[10px] font-mono font-semibold uppercase tracking-[0.14em] text-rose-600 dark:text-rose-400">
                      Data Integrity Notice
                    </span>
                    <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-rose-50 text-rose-700 dark:bg-rose-950/80 dark:text-rose-300 border border-rose-200/80 dark:border-rose-800/60">
                      {criticalIssues.length} {criticalIssues.length === 1 ? "Column Gap" : "Column Gaps"} Flagged
                    </span>
                  </div>
                  <h3 className="text-base sm:text-lg font-sans font-bold text-stone-900 dark:text-white tracking-tight">
                    Critical Data Hygiene Notices
                  </h3>
                  <p className="text-xs text-stone-600 dark:text-zinc-400 font-sans leading-relaxed max-w-2xl">
                    The following columns exhibit severe missingness (&gt;50%) or distribution anomalies that may distort statistical aggregations and degrade downstream predictive models.
                  </p>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3.5 mt-5">
              {criticalIssues.map((ci) => (
                <CriticalIssueNoticeCard key={ci} issue={ci} />
              ))}
            </div>
          </div>
        </div>
      )}

      <MLReadinessCard mlReadiness={confidence.ml_readiness} />
    </div>
  );
}
