import { useMemo, useState } from "react";
import { AlertTriangle, ChevronDown, ChevronUp, Search } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import type { ColumnConfidence, ConfidenceScores, DatasetInfo } from "@/types/api";
import { getGradeColorClass } from "./gradeTheme";

function TrustLedgerColumnRow({
  columnScore,
  dtypeLabel,
  isExpanded,
  onToggle,
}: {
  columnScore: ColumnConfidence;
  dtypeLabel: string;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  const hasIssues = Boolean(columnScore.issues && columnScore.issues.length > 0);

  return (
    <div
      className={`border rounded-2xl transition-all duration-200 ${
        isExpanded ? "border-primary bg-primary/5" : "border-border/80 bg-card hover:bg-muted/20"
      }`}
    >
      <button
        type="button"
        onClick={onToggle}
        className="w-full flex items-center justify-between p-3.5 cursor-pointer select-none gap-2 text-left bg-transparent border-0"
      >
        <div className="flex items-center gap-3 min-w-0 flex-1">
          <Badge className={`h-7 w-7 rounded-lg flex items-center justify-center p-0 font-bold shrink-0 ${getGradeColorClass(columnScore.grade)}`}>
            {columnScore.grade}
          </Badge>
          <span className="font-mono text-xs font-semibold text-foreground truncate max-w-[200px]" title={columnScore.column}>
            {columnScore.column}
          </span>
          <Badge variant="outline" className="text-[10px] font-mono px-2 py-0 h-4 bg-muted/20 border-border rounded-full hidden sm:inline-flex">
            {dtypeLabel}
          </Badge>
        </div>

        <div className="flex items-center gap-3 shrink-0 ml-auto">
          <span className="font-mono text-xs text-muted-foreground font-semibold">
            {columnScore.overall.toFixed(0)}%
          </span>
          {hasIssues && (
            <Badge variant="secondary" className="bg-amber-500/10 text-amber-400 border border-amber-500/30 h-5 px-2 rounded-full text-[9px] font-mono">
              {columnScore.issues.length} alert{columnScore.issues.length > 1 ? "s" : ""}
            </Badge>
          )}
          {isExpanded ? (
            <ChevronUp className="h-4 w-4 text-muted-foreground shrink-0" />
          ) : (
            <ChevronDown className="h-4 w-4 text-muted-foreground shrink-0" />
          )}
        </div>
      </button>

      {isExpanded && (
        <div className="p-4 border-t border-border/60 space-y-3 font-mono text-xs animate-in slide-in-from-top-1 duration-200">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-muted/20 p-3 rounded-xl">
            <div className="space-y-0.5">
              <span className="text-[10px] text-muted-foreground uppercase">Completeness</span>
              <div className="text-foreground font-bold">{columnScore.completeness.toFixed(1)}%</div>
            </div>
            <div className="space-y-0.5">
              <span className="text-[10px] text-muted-foreground uppercase">Consistency</span>
              <div className="text-foreground font-bold">{columnScore.consistency.toFixed(1)}%</div>
            </div>
            <div className="space-y-0.5">
              <span className="text-[10px] text-muted-foreground uppercase">Validity</span>
              <div className="text-foreground font-bold">{columnScore.validity.toFixed(1)}%</div>
            </div>
            <div className="space-y-0.5">
              <span className="text-[10px] text-muted-foreground uppercase">Stability</span>
              <div className="text-foreground font-bold">{columnScore.stability.toFixed(1)}%</div>
            </div>
          </div>

          {hasIssues && (
            <div className="rounded-2xl bg-gradient-to-b from-amber-500/[0.05] via-zinc-900/60 to-zinc-950/80 border border-amber-500/30 p-3.5 space-y-2.5 shadow-[0_2px_12px_rgba(0,0,0,0.35),inset_0_1px_0_rgba(255,255,255,0.06)] backdrop-blur-md">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="w-5 h-5 rounded-md bg-amber-500/20 text-amber-300 border border-amber-500/40 flex items-center justify-center shrink-0 shadow-[0_0_8px_rgba(245,158,11,0.25)]">
                    <AlertTriangle className="h-3 w-3 stroke-[2.4]" />
                  </span>
                  <span className="text-xs font-semibold text-zinc-100 font-sans tracking-tight">
                    Identified Quality Flags
                  </span>
                </div>
                <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40">
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-400 shadow-[0_0_6px_rgba(245,158,11,0.9)] animate-pulse" />
                  {columnScore.issues.length} Flagged
                </span>
              </div>
              <div className="flex flex-wrap gap-2 pt-0.5">
                {columnScore.issues.map((issueStr) => (
                  <span
                    key={`col-issue-${columnScore.column}-${issueStr}`}
                    className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-medium text-stone-800 bg-stone-100/90 hover:bg-stone-200/90 border border-stone-300 hover:border-amber-500/50 shadow-xs dark:text-zinc-100 dark:bg-zinc-900/90 dark:hover:bg-zinc-800/90 dark:border-white/10 dark:hover:border-amber-400/50 dark:shadow-[0_1px_3px_rgba(0,0,0,0.4),inset_0_1px_0_rgba(255,255,255,0.06)] hover:shadow-[0_0_14px_rgba(245,158,11,0.2)] transition-all duration-200 select-none cursor-default"
                  >
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-500 dark:bg-amber-400 shadow-[0_0_6px_rgba(245,158,11,0.8)]" />
                    <span className="font-sans tracking-tight text-stone-800 dark:text-zinc-200">{issueStr}</span>
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function TrustLedgerTab({
  confidence,
  info,
}: {
  confidence: ConfidenceScores;
  info: DatasetInfo;
}) {
  const [expandedColumns, setExpandedColumns] = useState<Record<string, boolean>>({});
  const [ledgerSearch, setLedgerSearch] = useState<string>("");
  const [ledgerGradeFilter, setLedgerGradeFilter] = useState<string>("all");

  const toggleColumnExpand = (colName: string) => {
    setExpandedColumns((prev) => ({ ...prev, [colName]: !prev[colName] }));
  };

  const filteredLedgerColumns = useMemo(() => {
    return (confidence?.columns || []).filter((c) => {
      const matchesSearch = (c.column || "").toLowerCase().includes(ledgerSearch.toLowerCase().trim());
      if (!matchesSearch) return false;
      if (ledgerGradeFilter === "all") return true;
      if (ledgerGradeFilter === "A") return (c.grade || "").toUpperCase() === "A";
      if (ledgerGradeFilter === "B") return (c.grade || "").toUpperCase() === "B";
      if (ledgerGradeFilter === "attention") return ["C", "D", "F"].includes((c.grade || "").toUpperCase());
      return true;
    });
  }, [confidence?.columns, ledgerSearch, ledgerGradeFilter]);

  return (
    <Card className="border border-border bg-card shadow-premium rounded-3xl p-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-border pb-4">
        <div>
          <h3 className="text-lg font-display font-bold text-foreground">Column Trust Ledger</h3>
          <p className="text-xs text-muted-foreground">Completeness, consistency, validity, and stability audits</p>
        </div>

        <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2">
          <div className="flex items-center gap-1 bg-muted/40 p-1 rounded-xl border border-border/60">
            <button
              type="button"
              onClick={() => setLedgerGradeFilter("all")}
              className={`px-2.5 py-1 rounded-lg text-xs font-mono font-semibold transition-all ${
                ledgerGradeFilter === "all"
                  ? "bg-card text-foreground border border-border/80 shadow-xs"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              All ({confidence.columns.length})
            </button>
            <button
              type="button"
              onClick={() => setLedgerGradeFilter("A")}
              className={`px-2.5 py-1 rounded-lg text-xs font-mono font-semibold transition-all ${
                ledgerGradeFilter === "A"
                  ? "bg-emerald-500/20 text-emerald-800 border border-emerald-500/30 shadow-xs dark:text-emerald-300"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Grade A
            </button>
            <button
              type="button"
              onClick={() => setLedgerGradeFilter("attention")}
              className={`px-2.5 py-1 rounded-lg text-xs font-mono font-semibold transition-all ${
                ledgerGradeFilter === "attention"
                  ? "bg-amber-500/20 text-amber-800 border border-amber-500/30 shadow-xs dark:text-amber-300"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Flags
            </button>
          </div>

          <div className="relative w-full sm:w-48">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
            <Input
              placeholder="Search variables..."
              value={ledgerSearch}
              onChange={(e) => setLedgerSearch(e.target.value)}
              className="pl-9 h-8 text-xs rounded-xl bg-background border-border/80 text-foreground placeholder:text-muted-foreground focus-visible:border-primary/50"
            />
          </div>
        </div>
      </div>

      <div className="space-y-2.5 pt-4">
        {filteredLedgerColumns.length > 0 ? (
          filteredLedgerColumns.map((c) => (
            <TrustLedgerColumnRow
              key={c.column}
              columnScore={c}
              dtypeLabel={info.dtypes[c.column] || "variable"}
              isExpanded={Boolean(expandedColumns[c.column])}
              onToggle={() => toggleColumnExpand(c.column)}
            />
          ))
        ) : (
          <div className="text-center py-8 text-xs font-mono text-muted-foreground">
            No columns match your filter criteria.
          </div>
        )}
      </div>
    </Card>
  );
}
