import { useState } from "react";
import { Check, AlertTriangle, Play, ShieldAlert, Trash2, Wrench, BarChart2, Hash, Calendar, CheckCircle2, Type } from "lucide-react";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from "@/components/ui/tooltip";
import type { InspectionReport, CleaningRulesMap, CleaningRule } from "@/types/api";

interface DataHealthCheckProps {
    report: InspectionReport;
    onContinue: (rules: CleaningRulesMap) => void;
    isProcessing: boolean;
}

const getTypeStyle = (type: string) => {
    const t = (type || "").toLowerCase();
    if (t.includes("int") || t.includes("float") || t.includes("num")) {
        return {
            icon: <Hash className="h-3 w-3 text-sky-400" />,
            className: "bg-sky-500/10 text-sky-400 border-sky-500/25",
        };
    }
    if (t.includes("datetime") || t.includes("date") || t.includes("time")) {
        return {
            icon: <Calendar className="h-3 w-3 text-amber-400" />,
            className: "bg-amber-500/10 text-amber-400 border-amber-500/25",
        };
    }
    if (t.includes("bool")) {
        return {
            icon: <CheckCircle2 className="h-3 w-3 text-emerald-400" />,
            className: "bg-emerald-500/10 text-emerald-400 border-emerald-500/25",
        };
    }
    return {
        icon: <Type className="h-3 w-3 text-violet-400" />,
        className: "bg-violet-500/10 text-violet-400 border-violet-500/25",
    };
};

export const DataHealthCheck = ({ report, onContinue, isProcessing }: DataHealthCheckProps) => {
    // Automatically pre-populate rules with dynamic suggestions from the backend engine
    const [rules, setRules] = useState<CleaningRulesMap>(() => {
        const initial: CleaningRulesMap = {};
        (report?.issues || []).forEach(issue => {
            if (issue.column && issue.column !== "Multiple" && issue.action) {
                initial[issue.column] = {
                    action: issue.action as CleaningRule["action"],
                    value: issue.action === "fill_value" ? "Unknown" : undefined
                };
            }
        });
        return initial;
    });

    const handleActionChange = (column: string, action: string) => {
        setRules(prev => ({
            ...prev,
            [column]: {
                // SAFETY: SelectItem values only contain valid CleaningRule action values
                action: action as CleaningRule["action"],
                value: action === "fill_value" ? "Unknown" : undefined
            }
        }));
    };

    const handleApplyAllSuggestions = () => {
        const updated: CleaningRulesMap = {};
        (report?.issues || []).forEach(issue => {
            if (issue.column && issue.column !== "Multiple" && issue.action) {
                updated[issue.column] = {
                    action: issue.action as CleaningRule["action"],
                    value: issue.action === "fill_value" ? "Unknown" : undefined
                };
            }
        });
        setRules(updated);
    };

    const handleResetAllToIgnore = () => {
        setRules({});
    };

    const getActionForColumn = (column: string) => {
        return rules[column]?.action || "default"; // "default" means auto-pilot
    };

    const handleSubmit = () => {
        onContinue(rules);
    };

    const columnsWithIssues = (report?.columns || []).filter(col => {
        const issue = (report?.issues || []).find(i => i.column === col.name);
        return col.missing_count > 0 || (issue && ['outliers', 'high_cardinality', 'class_imbalance'].includes(issue.type));
    });

    return (
        <div className="space-y-8 max-w-4xl mx-auto animate-in fade-in slide-in-from-bottom-4 duration-500">

            {/* Header Section */}
            <div className="text-center space-y-2 max-w-xl mx-auto">
                <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary text-xs font-semibold uppercase tracking-wider mb-1">
                    <ShieldAlert className="w-3.5 h-3.5" /> Hygiene Audit & Rules
                </div>
                <h2 className="text-3xl font-display font-bold text-foreground tracking-tight">Data Health Check</h2>
                <p className="text-sm text-muted-foreground leading-relaxed font-sans">
                    We scanned your dataset for structural anomalies, missing values, and outliers. Review suggested fixes before commencing analysis.
                </p>
            </div>

            {/* ─── Global Warnings ─── */}
            {(report?.issues || []).flatMap((issue) => {
                if (issue.column !== "Multiple") return [];
                return [
                    <div key={`global-warning-${issue.type}-${issue.count}`} className="bg-gradient-to-r from-amber-500/10 via-amber-500/5 to-transparent border border-amber-500/30 p-5 rounded-2xl flex items-start gap-4 shadow-sm animate-in fade-in duration-300">
                        <div className="w-10 h-10 rounded-xl bg-amber-500/20 text-amber-400 border border-amber-500/30 flex items-center justify-center shrink-0 shadow-2xs">
                            <AlertTriangle className="h-5 w-5" />
                        </div>
                        <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                                <h4 className="text-sm font-display font-bold text-amber-300">
                                    {issue.type === 'partial_duplicates' ? "Ambiguous Data Detected" : "Quality Warning"}
                                </h4>
                                <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/30">
                                    {issue.count} rows
                                </span>
                            </div>
                            <p className="text-xs text-amber-200/90 mt-1 font-sans leading-relaxed">
                                {issue.type === 'partial_duplicates'
                                    ? `Found ${issue.count} rows that look identical but have different IDs (Partial Duplicates). Review rules below.`
                                    : `System detected ${issue.count} potential quality conflicts across multiple attributes.`
                                }
                            </p>
                        </div>
                    </div>
                ];
            })}

            {/* ─── Dynamic Rules Toolbar ─── */}
            {columnsWithIssues.length > 0 && (
                <div className="flex flex-wrap items-center justify-between gap-3 bg-muted/20 border border-border/50 p-3 rounded-xl">
                    <div className="text-xs text-muted-foreground font-sans">
                        Dynamic suggestions calculated based on column distribution, skewness, and cardinality.
                    </div>
                    <div className="flex items-center gap-2">
                        <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            onClick={handleApplyAllSuggestions}
                            className="text-xs h-8 rounded-lg cursor-pointer hover:bg-primary/10 hover:text-primary border-border/70 font-sans"
                        >
                            <CheckCircle2 className="w-3.5 h-3.5 mr-1.5 text-primary" />
                            Apply Dynamic Suggestions
                        </Button>
                        <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            onClick={handleResetAllToIgnore}
                            className="text-xs h-8 rounded-lg cursor-pointer text-muted-foreground hover:text-foreground font-sans"
                        >
                            Ignore All
                        </Button>
                    </div>
                </div>
            )}

            {/* Column Health Grid */}
            <div className="grid gap-5 md:grid-cols-2 lg:grid-cols-3">
                {(report?.columns || []).map((col) => {
                    const issue = (report?.issues || []).find(i => i.column === col.name);
                    const hasIssue = col.missing_count > 0 || (issue && ['outliers', 'high_cardinality', 'class_imbalance'].includes(issue.type));

                    if (!hasIssue) return null;

                    const typeStyle = getTypeStyle(col.inferred_type);

                    return (
                        <Card key={col.name} className="border border-border/80 bg-card rounded-2xl shadow-premium overflow-hidden flex flex-col justify-between t-card-lift">
                            <CardHeader className="pb-3 bg-muted/10 border-b border-border/40">
                                <div className="flex justify-between items-start gap-2">
                                    <CardTitle className="text-base font-display font-bold text-foreground truncate" title={col.name}>
                                        {col.name}
                                    </CardTitle>
                                    <TooltipProvider>
                                        <Tooltip>
                                            <TooltipTrigger>
                                                <span
                                                    className={`inline-flex items-center text-[10px] font-mono gap-1 px-2.5 py-0.5 rounded-full border ${typeStyle.className} shrink-0 shadow-2xs`}
                                                >
                                                    {typeStyle.icon}
                                                    {col.inferred_type}
                                                </span>
                                            </TooltipTrigger>
                                            <TooltipContent className="rounded-xl border border-border/80 bg-popover text-popover-foreground shadow-xl">
                                                <p className="text-xs font-mono">Inferred Type: {col.inferred_type}</p>
                                            </TooltipContent>
                                        </Tooltip>
                                    </TooltipProvider>
                                </div>
                                <CardDescription className="flex items-center gap-1.5 text-amber-400 font-sans text-xs font-semibold mt-1">
                                    <AlertTriangle className="h-3.5 w-3.5 text-amber-400 shrink-0" />
                                    <span>
                                        {issue?.type === 'outliers'
                                            ? `${issue.count} outliers detected`
                                            : issue?.type === 'high_cardinality'
                                                ? `${issue.count} unique values`
                                                : issue?.type === 'class_imbalance'
                                                    ? `Top category dominates`
                                                    : `${col.missing_count} missing (${col.missing_percentage}%)`
                                        }
                                    </span>
                                </CardDescription>
                            </CardHeader>

                            <CardContent className="py-4 space-y-3">
                                <div className="p-3 rounded-xl bg-muted/30 border border-border/60 text-xs text-muted-foreground font-sans space-y-1.5">
                                    <div className="flex items-center justify-between">
                                        <span className="text-[11px] text-muted-foreground uppercase font-semibold tracking-wider">Auto-suggestion</span>
                                        {issue?.action && (
                                            <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-primary/10 text-primary border border-primary/20">
                                                Dynamic
                                            </span>
                                        )}
                                    </div>
                                    <div className="font-semibold text-foreground text-sm flex items-center gap-1.5">
                                        <Wrench className="w-3.5 h-3.5 text-primary shrink-0" />
                                        <span className="truncate">{issue?.label || issue?.suggestion || "Ignore (Leave as is)"}</span>
                                    </div>
                                    {issue?.rationale && (
                                        <p className="text-[11px] text-muted-foreground leading-relaxed font-sans">
                                            {issue.rationale}
                                        </p>
                                    )}
                                </div>
                                {col.distribution && <SparklineHistogram data={col.distribution} />}
                            </CardContent>

                            <CardFooter className="pt-0 pb-4 px-4">
                                <Select
                                    value={getActionForColumn(col.name)}
                                    onValueChange={(val) => handleActionChange(col.name, val)}
                                >
                                    <SelectTrigger className="w-full bg-secondary/80 hover:bg-secondary border-border/80 text-foreground rounded-xl text-xs h-10 font-medium hover:border-primary/40 focus:ring-primary/20 shadow-2xs transition-colors cursor-pointer">
                                        <SelectValue placeholder="Select action…" />
                                    </SelectTrigger>
                                    <SelectContent className="rounded-xl border border-border/80 bg-popover text-popover-foreground shadow-2xl backdrop-blur-xl">
                                        <SelectItem value="default" className="text-xs font-medium text-foreground cursor-pointer focus:bg-accent focus:text-foreground">
                                            <span className="text-muted-foreground hover:text-foreground flex items-center gap-2">
                                                Ignore (Leave as is)
                                            </span>
                                        </SelectItem>
                                        <SelectItem value="drop_rows" className="text-xs font-medium text-rose-400 focus:text-rose-300 focus:bg-rose-500/15 cursor-pointer">
                                            <span className="flex items-center justify-between w-full gap-2">
                                                <span className="flex items-center gap-2">
                                                    <Trash2 className="w-3.5 h-3.5 text-rose-400" /> Drop Rows
                                                </span>
                                                {issue?.action === "drop_rows" && (
                                                    <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                                                        Recommended
                                                    </span>
                                                )}
                                            </span>
                                        </SelectItem>

                                        {col.inferred_type === 'numeric' && (
                                            <>
                                                {issue?.type !== 'outliers' && (
                                                    <>
                                                        <SelectItem value="fill_median" className="text-xs font-medium text-foreground focus:bg-accent focus:text-foreground cursor-pointer">
                                                            <span className="flex items-center justify-between w-full gap-2">
                                                                <span className="flex items-center gap-2">
                                                                    <Wrench className="w-3.5 h-3.5 text-sky-400" /> Fill with Median
                                                                </span>
                                                                {issue?.action === "fill_median" && (
                                                                    <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                                                                        Recommended
                                                                    </span>
                                                                )}
                                                            </span>
                                                        </SelectItem>
                                                        <SelectItem value="fill_mean" className="text-xs font-medium text-foreground focus:bg-accent focus:text-foreground cursor-pointer">
                                                            <span className="flex items-center justify-between w-full gap-2">
                                                                <span className="flex items-center gap-2">
                                                                    <Wrench className="w-3.5 h-3.5 text-sky-400" /> Fill with Average
                                                                </span>
                                                                {issue?.action === "fill_mean" && (
                                                                    <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                                                                        Recommended
                                                                    </span>
                                                                )}
                                                            </span>
                                                        </SelectItem>
                                                        <SelectItem value="fill_mode" className="text-xs font-medium text-foreground focus:bg-accent focus:text-foreground cursor-pointer">
                                                            <span className="flex items-center justify-between w-full gap-2">
                                                                <span className="flex items-center gap-2">
                                                                    <Wrench className="w-3.5 h-3.5 text-indigo-400" /> Fill with Most Frequent
                                                                </span>
                                                                {issue?.action === "fill_mode" && (
                                                                    <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                                                                        Recommended
                                                                    </span>
                                                                )}
                                                            </span>
                                                        </SelectItem>
                                                    </>
                                                )}
                                                {issue?.type === 'outliers' && (
                                                    <SelectItem value="replace_outliers_median" className="text-xs font-medium text-foreground focus:bg-accent focus:text-foreground cursor-pointer">
                                                        <span className="flex items-center justify-between w-full gap-2">
                                                            <span className="flex items-center gap-2">
                                                                <Wrench className="w-3.5 h-3.5 text-amber-400" /> Cap Outliers (Median)
                                                            </span>
                                                            {issue?.action === "replace_outliers_median" && (
                                                                <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                                                                    Recommended
                                                                </span>
                                                            )}
                                                        </span>
                                                    </SelectItem>
                                                )}
                                            </>
                                        )}
                                        {col.inferred_type !== 'numeric' && (
                                            <>
                                                <SelectItem value="fill_mode" className="text-xs font-medium text-foreground focus:bg-accent focus:text-foreground cursor-pointer">
                                                    <span className="flex items-center justify-between w-full gap-2">
                                                        <span className="flex items-center gap-2">
                                                            <Wrench className="w-3.5 h-3.5 text-indigo-400" /> Fill with Most Frequent
                                                        </span>
                                                        {issue?.action === "fill_mode" && (
                                                            <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                                                                Recommended
                                                            </span>
                                                        )}
                                                    </span>
                                                </SelectItem>
                                                <SelectItem value="fill_value" className="text-xs font-medium text-foreground focus:bg-accent focus:text-foreground cursor-pointer">
                                                    <span className="flex items-center justify-between w-full gap-2">
                                                        <span className="flex items-center gap-2">
                                                            <Wrench className="w-3.5 h-3.5 text-indigo-400" /> Fill with &ldquo;Unknown&rdquo;
                                                        </span>
                                                        {issue?.action === "fill_value" && (
                                                            <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                                                                Recommended
                                                            </span>
                                                        )}
                                                    </span>
                                                </SelectItem>
                                            </>
                                        )}
                                    </SelectContent>
                                </Select>
                            </CardFooter>
                        </Card>
                    );
                })}
            </div>

            {report.issues.length === 0 && (
                <div className="bg-card border border-border/80 shadow-premium rounded-2xl p-8 max-w-2xl mx-auto text-center mt-6">
                    <div className="w-16 h-16 bg-emerald-500/15 rounded-full flex items-center justify-center mx-auto mb-5 border border-emerald-500/30 shadow-2xs">
                        <Check className="h-8 w-8 text-emerald-400" />
                    </div>
                    <h3 className="text-2xl font-display font-bold text-foreground">Data Quality: Excellent</h3>
                    <p className="text-muted-foreground mt-2 max-w-md mx-auto text-sm leading-relaxed">
                        We have successfully run our pre-analysis checks and found no critical issues, missing values, or problematic distributions. Your dataset is clean and ready for deep analysis.
                    </p>
                    
                    <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 gap-4 text-left">
                        <div className="bg-muted/30 border border-border/60 rounded-xl p-4 flex items-start gap-3">
                            <div className="bg-emerald-500/15 p-1.5 rounded-md mt-0.5 border border-emerald-500/25"><Check className="h-4 w-4 text-emerald-400" /></div>
                            <div>
                                <h4 className="text-sm font-semibold text-foreground">Format Integrity</h4>
                                <p className="text-xs text-muted-foreground mt-1">All columns contain valid types.</p>
                            </div>
                        </div>
                        <div className="bg-muted/30 border border-border/60 rounded-xl p-4 flex items-start gap-3">
                            <div className="bg-emerald-500/15 p-1.5 rounded-md mt-0.5 border border-emerald-500/25"><Check className="h-4 w-4 text-emerald-400" /></div>
                            <div>
                                <h4 className="text-sm font-semibold text-foreground">Data Completeness</h4>
                                <p className="text-xs text-muted-foreground mt-1">No missing cells detected.</p>
                            </div>
                        </div>
                    </div>
                </div>
            )}


            {/* ─── DATA PREVIEW TABLE ─── */}
            {report.preview && report.preview.length > 0 && (
                <div className="border border-border/80 bg-card shadow-premium rounded-2xl overflow-hidden">
                    <div className="bg-muted/20 px-5 py-3.5 border-b border-border/60 flex items-center justify-between">
                        <h3 className="text-sm font-display font-bold text-foreground tracking-tight">Data Preview (First 5 Rows)</h3>
                        <span className="text-xs font-mono text-muted-foreground">{report.preview.length} sample rows</span>
                    </div>
                    <div className="overflow-x-auto custom-scrollbar">
                        <table className="w-full text-sm text-left border-collapse">
                            <thead className="bg-muted/40 text-foreground font-semibold">
                                <tr>
                                    {Object.keys(report.preview[0]).map((header) => (
                                        <th key={header} className="px-4 py-3 border-b border-r border-border/60 whitespace-nowrap font-mono text-xs last:border-r-0">
                                            {header}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-border/60 bg-card">
                                {report.preview.map((row, rowPos) => (
                                    <tr key={`row_pos_${rowPos}`} className="border-b border-border/40 last:border-0 hover:bg-primary/[0.02] transition-colors">
                                        {Object.entries(row).map(([header, cell]) => (
                                            <td key={`cell_${header}_${rowPos}`} className="px-4 py-2.5 border-r border-border/60 font-mono text-xs whitespace-nowrap max-w-[200px] truncate last:border-r-0 text-foreground/90" title={String(cell)}>
                                                {cell === null ? <span className="text-muted-foreground italic font-sans">null</span> : String(cell)}
                                            </td>
                                        ))}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {/* Action Bar */}
            <div className="flex justify-center pt-4">
                <Button
                    size="lg"
                    onClick={handleSubmit}
                    disabled={isProcessing}
                    className="w-full sm:w-auto min-w-[220px] rounded-xl transition-all duration-150 hover:-translate-y-0.5 active:scale-95 font-semibold text-sm py-6 bg-gradient-to-r from-violet-600 via-purple-600 to-indigo-600 hover:from-violet-500 hover:via-purple-500 hover:to-indigo-500 text-white font-display border border-violet-400/40 ring-1 ring-white/20 shadow-[0_4px_14px_-2px_rgba(124,58,237,0.38),inset_0_1px_1px_rgba(255,255,255,0.3)] cursor-pointer"
                >
                    {isProcessing ? (
                        "Processing…"
                    ) : (
                        <>
                            <Play className="mr-2 h-4 w-4 fill-current" />
                            Start Analysis
                        </>
                    )}
                </Button>
            </div>

        </div>
    );
};

// ─── Sparkline Histogram Component ───
const SparklineHistogram = ({ data }: { data: { count: number; label: string }[] }) => {
    if (!data || data.length === 0) return null;
    const max = Math.max(...data.map(d => d.count)) || 1;

    return (
        <div className="mt-3.5 pt-2 border-t border-border/40">
            <div className="flex items-center justify-between text-xs text-muted-foreground mb-1.5 font-sans">
                <span className="font-semibold text-foreground/80 flex items-center gap-1.5">
                    <BarChart2 className="w-3.5 h-3.5 text-primary" /> Distribution
                </span>
                <span className="text-[10px] font-mono">{data.length} bins</span>
            </div>
            <div className="flex items-end h-14 gap-[3px] w-full bg-muted/20 p-2 rounded-xl border border-border/50">
                {data.map((d) => (
                    <TooltipProvider key={`hist_bin_${d.label}`}>
                        <Tooltip>
                            <TooltipTrigger asChild>
                                <div
                                    className="flex-1 bg-gradient-to-t from-primary/30 to-primary/80 hover:from-primary hover:to-primary/90 transition-all rounded-t-xs cursor-pointer shadow-2xs"
                                    style={{ height: `${Math.max(12, (d.count / max) * 100)}%` }}
                                />
                            </TooltipTrigger>
                            <TooltipContent className="rounded-xl border border-border/80 bg-popover text-popover-foreground shadow-xl">
                                <p className="text-xs font-sans text-popover-foreground">{d.label}: <strong className="font-mono text-primary">{d.count}</strong></p>
                            </TooltipContent>
                        </Tooltip>
                    </TooltipProvider>
                ))}
            </div>
        </div>
    );
};

