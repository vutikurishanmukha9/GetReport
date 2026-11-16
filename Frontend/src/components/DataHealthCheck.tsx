import { useState, useMemo, useEffect } from "react";
import {
    Check,
    AlertTriangle,
    Play,
    ShieldAlert,
    Trash2,
    Wrench,
    BarChart2,
    Hash,
    Calendar,
    CheckCircle2,
    Type,
    TrendingUp,
    Layers,
    RotateCcw,
    ArrowRight,
    SlidersHorizontal,
    CheckCheck,
    ShieldCheck
} from "lucide-react";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useToast } from "@/hooks/use-toast";
import {
    Select,
    SelectContent,
    SelectGroup,
    SelectItem,
    SelectLabel,
    SelectSeparator,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from "@/components/ui/tooltip";
import type { InspectionReport, CleaningRulesMap, CleaningRule, QualityIssue } from "@/types/api";

const KNOWN_ACTIONS = new Set([
    "default",
    "cap_outliers",
    "replace_outliers_median",
    "winsorize",
    "flag_outliers",
    "drop_rows",
    "coerce_numeric",
    "clean_masked_nulls",
    "clean_text",
    "fix_mojibake",
    "standardize_categories",
    "cast_boolean",
    "keep_as_text",
    "parse_date",
    "nullify_sentinel",
    "nullify_out_of_range",
    "nullify_invalid_regex",
    "drop_column",
    "drop_duplicate_keys",
    "drop_duplicate_rows",
    "drop_empty_rows",
    "keep_first",
    "keep_last",
    "fill_median",
    "fill_mean",
    "fill_mode",
    "fill_value",
    "fill_group_median",
    "add_missing_indicator",
    "log1p_transform",
    "hash_pii",
]);

const DESTRUCTIVE_ACTIONS = new Set([
    "drop_rows",
    "drop_column",
    "drop_duplicate_keys",
    "drop_duplicate_rows",
    "drop_empty_rows",
    "nullify_out_of_range",
    "nullify_invalid_regex",
]);

const getActionDisplayLabel = (action: string, issue?: QualityIssue | null): string => {
    switch (action) {
        case "cap_outliers":
        case "replace_outliers_median":
        case "winsorize":
            return "Cap Outliers (Tukey IQR)";
        case "flag_outliers":
            return "Flag Outliers (Indicator)";
        case "drop_rows":
            return "Drop Missing Rows";
        case "coerce_numeric":
            return "Coerce to Numeric (Strip Currency/%)";
        case "clean_masked_nulls":
            return "Clean Masked Nulls (to null)";
        case "clean_text":
            return "Clean Text (Trim Whitespace)";
        case "fix_mojibake":
            return "Fix Mojibake Encoding";
        case "standardize_categories":
            return "Standardize Categories";
        case "cast_boolean":
            return "Cast to Boolean";
        case "keep_as_text":
            return "Preserve as Text";
        case "parse_date":
            return "Parse Date (ISO 8601)";
        case "nullify_sentinel":
            return "Nullify Sentinel Values";
        case "nullify_out_of_range":
            return "Nullify Out-of-Range";
        case "nullify_invalid_regex":
            return "Nullify Invalid Format";
        case "drop_column":
            return "Drop Column";
        case "drop_duplicate_keys":
        case "keep_first":
            return issue?.label || "Keep First Row per Key";
        case "keep_last":
            return "Keep Last Row per Key";
        case "drop_duplicate_rows":
            return "Drop Duplicate Rows";
        case "drop_empty_rows":
            return "Drop Empty Rows";
        case "fill_median":
            return "Fill with Median";
        case "fill_mean":
            return "Fill with Average";
        case "fill_mode":
            return "Fill with Most Frequent";
        case "fill_value":
            return "Fill with 'Unknown'";
        case "fill_group_median":
            return "Fill with Group Median";
        case "add_missing_indicator":
            return "Add Missing Indicator";
        case "log1p_transform":
            return "Log1p Transform (Reduce Skew)";
        case "hash_pii":
            return "Hash / Mask PII";
        case "default":
            return "Ignore (Leave raw as is)";
        default:
            return issue?.label || issue?.suggestion || action || "Ignore (Leave raw as is)";
    }
};

const getActionIcon = (action: string) => {
    switch (action) {
        case "cap_outliers":
        case "replace_outliers_median":
        case "winsorize":
            return <Wrench className="w-3.5 h-3.5 text-amber-700 dark:text-amber-400 shrink-0" />;
        case "flag_outliers":
            return <CheckCircle2 className="w-3.5 h-3.5 text-sky-700 dark:text-sky-400 shrink-0" />;
        case "log1p_transform":
            return <BarChart2 className="w-3.5 h-3.5 text-blue-700 dark:text-blue-400 shrink-0" />;
        case "drop_rows":
        case "drop_column":
        case "drop_duplicate_rows":
        case "drop_empty_rows":
            return <Trash2 className="w-3.5 h-3.5 text-rose-700 dark:text-rose-400 shrink-0" />;
        case "drop_duplicate_keys":
        case "keep_first":
        case "keep_last":
            return <Wrench className="w-3.5 h-3.5 text-stone-700 dark:text-zinc-300 shrink-0" />;
        case "coerce_numeric":
            return <Hash className="w-3.5 h-3.5 text-blue-700 dark:text-blue-400 shrink-0" />;
        case "parse_date":
            return <Calendar className="w-3.5 h-3.5 text-teal-700 dark:text-teal-400 shrink-0" />;
        case "clean_masked_nulls":
        case "nullify_sentinel":
            return <Wrench className="w-3.5 h-3.5 text-amber-700 dark:text-amber-400 shrink-0" />;
        case "nullify_out_of_range":
        case "nullify_invalid_regex":
            return <AlertTriangle className="w-3.5 h-3.5 text-rose-700 dark:text-rose-400 shrink-0" />;
        case "clean_text":
            return <Type className="w-3.5 h-3.5 text-violet-700 dark:text-violet-400 shrink-0" />;
        case "fix_mojibake":
            return <Wrench className="w-3.5 h-3.5 text-emerald-700 dark:text-emerald-400 shrink-0" />;
        case "standardize_categories":
            return <SlidersHorizontal className="w-3.5 h-3.5 text-indigo-700 dark:text-indigo-400 shrink-0" />;
        case "cast_boolean":
            return <CheckCircle2 className="w-3.5 h-3.5 text-emerald-700 dark:text-emerald-400 shrink-0" />;
        case "keep_as_text":
            return <Type className="w-3.5 h-3.5 text-stone-600 dark:text-zinc-400 shrink-0" />;
        case "fill_median":
        case "fill_mean":
        case "fill_mode":
        case "fill_value":
            return <Wrench className="w-3.5 h-3.5 text-stone-600 dark:text-zinc-400 shrink-0" />;
        case "fill_group_median":
            return <Layers className="w-3.5 h-3.5 text-purple-700 dark:text-purple-400 shrink-0" />;
        case "add_missing_indicator":
            return <CheckCircle2 className="w-3.5 h-3.5 text-sky-700 dark:text-sky-400 shrink-0" />;
        case "hash_pii":
            return <ShieldAlert className="w-3.5 h-3.5 text-emerald-700 dark:text-emerald-400 shrink-0" />;
        case "default":
            return <RotateCcw className="w-3.5 h-3.5 text-stone-500 dark:text-zinc-400 shrink-0" />;
        default:
            return <Wrench className="w-3.5 h-3.5 text-stone-700 dark:text-zinc-300 shrink-0" />;
    }
};

interface DataHealthCheckProps {
    report: InspectionReport;
    onContinue: (rules: CleaningRulesMap) => void;
    isProcessing: boolean;
}

const getTypeStyle = (type: string) => {
    const t = (type || "").toLowerCase();
    if (t.includes("int") || t.includes("float") || t.includes("num")) {
        return {
            icon: <Hash className="h-3 w-3 text-stone-700 dark:text-zinc-300" />,
            className: "bg-stone-200/80 text-stone-800 border-stone-300 dark:bg-white/5 dark:text-zinc-300 dark:border-white/10",
        };
    }
    if (t.includes("datetime") || t.includes("date") || t.includes("time")) {
        return {
            icon: <Calendar className="h-3 w-3 text-amber-700 dark:text-amber-400" />,
            className: "bg-amber-100 text-amber-900 border-amber-300 dark:bg-amber-500/10 dark:text-amber-400 dark:border-amber-500/25",
        };
    }
    if (t.includes("bool")) {
        return {
            icon: <CheckCircle2 className="h-3 w-3 text-emerald-700 dark:text-emerald-400" />,
            className: "bg-emerald-100 text-emerald-900 border-emerald-300 dark:bg-emerald-500/10 dark:text-emerald-400 dark:border-emerald-500/25",
        };
    }
    return {
        icon: <Type className="h-3 w-3 text-violet-700 dark:text-violet-400" />,
        className: "bg-violet-100 text-violet-900 border-violet-300 dark:bg-violet-500/10 dark:text-violet-400 dark:border-violet-500/25",
    };
};

export const DataHealthCheck = ({ report, onContinue, isProcessing }: DataHealthCheckProps) => {
    const { toast } = useToast();
    const [activePreset, setActivePreset] = useState<"all" | "high" | "conservative" | "raw" | "custom">("all");

    // Automatically pre-populate rules with dynamic suggestions from the backend engine
    const [rules, setRules] = useState<CleaningRulesMap>(() => {
        const initial: CleaningRulesMap = {};
        (report?.issues || []).forEach(issue => {
            if (issue.column && issue.column !== "Multiple" && issue.action && issue.action !== "default") {
                const normalizedAction = (issue.action === "replace_outliers_median" || issue.action === "winsorize")
                    ? "cap_outliers"
                    : issue.action;
                initial[issue.column] = {
                    // SAFETY: normalizedAction is validated from issue.action
                    action: normalizedAction as CleaningRule["action"],
                    value: normalizedAction === "fill_value" ? "Unknown" : undefined
                };
            }
        });
        return initial;
    });

    // Synchronize rules if report issues load or update
    useEffect(() => {
        if (!report?.issues || report.issues.length === 0) return;
        setRules(prev => {
            if (Object.keys(prev).length > 0) return prev;
            const initial: CleaningRulesMap = {};
            (report.issues || []).forEach(issue => {
                if (issue.column && issue.column !== "Multiple" && issue.action && issue.action !== "default") {
                    const normalizedAction = (issue.action === "replace_outliers_median" || issue.action === "winsorize")
                        ? "cap_outliers"
                        : issue.action;
                    initial[issue.column] = {
                        // SAFETY: normalizedAction is validated from issue.action
                        action: normalizedAction as CleaningRule["action"],
                        value: normalizedAction === "fill_value" ? "Unknown" : undefined
                    };
                }
            });
            return initial;
        });
    }, [report]);

    const activeRulesCount = useMemo(() => {
        return Object.values(rules).filter(r => r?.action && r.action !== "default").length;
    }, [rules]);

    const handleActionChange = (column: string, action: string) => {
        const normalizedAction = (action === "replace_outliers_median" || action === "winsorize")
            ? "cap_outliers"
            : action;
        setRules(prev => ({
            ...prev,
            [column]: {
                // SAFETY: normalizedAction is validated from action selection
                action: normalizedAction as CleaningRule["action"],
                value: normalizedAction === "fill_value" ? "Unknown" : undefined
            }
        }));
        setActivePreset("custom");
    };

    const handleApplyAllSuggestions = () => {
        const updated: CleaningRulesMap = {};
        let count = 0;
        (report?.issues || []).forEach(issue => {
            if (issue.column && issue.column !== "Multiple" && issue.action && issue.action !== "default") {
                const normalizedAction = (issue.action === "replace_outliers_median" || issue.action === "winsorize")
                    ? "cap_outliers"
                    : issue.action;
                updated[issue.column] = {
                    // SAFETY: normalizedAction is validated from issue.action
                    action: normalizedAction as CleaningRule["action"],
                    value: normalizedAction === "fill_value" ? "Unknown" : undefined
                };
                count += 1;
            }
        });
        setRules(updated);
        setActivePreset("all");
        toast({
            title: "Applied All Suggestions",
            description: `Activated all ${count} automated cleaning rules across your flagged columns.`,
        });
    };

    const handleApplyHighConfidenceOnly = () => {
        const updated: CleaningRulesMap = {};
        let appliedCount = 0;
        let skippedCount = 0;
        (report?.issues || []).forEach(issue => {
            if (issue.column && issue.column !== "Multiple" && issue.action && issue.action !== "default") {
                const conf = issue.confidence_score ?? 100;
                if (conf >= 90) {
                    const normalizedAction = (issue.action === "replace_outliers_median" || issue.action === "winsorize")
                        ? "cap_outliers"
                        : issue.action;
                    updated[issue.column] = {
                        // SAFETY: normalizedAction is validated from issue.action
                        action: normalizedAction as CleaningRule["action"],
                        value: normalizedAction === "fill_value" ? "Unknown" : undefined
                    };
                    appliedCount += 1;
                } else {
                    skippedCount += 1;
                }
            }
        });
        setRules(updated);
        setActivePreset("high");
        toast({
            title: "Applied High-Confidence Suggestions",
            description: skippedCount > 0
                ? `Activated ${appliedCount} high-confidence rules (>=90%). Bypassed ${skippedCount} lower-confidence rule(s).`
                : `Activated all ${appliedCount} high-confidence cleaning rules.`,
        });
    };

    const handleApplyConservative = () => {
        const updated: CleaningRulesMap = {};
        let safeApplied = 0;
        let destructiveSkipped = 0;

        (report?.issues || []).forEach(issue => {
            if (issue.column && issue.column !== "Multiple" && issue.action && issue.action !== "default") {
                const normalizedAction = (issue.action === "replace_outliers_median" || issue.action === "winsorize")
                    ? "cap_outliers"
                    : issue.action;

                if (!DESTRUCTIVE_ACTIONS.has(normalizedAction)) {
                    updated[issue.column] = {
                        // SAFETY: normalizedAction is validated and verified non-destructive
                        action: normalizedAction as CleaningRule["action"],
                        value: normalizedAction === "fill_value" ? "Unknown" : undefined
                    };
                    safeApplied += 1;
                } else {
                    destructiveSkipped += 1;
                }
            }
        });

        setRules(updated);
        setActivePreset("conservative");
        toast({
            title: "Conservative Preset Applied",
            description: destructiveSkipped > 0
                ? `Applied ${safeApplied} safe transformation(s). Skipped ${destructiveSkipped} destructive row/column deletion rule(s).`
                : `Applied ${safeApplied} safe transformation(s) across flagged columns.`,
        });
    };

    const handleResetAllToIgnore = () => {
        setRules({});
        setActivePreset("raw");
        toast({
            title: "Reset to Raw Data",
            description: "All automated cleaning rules disabled. Dataset will be inspected in its raw original state.",
        });
    };

    const getActionForColumn = (column: string) => {
        const raw = rules[column]?.action;
        if (!raw) return "default";
        if (raw === "replace_outliers_median" || raw === "winsorize") return "cap_outliers";
        return raw;
    };

    const handleSubmit = () => {
        onContinue(rules);
    };

    const columnsWithIssues = useMemo(() => {
        return (report?.columns || []).filter(col => {
            const hasIssue = (report?.issues || []).some(i => i.column === col.name);
            return col.missing_count > 0 || hasIssue;
        });
    }, [report]);

    const highConfidenceCount = useMemo(() => {
        return (report?.issues || []).filter(
            i => i.column !== "Multiple" && i.action && (i.confidence_score ?? 100) >= 90
        ).length;
    }, [report]);

    return (
        <div className="space-y-8 max-w-4xl mx-auto animate-in fade-in slide-in-from-bottom-4 duration-500">

            {/* Header Section */}
            <div className="text-center space-y-2 max-w-xl mx-auto">
                <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary text-xs font-semibold uppercase tracking-wider mb-1">
                    <ShieldAlert className="w-3.5 h-3.5" /> Hygiene Audit & Rules
                </div>
                <h2 className="text-3xl font-sans font-bold text-foreground tracking-[-0.03em]">Data Health Check</h2>
                <p className="text-sm text-muted-foreground leading-relaxed font-sans">
                    We scanned your dataset with distribution-aware heuristics to detect masked nulls, formatted numbers, zero-variance columns, and outliers.
                </p>
            </div>

            {/* Global Warnings */}
            {(report?.issues || []).flatMap((issue) => {
                if (issue.column !== "Multiple") return [];
                return [
                    <div key={`global-warning-${issue.type}-${issue.count}`} className="bg-gradient-to-r from-amber-500/10 via-amber-500/5 to-transparent border border-amber-500/30 p-5 rounded-2xl flex items-start gap-4 shadow-sm animate-in fade-in duration-300">
                        <div className="w-10 h-10 rounded-xl bg-amber-500/20 text-amber-400 border border-amber-500/30 flex items-center justify-center shrink-0 shadow-2xs">
                            <AlertTriangle className="h-5 w-5" />
                        </div>
                        <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                                <h4 className="text-sm font-sans font-semibold text-amber-300">
                                    {issue.type === 'partial_duplicates' ? "Ambiguous Data Detected" : "Quality Warning"}
                                </h4>
                                <span className="text-[10px] font-sans font-bold tabular-nums px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/30">
                                    {issue.count} rows
                                </span>
                            </div>
                            <p className="text-xs text-amber-200/90 mt-1 font-sans leading-relaxed">
                                {issue.type === 'partial_duplicates'
                                    ? `Found ${issue.count} rows that look identical across data attributes but have different identifiers. Review rules below.`
                                    : `System detected ${issue.count} potential quality conflicts across multiple attributes.`
                                }
                            </p>
                        </div>
                    </div>
                ];
            })}

            {/* Strategy Presets Toolbar */}
            {columnsWithIssues.length > 0 && (
                <div className="bg-[#FAF8F5] dark:bg-[#0c0d12] border border-stone-200/90 dark:border-white/10 p-3.5 sm:p-4 rounded-2xl space-y-3 shadow-2xs">
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-stone-200/80 dark:border-white/5 pb-2.5">
                        <div className="flex items-center gap-2 flex-wrap">
                            <SlidersHorizontal className="w-4 h-4 text-[#800000] dark:text-primary" />
                            <span className="text-xs font-semibold text-stone-900 dark:text-foreground tracking-tight">
                                Auto-Suggestion Presets
                            </span>
                            <span className="text-[11px] font-sans font-medium tabular-nums px-2 py-0.5 rounded-md bg-stone-200/70 text-stone-700 border border-stone-300/80 dark:bg-white/5 dark:text-zinc-400 dark:border-white/10">
                                {columnsWithIssues.length} columns flagged
                            </span>
                            <span className="text-[11px] font-sans font-medium tabular-nums px-2 py-0.5 rounded-md bg-stone-900/5 text-stone-800 border border-stone-300/60 dark:bg-white/10 dark:text-zinc-300 dark:border-white/15">
                                {activePreset === "all" && `All Suggestions (${activeRulesCount} rules)`}
                                {activePreset === "conservative" && `Conservative (${activeRulesCount} safe rules)`}
                                {activePreset === "high" && `High-Confidence (${activeRulesCount} rules)`}
                                {activePreset === "raw" && `Raw Data (0 rules)`}
                                {activePreset === "custom" && `Custom (${activeRulesCount} active)`}
                            </span>
                        </div>
                        <div className="text-[11px] text-stone-500 dark:text-muted-foreground font-sans">
                            Select a preset to reconfigure all column actions below
                        </div>
                    </div>

                    <div className="flex flex-wrap items-center gap-2">
                        <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            onClick={handleApplyAllSuggestions}
                            className={cn(
                                "text-xs h-9 px-3.5 rounded-xl cursor-pointer transition-all duration-150 font-sans flex items-center gap-2 select-none",
                                activePreset === "all"
                                    ? "bg-[#800000] hover:bg-[#6b0000] text-white border-[#800000] dark:bg-zinc-100 dark:hover:bg-zinc-200 dark:text-zinc-950 dark:border-white shadow-xs font-semibold active:scale-[0.98]"
                                    : "bg-white hover:bg-stone-50 text-stone-700 hover:text-stone-950 border border-stone-200/90 dark:bg-white/[0.04] dark:hover:bg-white/[0.08] dark:text-zinc-300 dark:hover:text-white dark:border-white/10 shadow-2xs font-medium active:scale-[0.98]"
                            )}
                        >
                            <Layers className={cn("w-3.5 h-3.5 shrink-0", activePreset === "all" ? "text-white dark:text-zinc-950" : "text-stone-500 dark:text-zinc-400")} />
                            <span>Apply All Suggestions</span>
                        </Button>

                        <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            onClick={handleApplyConservative}
                            className={cn(
                                "text-xs h-9 px-3.5 rounded-xl cursor-pointer transition-all duration-150 font-sans flex items-center gap-2 select-none",
                                activePreset === "conservative"
                                    ? "bg-[#800000] hover:bg-[#6b0000] text-white border-[#800000] dark:bg-zinc-100 dark:hover:bg-zinc-200 dark:text-zinc-950 dark:border-white shadow-xs font-semibold active:scale-[0.98]"
                                    : "bg-white hover:bg-stone-50 text-stone-700 hover:text-stone-950 border border-stone-200/90 dark:bg-white/[0.04] dark:hover:bg-white/[0.08] dark:text-zinc-300 dark:hover:text-white dark:border-white/10 shadow-2xs font-medium active:scale-[0.98]"
                            )}
                        >
                            <ShieldCheck className={cn("w-3.5 h-3.5 shrink-0", activePreset === "conservative" ? "text-white dark:text-zinc-950" : "text-stone-500 dark:text-zinc-400")} />
                            <span>Conservative (Safe Only)</span>
                        </Button>

                        <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            onClick={handleApplyHighConfidenceOnly}
                            className={cn(
                                "text-xs h-9 px-3.5 rounded-xl cursor-pointer transition-all duration-150 font-sans flex items-center gap-2 select-none",
                                activePreset === "high"
                                    ? "bg-[#800000] hover:bg-[#6b0000] text-white border-[#800000] dark:bg-zinc-100 dark:hover:bg-zinc-200 dark:text-zinc-950 dark:border-white shadow-xs font-semibold active:scale-[0.98]"
                                    : "bg-white hover:bg-stone-50 text-stone-700 hover:text-stone-950 border border-stone-200/90 dark:bg-white/[0.04] dark:hover:bg-white/[0.08] dark:text-zinc-300 dark:hover:text-white dark:border-white/10 shadow-2xs font-medium active:scale-[0.98]"
                            )}
                        >
                            <CheckCheck className={cn("w-3.5 h-3.5 shrink-0", activePreset === "high" ? "text-white dark:text-zinc-950" : "text-stone-500 dark:text-zinc-400")} />
                            <span>Apply High-Confidence</span>
                            <span className={cn(
                                "text-[10px] font-mono font-medium px-1.5 py-0.5 rounded-md tabular-nums leading-none",
                                activePreset === "high"
                                    ? "bg-white/20 text-white dark:bg-black/15 dark:text-zinc-950"
                                    : "bg-stone-100 text-stone-600 border border-stone-200/80 dark:bg-white/10 dark:text-zinc-300 dark:border-white/5"
                            )}>
                                {highConfidenceCount}
                            </span>
                        </Button>

                        <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            onClick={handleResetAllToIgnore}
                            className={cn(
                                "text-xs h-9 px-3 rounded-xl cursor-pointer transition-all duration-150 font-sans flex items-center gap-1.5 sm:ml-auto select-none",
                                activePreset === "raw"
                                    ? "bg-stone-900 text-white border-stone-900 dark:bg-zinc-200 dark:text-zinc-950 dark:border-white shadow-xs font-semibold"
                                    : "bg-transparent hover:bg-stone-200/50 text-stone-600 hover:text-stone-900 border border-dashed border-stone-300 hover:border-stone-400 dark:border-white/15 dark:text-zinc-400 dark:hover:text-zinc-200"
                            )}
                        >
                            <RotateCcw className="w-3.5 h-3.5 shrink-0" />
                            <span>Reset to Raw</span>
                        </Button>
                    </div>
                </div>
            )}

            {/* Column Health Grid */}
            <div className="grid gap-5 md:grid-cols-2 lg:grid-cols-3">
                {(report?.columns || []).map((col) => {
                    const issue = (report?.issues || []).find(i => i.column === col.name);
                    const hasIssue = col.missing_count > 0 || !!issue;

                    if (!hasIssue) return null;

                    const typeStyle = getTypeStyle(col.inferred_type);
                    const currentAction = getActionForColumn(col.name);
                    const isActionActive = currentAction !== "default";
                    const recommendedAction = issue?.action && issue.action !== "default"
                        ? (issue.action === "replace_outliers_median" || issue.action === "winsorize" ? "cap_outliers" : issue.action)
                        : null;

                    return (
                        <Card
                            key={col.name}
                            className={cn(
                                "border rounded-2xl shadow-sm dark:shadow-premium overflow-hidden flex flex-col justify-between t-card-lift transition-all duration-200",
                                isActionActive
                                    ? "border-emerald-300/90 dark:border-emerald-500/30 bg-[#FAF8F5] dark:bg-[#0c0d12] ring-1 ring-emerald-500/15"
                                    : "border-stone-300 dark:border-white/10 bg-stone-50/60 dark:bg-[#0c0d12]/60 opacity-90"
                            )}
                        >
                            <CardHeader className={cn(
                                "pb-3 border-b transition-colors",
                                isActionActive
                                    ? "bg-emerald-50/50 border-emerald-100 dark:bg-emerald-950/10 dark:border-white/5"
                                    : "bg-stone-100/70 border-stone-200 dark:bg-white/[0.02] dark:border-white/5"
                            )}>
                                <div className="flex justify-between items-start gap-2">
                                    <CardTitle className="text-base font-sans font-semibold text-foreground truncate tracking-tight" title={col.name}>
                                        {col.name}
                                    </CardTitle>
                                    <div className="flex items-center gap-1.5 shrink-0">
                                        {isActionActive ? (
                                            <span className="inline-flex items-center gap-1 text-[10px] font-sans font-semibold px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 border border-emerald-300 dark:bg-emerald-500/15 dark:text-emerald-300 dark:border-emerald-500/30 shadow-2xs">
                                                <Check className="w-2.5 h-2.5 text-emerald-700 dark:text-emerald-400" />
                                                Active Fix
                                            </span>
                                        ) : (
                                            <span className="inline-flex items-center gap-1 text-[10px] font-sans font-medium px-2 py-0.5 rounded-full bg-stone-200 text-stone-700 border border-stone-300 dark:bg-white/5 dark:text-zinc-400 dark:border-white/10 shadow-2xs">
                                                <RotateCcw className="w-2.5 h-2.5 text-stone-500 dark:text-zinc-400" />
                                                Raw
                                            </span>
                                        )}
                                        <TooltipProvider>
                                            <Tooltip>
                                                <TooltipTrigger>
                                                    <span
                                                        className={`inline-flex items-center text-[10px] font-sans font-medium gap-1 px-2.5 py-0.5 rounded-full border ${typeStyle.className} shrink-0 shadow-2xs`}
                                                    >
                                                        {typeStyle.icon}
                                                        {col.inferred_type}
                                                    </span>
                                                </TooltipTrigger>
                                                <TooltipContent className="rounded-xl border border-stone-300 bg-[#FAF8F5] text-stone-900 dark:border-border/80 dark:bg-popover dark:text-popover-foreground shadow-xl">
                                                    <p className="text-xs font-sans">Inferred Type: <code className="font-mono text-emerald-700 dark:text-emerald-400 font-semibold">{col.inferred_type}</code> ({col.dtype})</p>
                                                </TooltipContent>
                                            </Tooltip>
                                        </TooltipProvider>
                                    </div>
                                </div>

                                <CardDescription className="flex items-center gap-1.5 text-amber-800 dark:text-amber-400 font-sans text-xs font-semibold mt-1">
                                    <AlertTriangle className="h-3.5 w-3.5 text-amber-700 dark:text-amber-400 shrink-0" />
                                    <span>
                                        {issue?.type === 'masked_nulls'
                                            ? `${issue.count} masked placeholder(s)`
                                            : issue?.type === 'dirty_numeric'
                                                ? `${issue.count} formatted string(s)`
                                                : issue?.type === 'constant_column'
                                                    ? `Zero variance / Constant`
                                                    : issue?.type === 'text_whitespace'
                                                        ? `${issue.count} untrimmed strings`
                                                        : issue?.type === 'outliers'
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
                                {/* Suggestion Header with Confidence & Impact */}
                                <div className={cn(
                                    "p-3 rounded-xl border text-xs font-sans space-y-2 transition-colors",
                                    isActionActive
                                        ? "bg-stone-100/90 border-stone-200 text-stone-700 dark:bg-white/[0.03] dark:border-white/10 dark:text-muted-foreground"
                                        : "bg-stone-100/50 border-stone-200/60 text-stone-500 dark:bg-white/[0.01] dark:border-white/5 dark:text-zinc-500"
                                )}>
                                    <div className="flex items-center justify-between gap-1 flex-wrap">
                                        <span className="text-[10px] text-stone-500 dark:text-muted-foreground uppercase font-semibold tracking-wider font-sans">
                                            {isActionActive ? "Current Cleaning Action" : "Bypassed (Original Data)"}
                                        </span>
                                        <div className="flex items-center gap-1.5">
                                            {issue?.confidence_score !== undefined && (
                                                <span className="inline-flex items-center gap-1 text-[10px] font-sans font-semibold tabular-nums px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 border border-emerald-300 dark:bg-emerald-500/10 dark:text-emerald-400 dark:border-emerald-500/20">
                                                    <CheckCircle2 className="w-2.5 h-2.5" />
                                                    {issue.confidence_score}%
                                                </span>
                                            )}
                                            {issue?.impact_estimate && (
                                                <span className="inline-flex items-center gap-1 text-[10px] font-sans font-semibold tabular-nums px-2 py-0.5 rounded-full bg-sky-100 text-sky-900 border border-sky-300 dark:bg-sky-500/10 dark:text-sky-400 dark:border-sky-500/20">
                                                    <TrendingUp className="w-2.5 h-2.5" />
                                                    {issue.impact_estimate}
                                                </span>
                                            )}
                                        </div>
                                    </div>

                                    <div className={cn(
                                        "font-semibold text-sm flex items-center gap-1.5 transition-colors",
                                        isActionActive
                                            ? "text-stone-900 dark:text-foreground"
                                            : "text-stone-500 dark:text-zinc-400"
                                    )}>
                                        {getActionIcon(currentAction)}
                                        <span className="truncate">{getActionDisplayLabel(currentAction, issue)}</span>
                                    </div>

                                    {issue?.rationale && (
                                        <p className="text-[11px] text-stone-600 dark:text-muted-foreground leading-relaxed font-sans">
                                            {issue.rationale}
                                        </p>
                                    )}

                                    {/* Before vs After Visual Diff */}
                                    {isActionActive && issue?.before_sample && issue?.after_sample && issue.before_sample.length > 0 && (
                                        <div className="p-2 rounded-lg bg-stone-100/90 border border-stone-200 dark:bg-black/40 dark:border-white/5 space-y-1 font-mono text-[11px] mt-2">
                                            <div className="flex items-center justify-between text-[9px] text-stone-600 dark:text-zinc-400 uppercase tracking-wider font-sans font-semibold">
                                                <span>Preview</span>
                                                <span>Before → After</span>
                                            </div>
                                            <div className="space-y-1 pt-0.5">
                                                {issue.before_sample.slice(0, 2).map((b, idx) => {
                                                    const afterVal = issue.after_sample?.[idx];
                                                    return (
                                                        <div key={`sample_${col.name}_${idx}_${String(b)}`} className="flex items-center justify-between gap-2 text-xs">
                                                            <span className="text-rose-700 dark:text-rose-400/90 truncate max-w-[110px]" title={String(b)}>
                                                                {b === null ? "null" : String(b)}
                                                            </span>
                                                            <ArrowRight className="w-3 h-3 text-stone-500 dark:text-zinc-400 shrink-0" />
                                                            <span className="text-emerald-700 dark:text-emerald-400/90 truncate max-w-[110px] font-semibold" title={String(afterVal)}>
                                                                {afterVal === null ? "null" : String(afterVal)}
                                                            </span>
                                                        </div>
                                                    );
                                                })}
                                            </div>
                                        </div>
                                    )}
                                </div>

                                {col.distribution && <SparklineHistogram data={col.distribution} />}
                            </CardContent>

                            <CardFooter className="pt-0 pb-4 px-4">
                                <Select
                                    value={currentAction}
                                    onValueChange={(val) => handleActionChange(col.name, val)}
                                >
                                            <SelectTrigger className="w-full bg-white hover:bg-stone-50 border-stone-300 text-stone-900 dark:bg-[#131620] dark:hover:bg-[#1a1f2c] dark:border-white/10 dark:text-foreground rounded-xl text-xs h-10 font-medium focus:border-stone-400 dark:hover:border-primary/40 focus:ring-1 focus:ring-stone-400/20 shadow-2xs transition-colors cursor-pointer">
                                                <div className="flex items-center gap-2 truncate">
                                                    {getActionIcon(currentAction)}
                                                    <SelectValue placeholder="Select action…">
                                                        {getActionDisplayLabel(currentAction, issue)}
                                                    </SelectValue>
                                                </div>
                                            </SelectTrigger>
                                            <SelectContent className="w-[320px] sm:w-[360px] max-h-[380px] rounded-xl border border-stone-300 bg-white text-stone-900 dark:border-white/10 dark:bg-[#0c0d12] dark:text-zinc-100 shadow-2xl backdrop-blur-xl p-1.5 custom-scrollbar">
                                                {/* ── 1. RECOMMENDED AUTO-SUGGESTION (ALWAYS AT THE VERY TOP) ── */}
                                                {recommendedAction && (
                                                    <>
                                                        <SelectGroup>
                                                            <SelectLabel className="flex items-center gap-1.5 text-[10px] font-sans font-bold uppercase tracking-wider text-emerald-800 dark:text-emerald-400 px-2.5 py-1">
                                                                <ShieldCheck className="w-3.5 h-3.5 text-emerald-700 dark:text-emerald-400 shrink-0" />
                                                                Recommended Fix
                                                            </SelectLabel>
                                                            <SelectItem
                                                                value={recommendedAction}
                                                                className="bg-emerald-50/90 hover:bg-emerald-100/90 border border-emerald-300/80 dark:bg-emerald-950/30 dark:hover:bg-emerald-950/50 dark:border-emerald-500/30 my-1"
                                                            >
                                                                <span className="flex items-center justify-between w-full gap-2">
                                                                    <span className="flex items-center gap-2 font-semibold text-emerald-950 dark:text-emerald-200 truncate">
                                                                        {getActionIcon(recommendedAction)}
                                                                        <span className="truncate">{getActionDisplayLabel(recommendedAction, issue)}</span>
                                                                    </span>
                                                                    <span className="inline-flex items-center gap-1 text-[10px] font-sans font-bold px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 border border-emerald-300 dark:bg-emerald-500/20 dark:text-emerald-300 dark:border-emerald-500/30 shrink-0 shadow-2xs">
                                                                        Recommended
                                                                    </span>
                                                                </span>
                                                            </SelectItem>
                                                        </SelectGroup>
                                                        <SelectSeparator className="my-1.5" />
                                                    </>
                                                )}

                                                {/* ── 2. MISSING VALUES & IMPUTATION ── */}
                                                <SelectGroup>
                                                    <SelectLabel>Missing Values & Imputation</SelectLabel>
                                                    {recommendedAction !== "fill_median" && (
                                                        <SelectItem value="fill_median">
                                                            <span className="flex items-center gap-2">
                                                                {getActionIcon("fill_median")} Fill with Median
                                                            </span>
                                                        </SelectItem>
                                                    )}
                                                    {recommendedAction !== "fill_mean" && (
                                                        <SelectItem value="fill_mean">
                                                            <span className="flex items-center gap-2">
                                                                {getActionIcon("fill_mean")} Fill with Average
                                                            </span>
                                                        </SelectItem>
                                                    )}
                                                    {recommendedAction !== "fill_mode" && (
                                                        <SelectItem value="fill_mode">
                                                            <span className="flex items-center gap-2">
                                                                {getActionIcon("fill_mode")} Fill with Most Frequent
                                                            </span>
                                                        </SelectItem>
                                                    )}
                                                    {recommendedAction !== "fill_value" && (
                                                        <SelectItem value="fill_value">
                                                            <span className="flex items-center gap-2">
                                                                {getActionIcon("fill_value")} Fill with &ldquo;Unknown&rdquo;
                                                            </span>
                                                        </SelectItem>
                                                    )}
                                                    {recommendedAction !== "fill_group_median" && (
                                                        <SelectItem value="fill_group_median">
                                                            <span className="flex items-center gap-2">
                                                                {getActionIcon("fill_group_median")} Fill with Group Median
                                                            </span>
                                                        </SelectItem>
                                                    )}
                                                    {recommendedAction !== "add_missing_indicator" && (
                                                        <SelectItem value="add_missing_indicator">
                                                            <span className="flex items-center gap-2">
                                                                {getActionIcon("add_missing_indicator")} Add Missing Indicator Column
                                                            </span>
                                                        </SelectItem>
                                                    )}
                                                    {recommendedAction !== "drop_rows" && (
                                                        <SelectItem value="drop_rows" className="text-rose-700 dark:text-rose-400">
                                                            <span className="flex items-center gap-2">
                                                                {getActionIcon("drop_rows")} Drop Missing Rows
                                                            </span>
                                                        </SelectItem>
                                                    )}
                                                </SelectGroup>

                                                <SelectSeparator className="my-1.5" />

                                                {/* ── 3. OUTLIERS & DATA QUALITY ── */}
                                                <SelectGroup>
                                                    <SelectLabel>Outliers & Data Quality</SelectLabel>
                                                    {recommendedAction !== "cap_outliers" && (
                                                        <SelectItem value="cap_outliers">
                                                            <span className="flex items-center gap-2">
                                                                {getActionIcon("cap_outliers")} Cap Outliers (Tukey IQR Fences)
                                                            </span>
                                                        </SelectItem>
                                                    )}
                                                    {recommendedAction !== "flag_outliers" && (
                                                        <SelectItem value="flag_outliers">
                                                            <span className="flex items-center gap-2">
                                                                {getActionIcon("flag_outliers")} Flag Outliers (Indicator Column)
                                                            </span>
                                                        </SelectItem>
                                                    )}
                                                    {recommendedAction !== "log1p_transform" && (
                                                        <SelectItem value="log1p_transform">
                                                            <span className="flex items-center gap-2">
                                                                {getActionIcon("log1p_transform")} Log1p Transform (Reduce Skew)
                                                            </span>
                                                        </SelectItem>
                                                    )}
                                                    {recommendedAction !== "nullify_out_of_range" && (
                                                        <SelectItem value="nullify_out_of_range">
                                                            <span className="flex items-center gap-2">
                                                                {getActionIcon("nullify_out_of_range")} Nullify Out-of-Range Values
                                                            </span>
                                                        </SelectItem>
                                                    )}
                                                    {recommendedAction !== "nullify_sentinel" && (
                                                        <SelectItem value="nullify_sentinel">
                                                            <span className="flex items-center gap-2">
                                                                {getActionIcon("nullify_sentinel")} Nullify Sentinels (-999, 9999)
                                                            </span>
                                                        </SelectItem>
                                                    )}
                                                    {recommendedAction !== "clean_masked_nulls" && (
                                                        <SelectItem value="clean_masked_nulls">
                                                            <span className="flex items-center gap-2">
                                                                {getActionIcon("clean_masked_nulls")} Clean Masked Nulls (to null)
                                                            </span>
                                                        </SelectItem>
                                                    )}
                                                </SelectGroup>

                                                <SelectSeparator className="my-1.5" />

                                                {/* ── 4. FORMATTING, TYPES & TEXT ── */}
                                                <SelectGroup>
                                                    <SelectLabel>Formatting, Types & Text</SelectLabel>
                                                    {recommendedAction !== "coerce_numeric" && (
                                                        <SelectItem value="coerce_numeric">
                                                            <span className="flex items-center gap-2">
                                                                {getActionIcon("coerce_numeric")} Coerce to Numeric (Strip Currency/%)
                                                            </span>
                                                        </SelectItem>
                                                    )}
                                                    {recommendedAction !== "parse_date" && (
                                                        <SelectItem value="parse_date">
                                                            <span className="flex items-center gap-2">
                                                                {getActionIcon("parse_date")} Parse Date (ISO 8601)
                                                            </span>
                                                        </SelectItem>
                                                    )}
                                                    {recommendedAction !== "clean_text" && (
                                                        <SelectItem value="clean_text">
                                                            <span className="flex items-center gap-2">
                                                                {getActionIcon("clean_text")} Clean Text (Trim Whitespace)
                                                            </span>
                                                        </SelectItem>
                                                    )}
                                                    {recommendedAction !== "fix_mojibake" && (
                                                        <SelectItem value="fix_mojibake">
                                                            <span className="flex items-center gap-2">
                                                                {getActionIcon("fix_mojibake")} Fix Mojibake Encoding
                                                            </span>
                                                        </SelectItem>
                                                    )}
                                                    {recommendedAction !== "standardize_categories" && (
                                                        <SelectItem value="standardize_categories">
                                                            <span className="flex items-center gap-2">
                                                                {getActionIcon("standardize_categories")} Standardize Categories
                                                            </span>
                                                        </SelectItem>
                                                    )}
                                                    {recommendedAction !== "cast_boolean" && (
                                                        <SelectItem value="cast_boolean">
                                                            <span className="flex items-center gap-2">
                                                                {getActionIcon("cast_boolean")} Cast to Boolean
                                                            </span>
                                                        </SelectItem>
                                                    )}
                                                    {recommendedAction !== "keep_as_text" && (
                                                        <SelectItem value="keep_as_text">
                                                            <span className="flex items-center gap-2">
                                                                {getActionIcon("keep_as_text")} Preserve as String / Text
                                                            </span>
                                                        </SelectItem>
                                                    )}
                                                    {recommendedAction !== "nullify_invalid_regex" && (
                                                        <SelectItem value="nullify_invalid_regex">
                                                            <span className="flex items-center gap-2">
                                                                {getActionIcon("nullify_invalid_regex")} Nullify Invalid Format
                                                            </span>
                                                        </SelectItem>
                                                    )}
                                                </SelectGroup>

                                                <SelectSeparator className="my-1.5" />

                                                {/* ── 5. KEY DEDUPLICATION & COLUMNS ── */}
                                                <SelectGroup>
                                                    <SelectLabel>Key Deduplication & Columns</SelectLabel>
                                                    {recommendedAction !== "drop_duplicate_keys" && recommendedAction !== "keep_first" && (
                                                        <SelectItem value="drop_duplicate_keys">
                                                            <span className="flex items-center gap-2">
                                                                {getActionIcon("drop_duplicate_keys")} Keep First Row per Key (Deduplicate)
                                                            </span>
                                                        </SelectItem>
                                                    )}
                                                    {recommendedAction !== "drop_duplicate_rows" && (
                                                        <SelectItem value="drop_duplicate_rows">
                                                            <span className="flex items-center gap-2">
                                                                {getActionIcon("drop_duplicate_rows")} Drop Duplicate Rows
                                                            </span>
                                                        </SelectItem>
                                                    )}
                                                    {recommendedAction !== "hash_pii" && (
                                                        <SelectItem value="hash_pii">
                                                            <span className="flex items-center gap-2">
                                                                {getActionIcon("hash_pii")} Hash / Mask PII Column
                                                            </span>
                                                        </SelectItem>
                                                    )}
                                                    {recommendedAction !== "drop_column" && (
                                                        <SelectItem value="drop_column" className="text-rose-700 dark:text-rose-400">
                                                            <span className="flex items-center gap-2">
                                                                {getActionIcon("drop_column")} Drop Column
                                                            </span>
                                                        </SelectItem>
                                                    )}
                                                </SelectGroup>

                                                <SelectSeparator className="my-1.5" />

                                                {/* ── 6. DEFAULT (RAW AS IS) ── */}
                                                <SelectGroup>
                                                    <SelectLabel>No Action</SelectLabel>
                                                    <SelectItem value="default" className="text-stone-600 dark:text-zinc-400">
                                                        <span className="flex items-center gap-2">
                                                            {getActionIcon("default")} Ignore (Leave raw as is)
                                                        </span>
                                                    </SelectItem>
                                                </SelectGroup>

                                                {/* ── 7. DYNAMIC FALLBACK (FOR CUSTOM ACTIONS) ── */}
                                                {!KNOWN_ACTIONS.has(currentAction) && currentAction !== "default" && currentAction !== recommendedAction && (
                                                    <>
                                                        <SelectSeparator className="my-1.5" />
                                                        <SelectGroup>
                                                            <SelectLabel>Custom Selected Action</SelectLabel>
                                                            <SelectItem value={currentAction}>
                                                                <span className="flex items-center justify-between w-full gap-2">
                                                                    <span className="flex items-center gap-2">
                                                                        {getActionIcon(currentAction)} {getActionDisplayLabel(currentAction, issue)}
                                                                    </span>
                                                                    <span className="text-[10px] font-sans font-bold px-2 py-0.5 rounded-full bg-stone-200 text-stone-800 border border-stone-300 dark:bg-white/10 dark:text-zinc-300">
                                                                        Active
                                                                    </span>
                                                                </span>
                                                            </SelectItem>
                                                        </SelectGroup>
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
                <div className="bg-[#FAF8F5] border border-stone-300 dark:bg-[#0c0d12] dark:border-white/10 shadow-sm dark:shadow-premium rounded-2xl p-8 max-w-2xl mx-auto text-center mt-6">
                    <div className="w-16 h-16 bg-emerald-100 dark:bg-emerald-500/15 rounded-full flex items-center justify-center mx-auto mb-5 border border-emerald-300 dark:border-emerald-500/30 shadow-2xs">
                        <Check className="h-8 w-8 text-emerald-700 dark:text-emerald-400" />
                    </div>
                    <h3 className="text-2xl font-sans font-bold text-stone-900 dark:text-foreground tracking-[-0.025em]">Data Quality: Excellent</h3>
                    <p className="text-stone-600 dark:text-muted-foreground mt-2 max-w-md mx-auto text-sm leading-relaxed font-sans">
                        We have successfully run our pre-analysis checks and found no critical issues, missing values, or problematic distributions. Your dataset is clean and ready for deep analysis.
                    </p>
                    
                    <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 gap-4 text-left font-sans">
                        <div className="bg-stone-100/80 border border-stone-200 dark:bg-white/[0.03] dark:border-white/10 rounded-xl p-4 flex items-start gap-3">
                            <div className="bg-emerald-100 text-emerald-800 dark:bg-emerald-500/15 dark:text-emerald-400 p-1.5 rounded-md mt-0.5 border border-emerald-300 dark:border-emerald-500/25"><Check className="h-4 w-4 text-emerald-700 dark:text-emerald-400" /></div>
                            <div>
                                <h4 className="text-sm font-semibold text-stone-900 dark:text-foreground">Format Integrity</h4>
                                <p className="text-xs text-stone-600 dark:text-muted-foreground mt-1">All columns contain valid types.</p>
                            </div>
                        </div>
                        <div className="bg-stone-100/80 border border-stone-200 dark:bg-white/[0.03] dark:border-white/10 rounded-xl p-4 flex items-start gap-3">
                            <div className="bg-emerald-100 text-emerald-800 dark:bg-emerald-500/15 dark:text-emerald-400 p-1.5 rounded-md mt-0.5 border border-emerald-300 dark:border-emerald-500/25"><Check className="h-4 w-4 text-emerald-700 dark:text-emerald-400" /></div>
                            <div>
                                <h4 className="text-sm font-semibold text-stone-900 dark:text-foreground">Data Completeness</h4>
                                <p className="text-xs text-stone-600 dark:text-muted-foreground mt-1">No missing cells detected.</p>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* DATA PREVIEW TABLE */}
            {report.preview && report.preview.length > 0 && (
                <div className="border border-stone-300 bg-[#FAF8F5] dark:border-white/10 dark:bg-[#0c0d12] shadow-sm dark:shadow-premium rounded-2xl overflow-hidden font-sans">
                    <div className="bg-stone-100/80 px-5 py-3.5 border-b border-stone-200 dark:bg-white/[0.03] dark:border-white/10 flex items-center justify-between">
                        <h3 className="text-sm font-sans font-semibold text-foreground tracking-tight">Data Preview (First 5 Rows)</h3>
                        <span className="text-xs font-sans font-medium text-muted-foreground tabular-nums">{report.preview.length} sample rows</span>
                    </div>
                    <div className="overflow-x-auto custom-scrollbar">
                        <table className="w-full text-sm text-left border-collapse">
                            <thead className="bg-stone-100/40 text-foreground font-semibold dark:bg-white/[0.02]">
                                <tr>
                                    {Object.keys(report.preview[0]).map((header) => (
                                        <th key={header} className="px-4 py-3 border-b border-r border-stone-200 dark:border-white/10 whitespace-nowrap font-sans font-semibold text-xs uppercase tracking-wider text-muted-foreground last:border-r-0">
                                            {header}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-stone-200 bg-[#FAF8F5] dark:divide-white/5 dark:bg-[#0c0d12]">
                                {report.preview.map((row, rowPos) => (
                                    <tr key={`row_pos_${rowPos}`} className="border-b border-stone-200 dark:border-white/5 last:border-0 hover:bg-stone-100/50 dark:hover:bg-white/[0.02] transition-colors">
                                        {Object.entries(row).map(([header, cell]) => (
                                            <td key={`cell_${header}_${rowPos}`} className="px-4 py-2.5 border-r border-stone-200 dark:border-white/10 font-sans text-xs whitespace-nowrap max-w-[200px] truncate last:border-r-0 text-foreground/90 font-normal tabular-nums" title={String(cell)}>
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
                    className="w-full sm:w-auto min-w-[220px] rounded-xl transition-all duration-150 hover:-translate-y-0.5 active:scale-95 font-semibold text-sm py-6 bg-white hover:bg-zinc-200 text-black font-sans shadow-md cursor-pointer"
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

// Sparkline Histogram Component
const SparklineHistogram = ({ data }: { data: { count: number; label: string }[] }) => {
    if (!data || data.length === 0) return null;
    const max = Math.max(...data.map(d => d.count)) || 1;

    return (
        <div className="mt-3.5 pt-2 border-t border-stone-200 dark:border-white/5">
            <div className="flex items-center justify-between text-xs text-muted-foreground mb-1.5 font-sans">
                <span className="font-semibold text-foreground/80 flex items-center gap-1.5">
                    <BarChart2 className="w-3.5 h-3.5 text-[#800000] dark:text-primary" /> Distribution
                </span>
                <span className="text-[10px] font-mono">{data.length} bins</span>
            </div>
            <div className="flex items-end h-14 gap-[3px] w-full bg-stone-100 dark:bg-black/30 p-2 rounded-xl border border-stone-200 dark:border-white/5">
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
