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
    SlidersHorizontal
} from "lucide-react";
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
import type { InspectionReport, CleaningRulesMap, CleaningRule, QualityIssue } from "@/types/api";

const KNOWN_ACTIONS = new Set([
    "default",
    "cap_outliers",
    "replace_outliers_median",
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
    "fill_median",
    "fill_mean",
    "fill_mode",
    "fill_value",
    "fill_group_median",
    "add_missing_indicator",
    "log1p_transform",
    "hash_pii",
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

interface DataHealthCheckProps {
    report: InspectionReport;
    onContinue: (rules: CleaningRulesMap) => void;
    isProcessing: boolean;
}

const getTypeStyle = (type: string) => {
    const t = (type || "").toLowerCase();
    if (t.includes("int") || t.includes("float") || t.includes("num")) {
        return {
            icon: <Hash className="h-3 w-3 text-slate-600 dark:text-zinc-300" />,
            className: "bg-slate-100 text-slate-700 border-slate-200 dark:bg-white/5 dark:text-zinc-300 dark:border-white/10",
        };
    }
    if (t.includes("datetime") || t.includes("date") || t.includes("time")) {
        return {
            icon: <Calendar className="h-3 w-3 text-amber-600 dark:text-amber-400" />,
            className: "bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-500/25",
        };
    }
    if (t.includes("bool")) {
        return {
            icon: <CheckCircle2 className="h-3 w-3 text-emerald-600 dark:text-emerald-400" />,
            className: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-500/25",
        };
    }
    return {
        icon: <Type className="h-3 w-3 text-violet-600 dark:text-violet-400" />,
        className: "bg-violet-500/10 text-violet-700 dark:text-violet-400 border-violet-500/25",
    };
};

export const DataHealthCheck = ({ report, onContinue, isProcessing }: DataHealthCheckProps) => {
    // Automatically pre-populate rules with dynamic suggestions from the backend engine
    const [rules, setRules] = useState<CleaningRulesMap>(() => {
        const initial: CleaningRulesMap = {};
        (report?.issues || []).forEach(issue => {
            if (issue.column && issue.column !== "Multiple" && issue.action && issue.action !== "default") {
                const normalizedAction = (issue.action === "replace_outliers_median" || issue.action === "winsorize")
                    ? "cap_outliers"
                    : issue.action;
                initial[issue.column] = {
                    action: normalizedAction as CleaningRule["action"],
                    value: normalizedAction === "fill_value" ? "Unknown" : undefined,
                    params: (issue as any).params || {}
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
                        action: normalizedAction as CleaningRule["action"],
                        value: normalizedAction === "fill_value" ? "Unknown" : undefined,
                        params: (issue as any).params || {}
                    };
                }
            });
            return initial;
        });
    }, [report]);

    const handleActionChange = (column: string, action: string) => {
        const normalizedAction = (action === "replace_outliers_median" || action === "winsorize")
            ? "cap_outliers"
            : action;
        const matchingIssue = (report?.issues || []).find(i => i.column === column && i.action === action);
        setRules(prev => ({
            ...prev,
            [column]: {
                action: normalizedAction as CleaningRule["action"],
                value: normalizedAction === "fill_value" ? "Unknown" : undefined,
                params: matchingIssue?.params || prev[column]?.params || {}
            }
        }));
    };

    const handleApplyAllSuggestions = () => {
        const updated: CleaningRulesMap = {};
        (report?.issues || []).forEach(issue => {
            if (issue.column && issue.column !== "Multiple" && issue.action && issue.action !== "default") {
                const normalizedAction = (issue.action === "replace_outliers_median" || issue.action === "winsorize")
                    ? "cap_outliers"
                    : issue.action;
                updated[issue.column] = {
                    action: normalizedAction as CleaningRule["action"],
                    value: normalizedAction === "fill_value" ? "Unknown" : undefined,
                    params: (issue as any).params || {}
                };
            }
        });
        setRules(updated);
    };

    const handleApplyHighConfidenceOnly = () => {
        const updated: CleaningRulesMap = {};
        (report?.issues || []).forEach(issue => {
            if (
                issue.column &&
                issue.column !== "Multiple" &&
                issue.action &&
                issue.action !== "default" &&
                (issue.confidence_score ?? 100) >= 90
            ) {
                const normalizedAction = (issue.action === "replace_outliers_median" || issue.action === "winsorize")
                    ? "cap_outliers"
                    : issue.action;
                updated[issue.column] = {
                    action: normalizedAction as CleaningRule["action"],
                    value: normalizedAction === "fill_value" ? "Unknown" : undefined,
                    params: (issue as any).params || {}
                };
            }
        });
        setRules(updated);
    };

    const handleApplyConservative = () => {
        const updated: CleaningRulesMap = {};
        (report?.issues || []).forEach(issue => {
            if (
                issue.column &&
                issue.column !== "Multiple" &&
                issue.action &&
                ["fill_median", "fill_mean", "fill_mode", "fill_value", "clean_text", "clean_masked_nulls"].includes(issue.action)
            ) {
                updated[issue.column] = {
                    action: issue.action as CleaningRule["action"],
                    value: issue.action === "fill_value" ? "Unknown" : undefined,
                    params: (issue as any).params || {}
                };
            }
        });
        setRules(updated);
    };

    const handleResetAllToIgnore = () => {
        setRules({});
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
                        <div className="w-10 h-10 rounded-xl bg-amber-500/20 text-amber-600 dark:text-amber-400 border border-amber-500/30 flex items-center justify-center shrink-0 shadow-2xs">
                            <AlertTriangle className="h-5 w-5" />
                        </div>
                        <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                                <h4 className="text-sm font-sans font-semibold text-amber-900 dark:text-amber-300">
                                    {issue.type === 'partial_duplicates' ? "Ambiguous Data Detected" : "Quality Warning"}
                                </h4>
                                <span className="text-[10px] font-sans font-bold tabular-nums px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-900 dark:text-amber-300 border border-amber-500/30">
                                    {issue.count} rows
                                </span>
                            </div>
                            <p className="text-xs text-amber-800 dark:text-amber-200/90 mt-1 font-sans leading-relaxed">
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
                <div className="bg-white dark:bg-[#0e1017] border border-slate-200 dark:border-white/10 shadow-sm dark:shadow-none p-3.5 rounded-2xl space-y-3">
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-200/80 dark:border-white/5 pb-2.5">
                        <div className="flex items-center gap-2">
                            <SlidersHorizontal className="w-4 h-4 text-emerald-600 dark:text-primary" />
                            <span className="text-xs font-semibold text-slate-900 dark:text-foreground tracking-tight">
                                Auto-Suggestion Presets
                            </span>
                            <span className="text-[11px] font-sans font-medium tabular-nums px-2 py-0.5 rounded-full bg-slate-100 dark:bg-white/5 text-slate-600 dark:text-zinc-400 border border-slate-200 dark:border-white/10">
                                {columnsWithIssues.length} columns flagged
                            </span>
                        </div>
                        <div className="text-[11px] text-slate-500 dark:text-muted-foreground font-sans">
                            Distribution-aware actions calibrated with confidence scores & impact estimates.
                        </div>
                    </div>

                    <div className="flex flex-wrap items-center gap-2">
                        <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            onClick={handleApplyHighConfidenceOnly}
                            className="text-xs h-8 rounded-lg cursor-pointer bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-700 dark:text-primary border border-emerald-500/30 dark:border-primary/30 font-medium font-sans"
                        >
                            <CheckCircle2 className="w-3.5 h-3.5 mr-1.5" />
                            Apply High-Confidence ({highConfidenceCount})
                        </Button>

                        <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            onClick={handleApplyAllSuggestions}
                            className="text-xs h-8 rounded-lg cursor-pointer bg-slate-900 hover:bg-slate-800 text-white dark:bg-transparent dark:hover:bg-white/10 dark:text-foreground border border-slate-900 dark:border-white/15 font-sans font-medium"
                        >
                            <Layers className="w-3.5 h-3.5 mr-1.5 text-zinc-300 dark:text-zinc-400" />
                            Apply All Suggestions
                        </Button>

                        <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            onClick={handleApplyConservative}
                            className="text-xs h-8 rounded-lg cursor-pointer bg-slate-100 hover:bg-slate-200 text-slate-800 dark:bg-transparent dark:hover:bg-white/10 dark:text-zinc-300 border border-slate-200 dark:border-white/10 font-sans"
                        >
                            Conservative (Safe Only)
                        </Button>

                        <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            onClick={handleResetAllToIgnore}
                            className="text-xs h-8 rounded-lg cursor-pointer text-slate-500 hover:text-slate-900 dark:text-muted-foreground dark:hover:text-foreground font-sans ml-auto"
                        >
                            <RotateCcw className="w-3 h-3 mr-1" />
                            Reset to Raw
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

                    return (
                        <Card key={col.name} className="border border-slate-200 dark:border-white/10 bg-white dark:bg-[#0c0d12] rounded-2xl shadow-sm dark:shadow-premium overflow-hidden flex flex-col justify-between t-card-lift">
                            <CardHeader className="pb-3 bg-slate-50/50 dark:bg-white/[0.02] border-b border-slate-200/80 dark:border-white/5">
                                <div className="flex justify-between items-start gap-2">
                                    <CardTitle className="text-base font-sans font-semibold text-slate-900 dark:text-foreground truncate tracking-tight" title={col.name}>
                                        {col.name}
                                    </CardTitle>
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
                                            <TooltipContent className="rounded-xl border border-border/80 bg-popover text-popover-foreground shadow-xl">
                                                <p className="text-xs font-sans">Inferred Type: <code className="font-mono text-emerald-400 font-semibold">{col.inferred_type}</code> ({col.dtype})</p>
                                            </TooltipContent>
                                        </Tooltip>
                                    </TooltipProvider>
                                </div>

                                <CardDescription className="flex items-center gap-1.5 text-amber-700 dark:text-amber-400 font-sans text-xs font-semibold mt-1">
                                    <AlertTriangle className="h-3.5 w-3.5 text-amber-600 dark:text-amber-400 shrink-0" />
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
                                <div className="p-3 rounded-xl bg-slate-50/90 dark:bg-white/[0.03] border border-slate-200/90 dark:border-white/10 text-xs text-slate-600 dark:text-muted-foreground font-sans space-y-2">
                                    <div className="flex items-center justify-between gap-1 flex-wrap">
                                        <span className="text-[10px] text-slate-500 dark:text-muted-foreground uppercase font-semibold tracking-wider font-sans">
                                            Auto-Suggestion
                                        </span>
                                        <div className="flex items-center gap-1.5">
                                            {issue?.confidence_score !== undefined && (
                                                <span className="inline-flex items-center gap-1 text-[10px] font-sans font-semibold tabular-nums px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border border-emerald-500/20">
                                                    <CheckCircle2 className="w-2.5 h-2.5" />
                                                    {issue.confidence_score}%
                                                </span>
                                            )}
                                            {issue?.impact_estimate && (
                                                <span className="inline-flex items-center gap-1 text-[10px] font-sans font-semibold tabular-nums px-2 py-0.5 rounded-full bg-sky-500/10 text-sky-700 dark:text-sky-400 border border-sky-500/20">
                                                    <TrendingUp className="w-2.5 h-2.5" />
                                                    {issue.impact_estimate}
                                                </span>
                                            )}
                                        </div>
                                    </div>

                                    <div className="font-semibold text-slate-900 dark:text-foreground text-sm flex items-center gap-1.5">
                                        <Wrench className="w-3.5 h-3.5 text-emerald-600 dark:text-primary shrink-0" />
                                        <span className="truncate">{issue?.label || issue?.suggestion || "Ignore (Leave as is)"}</span>
                                    </div>

                                    {issue?.rationale && (
                                        <p className="text-[11px] text-slate-600 dark:text-muted-foreground leading-relaxed font-sans">
                                            {issue.rationale}
                                        </p>
                                    )}

                                    {/* Before vs After Visual Diff */}
                                    {issue?.before_sample && issue?.after_sample && issue.before_sample.length > 0 && (
                                        <div className="p-2.5 rounded-xl bg-slate-100/70 dark:bg-black/40 border border-slate-200/80 dark:border-white/5 space-y-1.5 font-mono text-[11px] mt-2.5">
                                            <div className="flex items-center justify-between text-[10px] text-slate-500 dark:text-zinc-400 uppercase tracking-wider font-sans font-semibold">
                                                <span>Preview</span>
                                                <span>Before → After</span>
                                            </div>
                                            <div className="space-y-1 pt-0.5">
                                                {issue.before_sample.slice(0, 2).map((b, idx) => {
                                                    const afterVal = issue.after_sample?.[idx];
                                                    return (
                                                        <div key={idx} className="flex items-center justify-between gap-2 text-xs py-1 px-2.5 rounded-lg bg-white dark:bg-white/[0.04] border border-slate-200/80 dark:border-white/5 shadow-2xs">
                                                            <span className="text-rose-600 dark:text-rose-400 font-medium truncate max-w-[110px]" title={String(b)}>
                                                                {b === null ? "null" : String(b)}
                                                            </span>
                                                            <ArrowRight className="w-3 h-3 text-slate-400 dark:text-zinc-500 shrink-0" />
                                                            <span className="text-emerald-700 dark:text-emerald-400 truncate max-w-[110px] font-semibold" title={String(afterVal)}>
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
                                {(() => {
                                    const currentAction = getActionForColumn(col.name);
                                    return (
                                        <Select
                                            value={currentAction}
                                            onValueChange={(val) => handleActionChange(col.name, val)}
                                        >
                                            <SelectTrigger className="w-full bg-white hover:bg-slate-50 dark:bg-[#131620] dark:hover:bg-[#1a1f2c] border border-slate-300 dark:border-white/10 text-slate-900 dark:text-foreground rounded-xl text-xs h-10 font-semibold hover:border-emerald-500/50 dark:hover:border-primary/40 focus:ring-emerald-500/20 shadow-xs transition-all cursor-pointer">
                                                <SelectValue placeholder="Select action…">
                                                    {getActionDisplayLabel(currentAction, issue)}
                                                </SelectValue>
                                            </SelectTrigger>
                                            <SelectContent className="rounded-xl border border-slate-200 dark:border-white/10 bg-white dark:bg-[#0e1017] text-popover-foreground shadow-2xl backdrop-blur-xl max-h-72">
                                                <SelectItem value="default" className="text-xs font-medium text-foreground cursor-pointer focus:bg-accent focus:text-foreground">
                                                    <span className="text-muted-foreground hover:text-foreground flex items-center gap-2">
                                                        <RotateCcw className="w-3.5 h-3.5 text-muted-foreground" /> Ignore (Leave raw as is)
                                                    </span>
                                                </SelectItem>

                                                {/* Cap Outliers (Tukey IQR) */}
                                                <SelectItem value="cap_outliers" className="text-xs font-medium text-foreground focus:bg-accent focus:text-foreground cursor-pointer">
                                                    <span className="flex items-center justify-between w-full gap-2">
                                                        <span className="flex items-center gap-2">
                                                            <Wrench className="w-3.5 h-3.5 text-amber-400" /> Cap Outliers (Tukey IQR Fences)
                                                        </span>
                                                        {(issue?.action === "cap_outliers" || issue?.action === "replace_outliers_median" || issue?.action === "winsorize") && (
                                                            <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                                                                Recommended
                                                            </span>
                                                        )}
                                                    </span>
                                                </SelectItem>

                                                {/* Flag Outliers */}
                                                <SelectItem value="flag_outliers" className="text-xs font-medium text-foreground focus:bg-accent focus:text-foreground cursor-pointer">
                                                    <span className="flex items-center justify-between w-full gap-2">
                                                        <span className="flex items-center gap-2">
                                                            <CheckCircle2 className="w-3.5 h-3.5 text-sky-400" /> Flag Outliers (Indicator Column)
                                                        </span>
                                                        {issue?.action === "flag_outliers" && (
                                                            <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                                                                Recommended
                                                            </span>
                                                        )}
                                                    </span>
                                                </SelectItem>

                                                {/* Drop Missing Rows */}
                                                <SelectItem value="drop_rows" className="text-xs font-medium text-rose-400 focus:text-rose-300 focus:bg-rose-500/15 cursor-pointer">
                                                    <span className="flex items-center justify-between w-full gap-2">
                                                        <span className="flex items-center gap-2">
                                                            <Trash2 className="w-3.5 h-3.5 text-rose-400" /> Drop Missing Rows
                                                        </span>
                                                        {issue?.action === "drop_rows" && (
                                                            <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                                                                Recommended
                                                            </span>
                                                        )}
                                                    </span>
                                                </SelectItem>

                                                {/* Coerce Numeric */}
                                                <SelectItem value="coerce_numeric" className="text-xs font-medium text-foreground focus:bg-accent focus:text-foreground cursor-pointer">
                                                    <span className="flex items-center justify-between w-full gap-2">
                                                        <span className="flex items-center gap-2">
                                                            <Hash className="w-3.5 h-3.5 text-sky-400" /> Coerce to Numeric (Strip Currency/%)
                                                        </span>
                                                        {issue?.action === "coerce_numeric" && (
                                                            <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                                                                Recommended
                                                            </span>
                                                        )}
                                                    </span>
                                                </SelectItem>

                                                {/* Clean Masked Nulls */}
                                                <SelectItem value="clean_masked_nulls" className="text-xs font-medium text-foreground focus:bg-accent focus:text-foreground cursor-pointer">
                                                    <span className="flex items-center justify-between w-full gap-2">
                                                        <span className="flex items-center gap-2">
                                                            <Wrench className="w-3.5 h-3.5 text-amber-400" /> Clean Masked Nulls (to null)
                                                        </span>
                                                        {issue?.action === "clean_masked_nulls" && (
                                                            <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                                                                Recommended
                                                            </span>
                                                        )}
                                                    </span>
                                                </SelectItem>

                                                {/* Clean Text Whitespace */}
                                                <SelectItem value="clean_text" className="text-xs font-medium text-foreground focus:bg-accent focus:text-foreground cursor-pointer">
                                                    <span className="flex items-center justify-between w-full gap-2">
                                                        <span className="flex items-center gap-2">
                                                            <Type className="w-3.5 h-3.5 text-violet-400" /> Clean Text (Trim Whitespace)
                                                        </span>
                                                        {issue?.action === "clean_text" && (
                                                            <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                                                                Recommended
                                                            </span>
                                                        )}
                                                    </span>
                                                </SelectItem>

                                                {/* Fix Mojibake */}
                                                <SelectItem value="fix_mojibake" className="text-xs font-medium text-foreground focus:bg-accent focus:text-foreground cursor-pointer">
                                                    <span className="flex items-center justify-between w-full gap-2">
                                                        <span className="flex items-center gap-2">
                                                            <Wrench className="w-3.5 h-3.5 text-emerald-400" /> Fix Mojibake Encoding
                                                        </span>
                                                        {issue?.action === "fix_mojibake" && (
                                                            <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                                                                Recommended
                                                            </span>
                                                        )}
                                                    </span>
                                                </SelectItem>

                                                {/* Standardize Categories */}
                                                <SelectItem value="standardize_categories" className="text-xs font-medium text-foreground focus:bg-accent focus:text-foreground cursor-pointer">
                                                    <span className="flex items-center justify-between w-full gap-2">
                                                        <span className="flex items-center gap-2">
                                                            <SlidersHorizontal className="w-3.5 h-3.5 text-indigo-400" /> Standardize Categories
                                                        </span>
                                                        {issue?.action === "standardize_categories" && (
                                                            <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                                                                Recommended
                                                            </span>
                                                        )}
                                                    </span>
                                                </SelectItem>

                                                {/* Cast to Boolean */}
                                                <SelectItem value="cast_boolean" className="text-xs font-medium text-foreground focus:bg-accent focus:text-foreground cursor-pointer">
                                                    <span className="flex items-center justify-between w-full gap-2">
                                                        <span className="flex items-center gap-2">
                                                            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" /> Cast to Boolean
                                                        </span>
                                                        {issue?.action === "cast_boolean" && (
                                                            <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                                                                Recommended
                                                            </span>
                                                        )}
                                                    </span>
                                                </SelectItem>

                                                {/* Preserve as Text */}
                                                <SelectItem value="keep_as_text" className="text-xs font-medium text-foreground focus:bg-accent focus:text-foreground cursor-pointer">
                                                    <span className="flex items-center justify-between w-full gap-2">
                                                        <span className="flex items-center gap-2">
                                                            <Type className="w-3.5 h-3.5 text-slate-500 dark:text-zinc-400" /> Preserve as String / Text
                                                        </span>
                                                        {issue?.action === "keep_as_text" && (
                                                            <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                                                                Recommended
                                                            </span>
                                                        )}
                                                    </span>
                                                </SelectItem>

                                                {/* Parse Date */}
                                                <SelectItem value="parse_date" className="text-xs font-medium text-foreground focus:bg-accent focus:text-foreground cursor-pointer">
                                                    <span className="flex items-center justify-between w-full gap-2">
                                                        <span className="flex items-center gap-2">
                                                            <Calendar className="w-3.5 h-3.5 text-cyan-400" /> Parse Date (ISO 8601)
                                                        </span>
                                                        {issue?.action === "parse_date" && (
                                                            <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                                                                Recommended
                                                            </span>
                                                        )}
                                                    </span>
                                                </SelectItem>

                                                {/* Nullify Sentinels */}
                                                <SelectItem value="nullify_sentinel" className="text-xs font-medium text-foreground focus:bg-accent focus:text-foreground cursor-pointer">
                                                    <span className="flex items-center justify-between w-full gap-2">
                                                        <span className="flex items-center gap-2">
                                                            <Wrench className="w-3.5 h-3.5 text-amber-400" /> Nullify Sentinels (-999, 9999)
                                                        </span>
                                                        {issue?.action === "nullify_sentinel" && (
                                                            <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                                                                Recommended
                                                            </span>
                                                        )}
                                                    </span>
                                                </SelectItem>

                                                {/* Nullify Out-of-Range */}
                                                <SelectItem value="nullify_out_of_range" className="text-xs font-medium text-foreground focus:bg-accent focus:text-foreground cursor-pointer">
                                                    <span className="flex items-center justify-between w-full gap-2">
                                                        <span className="flex items-center gap-2">
                                                            <AlertTriangle className="w-3.5 h-3.5 text-rose-400" /> Nullify Out-of-Range
                                                        </span>
                                                        {issue?.action === "nullify_out_of_range" && (
                                                            <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                                                                Recommended
                                                            </span>
                                                        )}
                                                    </span>
                                                </SelectItem>

                                                {/* Nullify Invalid Format */}
                                                <SelectItem value="nullify_invalid_regex" className="text-xs font-medium text-foreground focus:bg-accent focus:text-foreground cursor-pointer">
                                                    <span className="flex items-center justify-between w-full gap-2">
                                                        <span className="flex items-center gap-2">
                                                            <AlertTriangle className="w-3.5 h-3.5 text-rose-400" /> Nullify Invalid Format
                                                        </span>
                                                        {issue?.action === "nullify_invalid_regex" && (
                                                            <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                                                                Recommended
                                                            </span>
                                                        )}
                                                    </span>
                                                </SelectItem>

                                                {/* Drop Column */}
                                                <SelectItem value="drop_column" className="text-xs font-medium text-rose-400 focus:text-rose-300 focus:bg-rose-500/15 cursor-pointer">
                                                    <span className="flex items-center justify-between w-full gap-2">
                                                        <span className="flex items-center gap-2">
                                                            <Trash2 className="w-3.5 h-3.5 text-rose-400" /> Drop Column
                                                        </span>
                                                        {issue?.action === "drop_column" && (
                                                            <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                                                                Recommended
                                                            </span>
                                                        )}
                                                    </span>
                                                </SelectItem>

                                                {/* Imputations */}
                                                <SelectItem value="fill_median" className="text-xs font-medium text-foreground focus:bg-accent focus:text-foreground cursor-pointer">
                                                    <span className="flex items-center justify-between w-full gap-2">
                                                        <span className="flex items-center gap-2">
                                                            <Wrench className="w-3.5 h-3.5 text-slate-500 dark:text-zinc-400" /> Fill with Median
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
                                                            <Wrench className="w-3.5 h-3.5 text-slate-500 dark:text-zinc-400" /> Fill with Average
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
                                                            <Wrench className="w-3.5 h-3.5 text-slate-500 dark:text-zinc-400" /> Fill with Most Frequent
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
                                                            <Wrench className="w-3.5 h-3.5 text-slate-500 dark:text-zinc-400" /> Fill with &ldquo;Unknown&rdquo;
                                                        </span>
                                                        {issue?.action === "fill_value" && (
                                                            <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                                                                Recommended
                                                            </span>
                                                        )}
                                                    </span>
                                                </SelectItem>

                                                <SelectItem value="fill_group_median" className="text-xs font-medium text-foreground focus:bg-accent focus:text-foreground cursor-pointer">
                                                    <span className="flex items-center justify-between w-full gap-2">
                                                        <span className="flex items-center gap-2">
                                                            <Layers className="w-3.5 h-3.5 text-purple-400" /> Fill with Group Median
                                                        </span>
                                                        {issue?.action === "fill_group_median" && (
                                                            <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                                                                Recommended
                                                            </span>
                                                        )}
                                                    </span>
                                                </SelectItem>

                                                <SelectItem value="add_missing_indicator" className="text-xs font-medium text-foreground focus:bg-accent focus:text-foreground cursor-pointer">
                                                    <span className="flex items-center justify-between w-full gap-2">
                                                        <span className="flex items-center gap-2">
                                                            <CheckCircle2 className="w-3.5 h-3.5 text-sky-400" /> Add Missing Indicator Column
                                                        </span>
                                                        {issue?.action === "add_missing_indicator" && (
                                                            <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                                                                Recommended
                                                            </span>
                                                        )}
                                                    </span>
                                                </SelectItem>

                                                <SelectItem value="log1p_transform" className="text-xs font-medium text-foreground focus:bg-accent focus:text-foreground cursor-pointer">
                                                    <span className="flex items-center justify-between w-full gap-2">
                                                        <span className="flex items-center gap-2">
                                                            <BarChart2 className="w-3.5 h-3.5 text-blue-400" /> Log1p Transform (Reduce Skew)
                                                        </span>
                                                        {issue?.action === "log1p_transform" && (
                                                            <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                                                                Recommended
                                                            </span>
                                                        )}
                                                    </span>
                                                </SelectItem>

                                                <SelectItem value="hash_pii" className="text-xs font-medium text-foreground focus:bg-accent focus:text-foreground cursor-pointer">
                                                    <span className="flex items-center justify-between w-full gap-2">
                                                        <span className="flex items-center gap-2">
                                                            <ShieldAlert className="w-3.5 h-3.5 text-emerald-400" /> Hash / Mask PII Column
                                                        </span>
                                                        {issue?.action === "hash_pii" && (
                                                            <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                                                                Recommended
                                                            </span>
                                                        )}
                                                    </span>
                                                </SelectItem>

                                                {/* Dynamic Fallback if custom action */}
                                                {!KNOWN_ACTIONS.has(currentAction) && currentAction !== "default" && (
                                                    <SelectItem value={currentAction} className="text-xs font-medium text-foreground cursor-pointer">
                                                        <span className="flex items-center justify-between w-full gap-2">
                                                            <span className="flex items-center gap-2">
                                                                <Wrench className="w-3.5 h-3.5 text-amber-400" /> {getActionDisplayLabel(currentAction, issue)}
                                                            </span>
                                                            <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                                                                Recommended
                                                            </span>
                                                        </span>
                                                    </SelectItem>
                                                )}
                                            </SelectContent>
                                        </Select>
                                    );
                                })()}
                            </CardFooter>
                        </Card>
                    );
                })}
            </div>

            {report.issues.length === 0 && (
                <div className="bg-white dark:bg-[#0c0d12] border border-slate-200 dark:border-white/10 shadow-sm dark:shadow-premium rounded-2xl p-8 max-w-2xl mx-auto text-center mt-6">
                    <div className="w-16 h-16 bg-emerald-500/15 rounded-full flex items-center justify-center mx-auto mb-5 border border-emerald-500/30 shadow-2xs">
                        <Check className="h-8 w-8 text-emerald-600 dark:text-emerald-400" />
                    </div>
                    <h3 className="text-2xl font-sans font-bold text-slate-900 dark:text-foreground tracking-[-0.025em]">Data Quality: Excellent</h3>
                    <p className="text-slate-600 dark:text-muted-foreground mt-2 max-w-md mx-auto text-sm leading-relaxed font-sans">
                        We have successfully run our pre-analysis checks and found no critical issues, missing values, or problematic distributions. Your dataset is clean and ready for deep analysis.
                    </p>
                    
                    <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 gap-4 text-left font-sans">
                        <div className="bg-slate-50 dark:bg-white/[0.03] border border-slate-200 dark:border-white/10 rounded-xl p-4 flex items-start gap-3">
                            <div className="bg-emerald-500/15 p-1.5 rounded-md mt-0.5 border border-emerald-500/25"><Check className="h-4 w-4 text-emerald-600 dark:text-emerald-400" /></div>
                            <div>
                                <h4 className="text-sm font-semibold text-slate-900 dark:text-foreground">Format Integrity</h4>
                                <p className="text-xs text-slate-500 dark:text-muted-foreground mt-1">All columns contain valid types.</p>
                            </div>
                        </div>
                        <div className="bg-slate-50 dark:bg-white/[0.03] border border-slate-200 dark:border-white/10 rounded-xl p-4 flex items-start gap-3">
                            <div className="bg-emerald-500/15 p-1.5 rounded-md mt-0.5 border border-emerald-500/25"><Check className="h-4 w-4 text-emerald-600 dark:text-emerald-400" /></div>
                            <div>
                                <h4 className="text-sm font-semibold text-slate-900 dark:text-foreground">Data Completeness</h4>
                                <p className="text-xs text-slate-500 dark:text-muted-foreground mt-1">No missing cells detected.</p>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* DATA PREVIEW TABLE */}
            {report.preview && report.preview.length > 0 && (
                <div className="border border-slate-200 dark:border-white/10 bg-white dark:bg-[#0c0d12] shadow-sm dark:shadow-premium rounded-2xl overflow-hidden font-sans">
                    <div className="bg-slate-50/70 dark:bg-white/[0.03] px-5 py-3.5 border-b border-slate-200 dark:border-white/10 flex items-center justify-between">
                        <h3 className="text-sm font-sans font-semibold text-slate-900 dark:text-foreground tracking-tight">Data Preview (First 5 Rows)</h3>
                        <span className="text-xs font-sans font-medium text-slate-500 dark:text-muted-foreground tabular-nums">{report.preview.length} sample rows</span>
                    </div>
                    <div className="overflow-x-auto custom-scrollbar">
                        <table className="w-full text-sm text-left border-collapse">
                            <thead className="bg-slate-50/50 dark:bg-white/[0.02] text-slate-900 dark:text-foreground font-semibold">
                                <tr>
                                    {Object.keys(report.preview[0]).map((header) => (
                                        <th key={header} className="px-4 py-3 border-b border-r border-slate-200 dark:border-white/10 whitespace-nowrap font-sans font-semibold text-xs uppercase tracking-wider text-slate-600 dark:text-muted-foreground last:border-r-0">
                                            {header}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-200/80 dark:divide-white/5 bg-white dark:bg-[#0c0d12]">
                                {report.preview.map((row, rowPos) => (
                                    <tr key={`row_pos_${rowPos}`} className="border-b border-slate-100 dark:border-white/5 last:border-0 hover:bg-slate-50/60 dark:hover:bg-white/[0.02] transition-colors">
                                        {Object.entries(row).map(([header, cell]) => (
                                            <td key={`cell_${header}_${rowPos}`} className="px-4 py-2.5 border-r border-slate-100 dark:border-white/10 font-sans text-xs whitespace-nowrap max-w-[200px] truncate last:border-r-0 text-slate-800 dark:text-foreground/90 font-normal tabular-nums" title={String(cell)}>
                                                {cell === null ? <span className="text-slate-400 dark:text-muted-foreground italic font-sans">null</span> : String(cell)}
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
                    className="w-full sm:w-auto min-w-[220px] rounded-xl transition-all duration-150 hover:-translate-y-0.5 active:scale-95 font-semibold text-sm py-6 bg-slate-950 hover:bg-slate-800 text-white dark:bg-white dark:hover:bg-zinc-200 dark:text-black font-sans shadow-md cursor-pointer"
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
        <div className="mt-3.5 pt-2 border-t border-slate-200/80 dark:border-white/5">
            <div className="flex items-center justify-between text-xs text-slate-500 dark:text-muted-foreground mb-1.5 font-sans">
                <span className="font-semibold text-slate-900 dark:text-foreground/80 flex items-center gap-1.5">
                    <BarChart2 className="w-3.5 h-3.5 text-emerald-600 dark:text-primary" /> Distribution
                </span>
                <span className="text-[10px] font-mono text-slate-400 dark:text-muted-foreground">{data.length} bins</span>
            </div>
            <div className="flex items-end h-14 gap-[3px] w-full bg-slate-100/90 dark:bg-black/30 p-2 rounded-xl border border-slate-200/80 dark:border-white/5">
                {data.map((d) => (
                    <TooltipProvider key={`hist_bin_${d.label}`}>
                        <Tooltip>
                            <TooltipTrigger asChild>
                                <div
                                    className="flex-1 bg-gradient-to-t from-emerald-500/30 to-emerald-600 hover:from-emerald-500 hover:to-emerald-700 dark:from-primary/30 dark:to-primary/80 dark:hover:from-primary dark:hover:to-primary/90 transition-all rounded-t-xs cursor-pointer shadow-2xs"
                                    style={{ height: `${Math.max(12, (d.count / max) * 100)}%` }}
                                />
                            </TooltipTrigger>
                            <TooltipContent className="rounded-xl border border-border/80 bg-popover text-popover-foreground shadow-xl">
                                <p className="text-xs font-sans text-popover-foreground">{d.label}: <strong className="font-mono text-emerald-600 dark:text-primary">{d.count}</strong></p>
                            </TooltipContent>
                        </Tooltip>
                    </TooltipProvider>
                ))}
            </div>
        </div>
    );
};
