import { useState } from "react";
import { Check, AlertTriangle, Play, ShieldAlert, Sparkles, Trash2, Wrench, BarChart2 } from "lucide-react";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
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

export const DataHealthCheck = ({ report, onContinue, isProcessing }: DataHealthCheckProps) => {
    const [rules, setRules] = useState<CleaningRulesMap>({});

    const handleActionChange = (column: string, action: string) => {
        setRules(prev => ({
            ...prev,
            [column]: {
                action: action as CleaningRule["action"],
                value: action === "fill_value" ? "Unknown" : undefined
            }
        }));
    };

    const getActionForColumn = (column: string) => {
        return rules[column]?.action || "default"; // "default" means auto-pilot
    };

    const handleSubmit = () => {
        onContinue(rules);
    };

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
            {(report?.issues || []).filter(i => i.column === "Multiple").map((issue, idx) => (
                <div key={idx} className="bg-gradient-to-r from-amber-500/10 via-amber-500/5 to-transparent border border-amber-500/30 p-5 rounded-2xl flex items-start gap-4 shadow-sm animate-in fade-in duration-300">
                    <div className="w-10 h-10 rounded-xl bg-amber-500/15 text-amber-700 border border-amber-500/30 flex items-center justify-center shrink-0 shadow-2xs">
                        <AlertTriangle className="h-5 w-5" />
                    </div>
                    <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                            <h4 className="text-sm font-display font-bold text-amber-900 uppercase tracking-wide">
                                {issue.type === 'partial_duplicates' ? "Ambiguous Data Detected" : "Quality Warning"}
                            </h4>
                            <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-800 border border-amber-500/30">
                                {issue.count} rows
                            </span>
                        </div>
                        <p className="text-xs text-amber-800/90 mt-1 font-sans leading-relaxed">
                            {issue.type === 'partial_duplicates'
                                ? `Found ${issue.count} rows that look identical but have different IDs (Partial Duplicates). Review rules below.`
                                : issue.suggestion}
                        </p>
                    </div>
                </div>
            ))}

            {/* Column Health Grid */}
            <div className="grid gap-5 md:grid-cols-2 lg:grid-cols-3">
                {(report?.columns || []).map((col) => {
                    const issue = (report?.issues || []).find(i => i.column === col.name);
                    const hasIssue = col.missing_count > 0 || (issue && ['outliers', 'high_cardinality', 'class_imbalance'].includes(issue.type));

                    if (!hasIssue) return null;

                    return (
                        <Card key={col.name} className="border border-border/80 bg-card hover:border-primary/30 hover:shadow-lg transition-all duration-300 rounded-2xl shadow-premium overflow-hidden flex flex-col justify-between">
                            <CardHeader className="pb-3 bg-muted/10 border-b border-border/40">
                                <div className="flex justify-between items-start gap-2">
                                    <CardTitle className="text-base font-display font-bold text-foreground truncate" title={col.name}>
                                        {col.name}
                                    </CardTitle>
                                    <TooltipProvider>
                                        <Tooltip>
                                            <TooltipTrigger>
                                                <Badge variant="outline" className="text-[10px] font-mono px-2 py-0.5 rounded-md bg-white border-border/80 text-foreground shrink-0">
                                                    {col.inferred_type}
                                                </Badge>
                                            </TooltipTrigger>
                                            <TooltipContent>
                                                <p className="text-xs font-mono">Inferred Type: {col.inferred_type}</p>
                                            </TooltipContent>
                                        </Tooltip>
                                    </TooltipProvider>
                                </div>
                                <CardDescription className="flex items-center gap-1.5 text-amber-700 font-sans text-xs font-semibold mt-1">
                                    <AlertTriangle className="h-3.5 w-3.5 text-amber-600 shrink-0" />
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
                                <div className="p-2.5 rounded-xl bg-muted/30 border border-border/60 text-xs text-muted-foreground font-sans">
                                    <span className="text-[11px] text-muted-foreground uppercase font-semibold block tracking-wider">Auto-suggestion</span>
                                    <span className="font-semibold text-foreground block truncate mt-0.5">
                                        {issue?.suggestion || "Ignore (Leave as is)"}
                                    </span>
                                </div>
                                {col.distribution && <SparklineHistogram data={col.distribution} />}
                            </CardContent>

                            <CardFooter className="pt-0 pb-4 px-4">
                                <Select
                                    value={getActionForColumn(col.name)}
                                    onValueChange={(val) => handleActionChange(col.name, val)}
                                >
                                    <SelectTrigger className="w-full bg-white border-border/80 rounded-xl text-xs h-10 font-medium hover:border-primary/40 focus:ring-primary/20 shadow-2xs">
                                        <SelectValue placeholder="Select action…" />
                                    </SelectTrigger>
                                    <SelectContent className="rounded-xl border-border bg-white shadow-xl">
                                        <SelectItem value="default" className="text-xs font-medium">
                                            <span className="text-muted-foreground flex items-center gap-2">
                                                <Sparkles className="w-3.5 h-3.5 text-amber-500" /> Ignore (Leave as is)
                                            </span>
                                        </SelectItem>
                                        <SelectItem value="drop_rows" className="text-xs font-medium text-rose-700">
                                            <span className="flex items-center gap-2">
                                                <Trash2 className="w-3.5 h-3.5 text-rose-500" /> Drop Rows
                                            </span>
                                        </SelectItem>

                                        {col.inferred_type === 'numeric' && (
                                            <>
                                                {issue?.type !== 'outliers' && (
                                                    <>
                                                        <SelectItem value="fill_median" className="text-xs font-medium">
                                                            <span className="flex items-center gap-2">
                                                                <Wrench className="w-3.5 h-3.5 text-primary" /> Fill with Median
                                                            </span>
                                                        </SelectItem>
                                                        <SelectItem value="fill_mean" className="text-xs font-medium">
                                                            <span className="flex items-center gap-2">
                                                                <Wrench className="w-3.5 h-3.5 text-primary" /> Fill with Average
                                                            </span>
                                                        </SelectItem>
                                                    </>
                                                )}
                                                {issue?.type === 'outliers' && (
                                                    <SelectItem value="replace_outliers_median" className="text-xs font-medium">
                                                        <span className="flex items-center gap-2">
                                                            <Wrench className="w-3.5 h-3.5 text-amber-600" /> Cap Outliers (Median)
                                                        </span>
                                                    </SelectItem>
                                                )}
                                            </>
                                        )}
                                        {col.inferred_type !== 'numeric' && (
                                            <>
                                                <SelectItem value="fill_mode" className="text-xs font-medium">
                                                    <span className="flex items-center gap-2">
                                                        <Wrench className="w-3.5 h-3.5 text-primary" /> Fill with Most Frequent
                                                    </span>
                                                </SelectItem>
                                                <SelectItem value="fill_value" className="text-xs font-medium">
                                                    <span className="flex items-center gap-2">
                                                        <Wrench className="w-3.5 h-3.5 text-primary" /> Fill with "Unknown"
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
                    <div className="w-16 h-16 bg-emerald-100 rounded-full flex items-center justify-center mx-auto mb-5 border border-emerald-200 shadow-2xs">
                        <Check className="h-8 w-8 text-emerald-600" />
                    </div>
                    <h3 className="text-2xl font-display font-bold text-foreground">Data Quality: Excellent</h3>
                    <p className="text-muted-foreground mt-2 max-w-md mx-auto text-sm leading-relaxed">
                        We have successfully run our pre-analysis checks and found no critical issues, missing values, or problematic distributions. Your dataset is clean and ready for deep analysis.
                    </p>
                    
                    <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 gap-4 text-left">
                        <div className="bg-muted/30 border border-border/60 rounded-xl p-4 flex items-start gap-3">
                            <div className="bg-emerald-100 p-1.5 rounded-md mt-0.5"><Check className="h-4 w-4 text-emerald-600" /></div>
                            <div>
                                <h4 className="text-sm font-semibold text-foreground">Format Integrity</h4>
                                <p className="text-xs text-muted-foreground mt-1">All columns contain valid types.</p>
                            </div>
                        </div>
                        <div className="bg-muted/30 border border-border/60 rounded-xl p-4 flex items-start gap-3">
                            <div className="bg-emerald-100 p-1.5 rounded-md mt-0.5"><Check className="h-4 w-4 text-emerald-600" /></div>
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
                    <div className="overflow-x-auto">
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
                            <tbody className="divide-y divide-border/60">
                                {report.preview.map((row, idx) => (
                                    <tr key={idx} className="border-b border-border/40 last:border-0 hover:bg-primary/[0.02] transition-colors">
                                        {Object.values(row).map((cell: unknown, cIdx) => (
                                            <td key={cIdx} className="px-4 py-2.5 border-r border-border/60 font-mono text-xs whitespace-nowrap max-w-[200px] truncate last:border-r-0 text-foreground/90" title={String(cell)}>
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
                    className="w-full sm:w-auto min-w-[220px] rounded-xl shadow-premium transition-all duration-150 hover:-translate-y-0.5 active:scale-95 font-semibold text-base py-6 bg-primary text-primary-foreground hover:bg-primary/90 cursor-pointer"
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
                {data.map((d, i) => (
                    <TooltipProvider key={i}>
                        <Tooltip>
                            <TooltipTrigger asChild>
                                <div
                                    className="flex-1 bg-gradient-to-t from-primary/30 to-primary/80 hover:from-primary hover:to-primary/90 transition-all rounded-t-xs cursor-pointer shadow-2xs"
                                    style={{ height: `${Math.max(12, (d.count / max) * 100)}%` }}
                                />
                            </TooltipTrigger>
                            <TooltipContent className="rounded-xl border-border bg-white shadow-xl">
                                <p className="text-xs font-sans">{d.label}: <strong className="font-mono text-primary">{d.count}</strong></p>
                            </TooltipContent>
                        </Tooltip>
                    </TooltipProvider>
                ))}
            </div>
        </div>
    );
};
