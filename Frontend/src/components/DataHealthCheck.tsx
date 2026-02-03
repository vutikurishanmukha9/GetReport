import { useState } from "react";
import { Check, AlertTriangle, Play, HelpCircle } from "lucide-react";
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
import type { InspectionReport, CleaningRulesMap } from "@/types/api";

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
                action: action as any,
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
        <div className="space-y-6 max-w-4xl mx-auto animate-in fade-in slide-in-from-bottom-4 duration-500">

            {/* Header Section */}
            <div className="text-center space-y-2">
                <h2 className="text-2xl font-bold tracking-tight">Data Health Check</h2>
                <p className="text-muted-foreground">
                    We found some issues. Review them before we analyze your data.
                </p>
            </div>

            {/* ─── Global Warnings ─── */}
            {report.issues.filter(i => i.column === "Multiple").map((issue, idx) => (
                <div key={idx} className="bg-orange-50 border-l-4 border-orange-500 p-4 mb-6 rounded-r-md flex items-start">
                    <AlertTriangle className="h-5 w-5 text-orange-600 mt-0.5 mr-3" />
                    <div>
                        <h4 className="text-sm font-bold text-orange-800 uppercase tracking-wide">
                            {issue.type === 'partial_duplicates' ? "Ambiguous Data Detected" : "Warning"}
                        </h4>
                        <p className="text-sm text-orange-700 mt-1">
                            {issue.type === 'partial_duplicates'
                                ? `Found ${issue.count} rows that look identical but have different IDs (Partial Duplicates).`
                                : issue.suggestion}
                        </p>
                    </div>
                </div>
            ))}

            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {report.columns.map((col) => {
                    const issue = report.issues.find(i => i.column === col.name);
                    // Special case: partial_duplicates has column="Multiple", but we want to show it somewhere.
                    // Actually, partial_duplicates is a dataset-level issue, not column-specific.
                    // We should render it separately or attach to "Multiple"?
                    // Current logic iterates columns. Let's create a global warnings section.

                    const hasIssue = col.missing_count > 0 || (issue && ['outliers', 'high_cardinality', 'class_imbalance'].includes(issue.type));

                    if (!hasIssue) return null;

                    return (
                        <Card key={col.name} className="border-l-4 border-l-yellow-500 shadow-sm">
                            <CardHeader className="pb-3">
                                <div className="flex justify-between items-start">
                                    <CardTitle className="text-lg font-medium truncate" title={col.name}>
                                        {col.name}
                                    </CardTitle>
                                    <TooltipProvider>
                                        <Tooltip>
                                            <TooltipTrigger>
                                                <Badge variant="outline" className="text-xs">
                                                    {col.inferred_type}
                                                </Badge>
                                            </TooltipTrigger>
                                            <TooltipContent>
                                                <p>Inferred Type: {col.inferred_type}</p>
                                            </TooltipContent>
                                        </Tooltip>
                                    </TooltipProvider>
                                </div>
                                <CardDescription className="flex items-center gap-1 text-yellow-600">
                                    <AlertTriangle className="h-3 w-3" />
                                    {issue?.type === 'outliers'
                                        ? `${issue.count} outliers detected`
                                        : issue?.type === 'high_cardinality'
                                            ? `${issue.count} unique values`
                                            : issue?.type === 'class_imbalance'
                                                ? `Top category dominates`
                                                : `${col.missing_count} missing (${col.missing_percentage}%)`
                                    }
                                </CardDescription>
                            </CardHeader>

                            <CardContent className="pb-3">
                                <div className="space-y-1 text-sm text-muted-foreground">
                                    <p>Auto-suggestion: <b>{issue?.suggestion || "Ignore"}</b></p>
                                </div>
                                {col.distribution && <SparklineHistogram data={col.distribution} />}
                            </CardContent>

                            <CardFooter className="pt-0">
                                <Select
                                    value={getActionForColumn(col.name)}
                                    onValueChange={(val) => handleActionChange(col.name, val)}
                                >
                                    <SelectTrigger className="w-full">
                                        <SelectValue placeholder="Select action..." />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="default">
                                            <span className="text-muted-foreground">Ignore (Leave as is)</span>
                                        </SelectItem>
                                        <SelectItem value="drop_rows">Drop Rows</SelectItem>

                                        {col.inferred_type === 'numeric' && (
                                            <>
                                                <SelectItem value="fill_median">Fill with Median (Robust)</SelectItem>
                                                <SelectItem value="fill_mean">Fill with Average</SelectItem>
                                                <SelectItem value="replace_outliers_median">Replace Outliers (Median)</SelectItem>
                                            </>
                                        )}
                                        {col.inferred_type !== 'numeric' && (
                                            <>
                                                <SelectItem value="fill_mode">Fill with Most Frequent</SelectItem>
                                                <SelectItem value="fill_value">Fill with "Unknown"</SelectItem>
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
                <div className="text-center p-8 border-2 border-dashed rounded-lg bg-muted/50">
                    <Check className="h-12 w-12 text-green-500 mx-auto mb-4" />
                    <h3 className="text-lg font-medium">Your data looks clean!</h3>
                    <p className="text-muted-foreground">No critical issues found.</p>
                </div>
            )}


            {/* ─── DATA PREVIEW (Added heavily requested feature) ─── */}
            {report.preview && report.preview.length > 0 && (
                <div className="border rounded-md shadow-sm overflow-hidden">
                    <div className="bg-muted/50 px-4 py-3 border-b">
                        <h3 className="text-sm font-medium">Data Preview (First 5 Rows)</h3>
                    </div>
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm text-left">
                            <thead className="bg-muted/20 text-muted-foreground font-medium">
                                <tr>
                                    {Object.keys(report.preview[0]).map((header) => (
                                        <th key={header} className="px-4 py-2 border-b whitespace-nowrap">
                                            {header}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {report.preview.map((row, idx) => (
                                    <tr key={idx} className="border-b last:border-0 hover:bg-muted/10">
                                        {Object.values(row).map((cell: any, cIdx) => (
                                            <td key={cIdx} className="px-4 py-2 whitespace-nowrap max-w-[200px] truncate" title={String(cell)}>
                                                {cell === null ? <span className="text-muted-foreground italic">null</span> : String(cell)}
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
                    className="w-full sm:w-auto min-w-[200px]"
                >
                    {isProcessing ? (
                        "Processing..."
                    ) : (
                        <>
                            <Play className="mr-2 h-4 w-4" />
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
        <div className="mt-3">
            <p className="text-xs text-muted-foreground mb-1">Distribution (Mugshot):</p>
            <div className="flex items-end h-12 gap-[2px] w-full">
                {data.map((d, i) => (
                    <TooltipProvider key={i}>
                        <Tooltip>
                            <TooltipTrigger asChild>
                                <div
                                    className="flex-1 bg-primary/20 hover:bg-primary/50 transition-colors rounded-t-sm"
                                    style={{ height: `${(d.count / max) * 100}%` }}
                                />
                            </TooltipTrigger>
                            <TooltipContent>
                                <p className="text-xs font-mono">{d.label}: <strong>{d.count}</strong></p>
                            </TooltipContent>
                        </Tooltip>
                    </TooltipProvider>
                ))}
            </div>
        </div>
    );
};
