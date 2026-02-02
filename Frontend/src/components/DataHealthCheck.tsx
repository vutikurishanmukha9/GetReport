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
            [column]: { action: action as any }
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

            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {report.columns.map((col) => {
                    const hasIssue = col.missing_count > 0;
                    const issue = report.issues.find(i => i.column === col.name);

                    if (!hasIssue) return null; // Only show columns with issues for now? Or show all?
                    // Showing all might be noisy. Let's filter to issues.

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
                                    {col.missing_count} missing ({col.missing_percentage}%)
                                </CardDescription>
                            </CardHeader>

                            <CardContent className="pb-3">
                                <div className="space-y-1 text-sm text-muted-foreground">
                                    <p>Auto-suggestion: <b>{issue?.suggestion || "Ignore"}</b></p>
                                </div>
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
                                            <span className="text-muted-foreground">Auto-Fix (Recommended)</span>
                                        </SelectItem>
                                        <SelectItem value="drop_rows">Drop Rows</SelectItem>

                                        {col.inferred_type === 'numeric' && (
                                            <SelectItem value="fill_mean">Fill with Average</SelectItem>
                                        )}
                                        {col.inferred_type !== 'numeric' && (
                                            <SelectItem value="fill_value">Fill with "Unknown"</SelectItem>
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
