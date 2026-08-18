import { useState } from "react";
import { ArrowLeft, ArrowRight, FileSpreadsheet, Hash, Calendar, Type, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { DatasetInfo, CleaningReport, AnalysisResult } from "@/types/api";

interface DataPreviewProps {
  info: DatasetInfo;
  cleaningReport: CleaningReport;
  analysis: AnalysisResult;
  onGenerateReport: () => void;
  onBack: () => void;
}

const getTypeIcon = (type: string) => {
  if (type.includes("int") || type.includes("float")) return <Hash className="h-3 w-3" />;
  if (type.includes("datetime")) return <Calendar className="h-3 w-3" />;
  return <Type className="h-3 w-3" />;
};

const getTypeBadgeVariant = (type: string): "default" | "secondary" | "outline" => {
  if (type.includes("int") || type.includes("float")) return "default";
  if (type.includes("datetime")) return "secondary";
  return "outline";
};

export const DataPreview = ({ info, cleaningReport, analysis, onGenerateReport, onBack }: DataPreviewProps) => {
  const [activeTab, setActiveTab] = useState("preview");

  return (
    <div className="max-w-6xl mx-auto space-y-6 sm:space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-400">

      {/* Header Section */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-start sm:items-center gap-3 sm:gap-4">
          <div className="flex h-10 w-10 sm:h-12 sm:w-12 shrink-0 items-center justify-center rounded-xl bg-primary/10 border border-primary/20 shadow-premium">
            <FileSpreadsheet className="h-5 w-5 sm:h-6 sm:w-6 text-primary" />
          </div>
          <div className="min-w-0">
            <h1 className="text-2xl sm:text-3xl font-display font-bold tracking-tight text-foreground truncate">
              Dataset Preview
            </h1>
            <p className="text-xs sm:text-sm text-muted-foreground font-mono mt-0.5">
              {info.rows.toLocaleString()} rows • {info.columns.length} columns • {cleaningReport.duplicate_rows_removed} duplicates removed
            </p>
          </div>
        </div>

        <div className="flex gap-2 sm:gap-3">
          <Button variant="outline" onClick={onBack} className="gap-2 rounded-xl shadow-2xs border-border/80 bg-white hover:bg-muted/40 transition-all duration-150 hover:-translate-y-0.5 active:scale-95 text-xs sm:text-sm font-semibold">
            <ArrowLeft className="h-4 w-4" />
            <span>Upload New</span>
          </Button>
          <Button onClick={onGenerateReport} className="gap-2 flex-1 sm:flex-none rounded-xl shadow-premium bg-primary text-primary-foreground hover:bg-primary/90 transition-all duration-150 hover:-translate-y-0.5 active:scale-95 text-xs sm:text-sm font-semibold px-5 cursor-pointer">
            <span>Generate Report</span>
            <ArrowRight className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="bg-muted/30 border border-border/80 p-1.5 rounded-2xl mb-6 shadow-2xs">
          <TabsTrigger value="preview" className="rounded-xl text-xs sm:text-sm px-5 py-2 font-semibold data-[state=active]:bg-white data-[state=active]:shadow-2xs data-[state=active]:text-foreground transition-all">Data Preview</TabsTrigger>
          <TabsTrigger value="quality" className="rounded-xl text-xs sm:text-sm px-5 py-2 font-semibold data-[state=active]:bg-white data-[state=active]:shadow-2xs data-[state=active]:text-foreground transition-all">Data Quality</TabsTrigger>
        </TabsList>

        <TabsContent value="preview" className="space-y-6">
          {/* Column Chips */}
          <Card className="border border-border/80 bg-card shadow-premium rounded-2xl">
            <CardHeader className="pb-3 sm:pb-4 border-b border-border/60">
              <CardTitle className="text-lg font-display font-bold text-foreground">Detected Columns</CardTitle>
            </CardHeader>
            <CardContent className="pt-5">
              <div className="flex flex-wrap gap-2">
                {info.columns.map((column) => (
                  <div
                    key={column}
                    className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white text-xs border border-border/80 shadow-2xs font-sans hover:border-primary/30 transition-colors"
                  >
                    <span className="font-semibold text-foreground">{column}</span>
                    <Badge
                      variant={getTypeBadgeVariant(info.dtypes[column])}
                      className="text-[10px] font-mono gap-1 px-2 py-0.5 border-border/40 rounded-full"
                    >
                      {getTypeIcon(info.dtypes[column])}
                      {info.dtypes[column]}
                    </Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Data Table */}
          <Card className="border border-border/80 bg-card shadow-premium rounded-2xl overflow-hidden">
            <CardHeader className="pb-3 sm:pb-4 border-b border-border/60 bg-muted/10">
              <CardTitle className="text-lg font-display font-bold text-foreground">First 10 Rows</CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <div className="w-full overflow-x-auto custom-scrollbar">
                <div className="min-w-[650px]">
                  <Table className="border-collapse">
                    <TableHeader className="bg-muted/30">
                      <TableRow className="border-b border-border/60 hover:bg-transparent">
                        {info.columns.map((column) => (
                          <TableHead key={column} className="font-display font-bold text-xs uppercase tracking-wider text-muted-foreground px-4 py-3 whitespace-nowrap border-r border-border/60 last:border-r-0">
                            {column}
                          </TableHead>
                        ))}
                      </TableRow>
                    </TableHeader>
                    <TableBody className="divide-y divide-border/60 bg-card">
                      {info.preview.map((row, rowIndex) => {
                        if (!row) return null;
                        return (
                          <TableRow key={rowIndex} className="border-b border-border/40 hover:bg-primary/[0.02] transition-colors">
                            {info.columns.map((column) => (
                              <TableCell key={column} className="font-mono text-xs text-foreground/90 whitespace-nowrap px-4 py-2.5 border-r border-border/60 last:border-r-0">
                                {row[column] === null || row[column] === undefined ? (
                                  <span className="text-muted-foreground italic font-sans">null</span>
                                ) : (
                                  String(row[column])
                                )}
                              </TableCell>
                            ))}
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="quality">
          <Card className="border border-border bg-card shadow-premium rounded-2xl overflow-hidden">
            <CardHeader className="border-b border-border/60">
              <CardTitle className="text-lg font-display font-bold text-foreground">Column Quality Analysis</CardTitle>
              <CardDescription className="text-xs font-sans mt-0.5">Review detected issues, missing values, and data types.</CardDescription>
            </CardHeader>
            <CardContent className="p-6">
              <div className="grid gap-4">
                {info.columns.map((col) => {
                  const missing = info.missing_values?.[col] || { count: 0, percentage: 0 };
                  const issues = analysis.column_quality_flags?.[col] || [];
                  const hasIssues = issues.length > 0 || missing.count > 0;

                  return (
                    <div key={col} className={`flex flex-col sm:flex-row sm:items-center justify-between p-4 border rounded-xl transition-all duration-200 bg-white ${hasIssues ? 'border-amber-300 shadow-sm' : 'border-border hover:bg-muted/10'}`}>
                      <div className="mb-2 sm:mb-0">
                        <div className="flex items-center gap-3">
                          <span className="font-display font-bold text-base text-foreground">{col}</span>
                          <Badge variant="outline" className="text-[10px] font-mono px-2 py-0.5 bg-muted/20 border-border rounded-full">
                            {info.dtypes[col]}
                          </Badge>
                        </div>
                        {issues.length > 0 && (
                          <div className="text-xs text-amber-700 mt-2 flex flex-wrap gap-1.5 font-mono">
                            {issues.map(issue => (
                              <span key={issue} className="flex items-center gap-1 bg-amber-50 px-2 py-0.5 rounded-full border border-amber-200 font-semibold">
                                <AlertTriangle className="h-3 w-3 shrink-0" /> {issue.toLowerCase()}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                      <div className="flex items-center gap-6 text-sm">
                        <div className="flex flex-col items-end">
                          <span className="text-muted-foreground text-[10px] font-mono uppercase tracking-wider">Missing</span>
                          <span className={`font-mono text-xs font-semibold mt-0.5 ${missing.count > 0 ? 'text-destructive bg-destructive/5 px-2 py-0.5 rounded-full border border-destructive/20' : 'text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded-full border border-emerald-250'}`}>
                            {missing.count > 0 ? `${missing.count} (${missing.percentage}%)` : "none"}
                          </span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};
