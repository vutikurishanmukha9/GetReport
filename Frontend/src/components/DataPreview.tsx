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
import { ScrollArea } from "@/components/ui/scroll-area";
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
    <div className="max-w-6xl mx-auto space-y-6 sm:space-y-8 animate-in fade-in slide-in-from-bottom-5 duration-500">

      {/* Header Section */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-start sm:items-center gap-3 sm:gap-4">
          <div className="flex h-10 w-10 sm:h-12 sm:w-12 shrink-0 items-center justify-center rounded-lg bg-primary/10">
            <FileSpreadsheet className="h-5 w-5 sm:h-6 sm:w-6 text-primary" />
          </div>
          <div className="min-w-0">
            <h1 className="text-xl sm:text-2xl md:text-3xl font-bold truncate">
              Dataset Preview
            </h1>
            <p className="text-sm sm:text-base text-muted-foreground">
              {info.rows.toLocaleString()} rows • {info.columns.length} columns • {cleaningReport.duplicate_rows_removed} duplicates removed
            </p>
          </div>
        </div>

        <div className="flex gap-2 sm:gap-3">
          <Button variant="outline" onClick={onBack} className="gap-2">
            <ArrowLeft className="h-4 w-4" />
            <span className="hidden sm:inline">Upload New</span>
          </Button>
          <Button onClick={onGenerateReport} className="gap-2 flex-1 sm:flex-none">
            Generate Report
            <ArrowRight className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-2 lg:w-[400px] mb-6">
          <TabsTrigger value="preview">Data Preview</TabsTrigger>
          <TabsTrigger value="quality">Data Quality</TabsTrigger>
        </TabsList>

        <TabsContent value="preview" className="space-y-6">
          {/* Column Chips */}
          <Card>
            <CardHeader className="pb-3 sm:pb-4">
              <CardTitle className="text-lg sm:text-xl">Detected Columns</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2 sm:gap-3">
                {info.columns.map((column) => (
                  <div
                    key={column}
                    className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-muted text-sm"
                  >
                    <span className="font-medium">{column}</span>
                    <Badge
                      variant={getTypeBadgeVariant(info.dtypes[column])}
                      className="text-xs gap-1"
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
          <Card>
            <CardHeader className="pb-3 sm:pb-4">
              <CardTitle className="text-lg sm:text-xl">First 10 Rows</CardTitle>
            </CardHeader>
            <CardContent className="p-0 sm:p-6 sm:pt-0">
              <ScrollArea className="w-full">
                <div className="min-w-[600px]">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        {info.columns.map((column) => (
                          <TableHead key={column} className="font-semibold whitespace-nowrap">
                            {column}
                          </TableHead>
                        ))}
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {info.preview.map((row, rowIndex) => {
                        if (!row) return null;
                        return (
                          <TableRow key={rowIndex}>
                            {info.columns.map((column) => (
                              <TableCell key={column} className="font-mono text-sm whitespace-nowrap">
                                {String(row[column] ?? "-")}
                              </TableCell>
                            ))}
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="quality">
          <Card>
            <CardHeader>
              <CardTitle>Column Quality Analysis</CardTitle>
              <CardDescription>Review detected issues, missing values, and data types.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4">
                {info.columns.map((col) => {
                  const missing = info.missing_values?.[col] || { count: 0, percentage: 0 };
                  const issues = analysis.column_quality_flags?.[col] || [];
                  const hasIssues = issues.length > 0 || missing.count > 0;

                  return (
                    <div key={col} className={`flex flex-col sm:flex-row sm:items-center justify-between p-4 border rounded-lg transition-colors ${hasIssues ? 'bg-orange-50/50 dark:bg-orange-950/20 border-orange-200 dark:border-orange-900' : 'hover:bg-muted/30'}`}>
                      <div className="mb-2 sm:mb-0">
                        <div className="flex items-center gap-3">
                          <span className="font-semibold text-base">{col}</span>
                          <Badge variant="outline" className="text-xs font-mono">
                            {info.dtypes[col]}
                          </Badge>
                        </div>
                        {issues.length > 0 && (
                          <div className="text-sm text-orange-600 dark:text-orange-400 mt-2 flex flex-wrap gap-2">
                            {issues.map(issue => (
                              <span key={issue} className="flex items-center gap-1 bg-orange-100 dark:bg-orange-900/40 px-2 py-0.5 rounded text-xs font-medium">
                                <AlertTriangle className="h-3 w-3" /> {issue}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                      <div className="flex items-center gap-6 text-sm">
                        <div className="flex flex-col items-end">
                          <span className="text-muted-foreground text-xs uppercase tracking-wider">Missing</span>
                          <span className={`font-medium ${missing.count > 0 ? 'text-destructive' : 'text-green-600'}`}>
                            {missing.count} ({missing.percentage}%)
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
