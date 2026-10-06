import { useState, useMemo } from "react";
import {
  ArrowLeft, ArrowRight, FileSpreadsheet, Hash, Calendar, Type,
  AlertTriangle, Search, CheckCircle2, Database, Layers, FileText
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
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
  if (type.includes("datetime") || type.includes("date")) return <Calendar className="h-3 w-3" />;
  return <Type className="h-3 w-3" />;
};

const getTypeBadgeVariant = (type: string): "default" | "secondary" | "outline" => {
  if (type.includes("int") || type.includes("float")) return "default";
  if (type.includes("datetime") || type.includes("date")) return "secondary";
  return "outline";
};

export const DataPreview = ({ info, cleaningReport, analysis, onGenerateReport, onBack }: DataPreviewProps) => {
  const [activeTab, setActiveTab] = useState<string>("preview");
  const [rowSearchQuery, setRowSearchQuery] = useState<string>("");
  const [colTypeFilter, setColTypeFilter] = useState<"all" | "numeric" | "categorical" | "datetime">("all");
  const [qualityFilter, setQualityFilter] = useState<"all" | "issues" | "clean">("all");
  const [colSearchQuery, setColSearchQuery] = useState<string>("");

  // Filter preview rows by text search
  const filteredRows = useMemo(() => {
    if (!rowSearchQuery.trim()) return info.preview;
    const q = rowSearchQuery.toLowerCase().trim();
    return info.preview.filter(row => {
      if (!row) return false;
      return Object.values(row).some(val =>
        val !== null && val !== undefined && String(val).toLowerCase().includes(q)
      );
    });
  }, [info.preview, rowSearchQuery]);

  // Filter columns based on category
  const visibleColumns = useMemo(() => {
    return info.columns.filter(col => {
      const dtype = (info.dtypes[col] || "").toLowerCase();
      if (colTypeFilter === "numeric") return dtype.includes("int") || dtype.includes("float");
      if (colTypeFilter === "categorical") return dtype.includes("str") || dtype.includes("object") || dtype.includes("cat") || dtype.includes("bool");
      if (colTypeFilter === "datetime") return dtype.includes("date") || dtype.includes("time");
      return true;
    });
  }, [info.columns, info.dtypes, colTypeFilter]);

  // Filter columns in the quality tab
  const qualityColumns = useMemo(() => {
    return info.columns.filter(col => {
      const matchesSearch = col.toLowerCase().includes(colSearchQuery.toLowerCase().trim());
      if (!matchesSearch) return false;

      const missing = info.missing_values?.[col] || { count: 0, percentage: 0 };
      const issues = analysis.column_quality_flags?.[col] || [];
      const hasIssues = issues.length > 0 || missing.count > 0;

      if (qualityFilter === "issues") return hasIssues;
      if (qualityFilter === "clean") return !hasIssues;
      return true;
    });
  }, [info.columns, info.missing_values, analysis.column_quality_flags, colSearchQuery, qualityFilter]);

  const totalIssuesCount = useMemo(() => {
    return info.columns.reduce((acc, col) => {
      const missing = info.missing_values?.[col]?.count || 0;
      const issues = analysis.column_quality_flags?.[col]?.length || 0;
      return acc + (missing > 0 || issues > 0 ? 1 : 0);
    }, 0);
  }, [info.columns, info.missing_values, analysis.column_quality_flags]);

  return (
    <div className="max-w-6xl mx-auto space-y-6 sm:space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-400">

      {/* Header Section */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-border/60 pb-6">
        <div className="flex items-start sm:items-center gap-3 sm:gap-4">
          <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-primary/10 border border-primary/20 shadow-premium">
            <FileSpreadsheet className="h-6 w-6 text-primary" />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h1 className="text-2xl sm:text-3xl font-display font-bold tracking-tight text-foreground truncate">
                Dataset Ingestion Preview
              </h1>
              <Badge variant="outline" className="text-[10px] font-mono uppercase bg-emerald-50 text-emerald-700 border-emerald-200">
                Validated
              </Badge>
            </div>
            <p className="text-xs sm:text-sm text-muted-foreground font-mono mt-0.5">
              Automated schema extraction complete • Ready for deep visual synthesis
            </p>
          </div>
        </div>

        <div className="flex gap-2 sm:gap-3">
          <Button
            variant="outline"
            onClick={onBack}
            className="gap-2 rounded-xl shadow-2xs border-slate-300 bg-white hover:bg-slate-50 transition-all duration-150 hover:-translate-y-0.5 active:scale-95 text-xs sm:text-sm font-semibold h-10 px-4 cursor-pointer"
          >
            <ArrowLeft className="h-4 w-4 text-slate-600" />
            <span>Upload New</span>
          </Button>
          <button
            onClick={onGenerateReport}
            className="gap-2 flex-1 sm:flex-none h-10 px-5 rounded-xl bg-gradient-to-r from-violet-600 via-purple-600 to-indigo-600 hover:from-violet-500 hover:via-purple-500 hover:to-indigo-500 text-white font-display text-xs sm:text-sm font-bold tracking-tight shadow-[0_4px_14px_-2px_rgba(124,58,237,0.38),inset_0_1px_1px_rgba(255,255,255,0.3)] border border-violet-400/40 ring-1 ring-white/20 transition-all duration-150 hover:-translate-y-0.5 active:scale-95 flex items-center justify-center cursor-pointer"
          >
            <FileText className="h-4 w-4" />
            <span>Generate Full Report</span>
            <ArrowRight className="h-4 w-4 opacity-90" />
          </button>
        </div>
      </div>

      {/* Fast Summary Metrics Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4">
        <Card className="border border-border/80 bg-card shadow-premium rounded-xl p-4">
          <div className="flex items-center justify-between text-muted-foreground text-xs font-mono uppercase">
            <span>Total Rows</span>
            <Database className="h-4 w-4 text-primary" />
          </div>
          <div className="mt-2 text-xl sm:text-2xl font-display font-bold text-foreground">
            {info.rows.toLocaleString()}
          </div>
          <div className="text-[11px] font-mono text-muted-foreground mt-0.5">
            {filteredRows.length} shown in sample preview
          </div>
        </Card>

        <Card className="border border-border/80 bg-card shadow-premium rounded-xl p-4">
          <div className="flex items-center justify-between text-muted-foreground text-xs font-mono uppercase">
            <span>Total Columns</span>
            <Layers className="h-4 w-4 text-primary" />
          </div>
          <div className="mt-2 text-xl sm:text-2xl font-display font-bold text-foreground">
            {info.columns.length}
          </div>
          <div className="text-[11px] font-mono text-muted-foreground mt-0.5">
            {info.numeric_columns?.length || 0} numeric • {info.categorical_columns?.length || 0} categorical
          </div>
        </Card>

        <Card className="border border-border/80 bg-card shadow-premium rounded-xl p-4">
          <div className="flex items-center justify-between text-muted-foreground text-xs font-mono uppercase">
            <span>Memory Footprint</span>
            <Hash className="h-4 w-4 text-primary" />
          </div>
          <div className="mt-2 text-xl sm:text-2xl font-display font-bold text-foreground">
            {(info.memory_usage_mb || 0).toFixed(2)} <span className="text-sm font-normal text-muted-foreground">MB</span>
          </div>
          <div className="text-[11px] font-mono text-muted-foreground mt-0.5">
            In-memory Polars table
          </div>
        </Card>

        <Card className="border border-border/80 bg-card shadow-premium rounded-xl p-4">
          <div className="flex items-center justify-between text-muted-foreground text-xs font-mono uppercase">
            <span>Quality Status</span>
            <CheckCircle2 className="h-4 w-4 text-emerald-600" />
          </div>
          <div className="mt-2 text-xl sm:text-2xl font-display font-bold text-foreground">
            {totalIssuesCount === 0 ? "100% Clean" : `${info.columns.length - totalIssuesCount}/${info.columns.length}`}
          </div>
          <div className="text-[11px] font-mono text-muted-foreground mt-0.5">
            {cleaningReport.duplicate_rows_removed} duplicates removed
          </div>
        </Card>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="bg-muted/30 border border-border/80 p-1.5 rounded-2xl mb-6 shadow-2xs">
          <TabsTrigger
            value="preview"
            className="rounded-xl text-xs sm:text-sm px-5 py-2 font-semibold data-[state=active]:bg-white data-[state=active]:shadow-2xs data-[state=active]:text-foreground transition-all"
          >
            Data Preview & Types
          </TabsTrigger>
          <TabsTrigger
            value="quality"
            className="rounded-xl text-xs sm:text-sm px-5 py-2 font-semibold data-[state=active]:bg-white data-[state=active]:shadow-2xs data-[state=active]:text-foreground transition-all flex items-center gap-2"
          >
            <span>Column Quality Audit</span>
            {totalIssuesCount > 0 && (
              <Badge variant="secondary" className="bg-amber-100 text-amber-800 text-[10px] px-1.5 py-0 h-4 rounded-full">
                {totalIssuesCount}
              </Badge>
            )}
          </TabsTrigger>
        </TabsList>

        <TabsContent value="preview" className="space-y-6">
          {/* Column Type Filter & Search Bar */}
          <Card className="border border-border/80 bg-card shadow-premium rounded-2xl p-5">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <h3 className="text-base font-display font-bold text-foreground">Schema Variables</h3>
                <p className="text-xs text-muted-foreground">Select a category to filter columns in the preview grid</p>
              </div>

              {/* Column Category Pills */}
              <div className="flex flex-wrap items-center gap-1.5">
                <button
                  type="button"
                  onClick={() => setColTypeFilter("all")}
                  className={`px-3 py-1 rounded-lg text-xs font-mono font-semibold transition-all ${
                    colTypeFilter === "all"
                      ? "bg-primary text-primary-foreground shadow-2xs"
                      : "bg-muted/40 hover:bg-muted text-muted-foreground"
                  }`}
                >
                  All ({info.columns.length})
                </button>
                <button
                  type="button"
                  onClick={() => setColTypeFilter("numeric")}
                  className={`px-3 py-1 rounded-lg text-xs font-mono font-semibold transition-all ${
                    colTypeFilter === "numeric"
                      ? "bg-primary text-primary-foreground shadow-2xs"
                      : "bg-muted/40 hover:bg-muted text-muted-foreground"
                  }`}
                >
                  Numeric ({info.numeric_columns?.length || 0})
                </button>
                <button
                  type="button"
                  onClick={() => setColTypeFilter("categorical")}
                  className={`px-3 py-1 rounded-lg text-xs font-mono font-semibold transition-all ${
                    colTypeFilter === "categorical"
                      ? "bg-primary text-primary-foreground shadow-2xs"
                      : "bg-muted/40 hover:bg-muted text-muted-foreground"
                  }`}
                >
                  Categorical ({info.categorical_columns?.length || 0})
                </button>
              </div>
            </div>

            {/* Column Chips */}
            <div className="flex flex-wrap gap-2 pt-4 border-t border-border/60 mt-4">
              {visibleColumns.map((column) => (
                <div
                  key={column}
                  className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white text-xs border border-border/80 shadow-2xs font-sans hover:border-primary/30 transition-colors"
                >
                  <span className="font-semibold text-foreground">{column}</span>
                  <Badge
                    variant={getTypeBadgeVariant(info.dtypes[column] || "")}
                    className="text-[10px] font-mono gap-1 px-2 py-0.5 border-border/40 rounded-full"
                  >
                    {getTypeIcon(info.dtypes[column] || "")}
                    {info.dtypes[column] || "unknown"}
                  </Badge>
                </div>
              ))}
            </div>
          </Card>

          {/* Interactive Data Table with Instant Search */}
          <Card className="border border-border/80 bg-card shadow-premium rounded-2xl overflow-hidden">
            <CardHeader className="pb-3 sm:pb-4 border-b border-border/60 bg-muted/10">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div>
                  <CardTitle className="text-lg font-display font-bold text-foreground">
                    Interactive Sample Rows
                  </CardTitle>
                  <CardDescription className="text-xs font-sans mt-0.5">
                    Showing preview records across {visibleColumns.length} visible columns
                  </CardDescription>
                </div>

                {/* Instant Search Input */}
                <div className="relative w-full sm:w-64">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
                  <Input
                    placeholder="Search in sample rows..."
                    value={rowSearchQuery}
                    onChange={(e) => setRowSearchQuery(e.target.value)}
                    className="pl-9 h-9 text-xs rounded-xl bg-white border-border/80"
                  />
                </div>
              </div>
            </CardHeader>
            <CardContent className="p-0">
              <div className="w-full overflow-x-auto custom-scrollbar">
                <div className="min-w-[650px]">
                  <Table className="border-collapse">
                    <TableHeader className="bg-muted/30">
                      <TableRow className="border-b border-border/60 hover:bg-transparent">
                        <TableHead className="font-mono font-bold text-[10px] text-muted-foreground px-3 py-3 w-12 text-center border-r border-border/60">
                          #
                        </TableHead>
                        {visibleColumns.map((column) => (
                          <TableHead key={column} className="font-display font-bold text-xs uppercase tracking-wider text-muted-foreground px-4 py-3 whitespace-nowrap border-r border-border/60 last:border-r-0">
                            {column}
                          </TableHead>
                        ))}
                      </TableRow>
                    </TableHeader>
                    <TableBody className="divide-y divide-border/60 bg-card">
                      {filteredRows.length > 0 ? (
                        filteredRows.map((row, rowIndex) => {
                          if (!row) return null;
                          return (
                            <TableRow key={rowIndex} className="border-b border-border/40 hover:bg-primary/[0.02] transition-colors">
                              <TableCell className="font-mono text-[10px] text-muted-foreground px-3 py-2.5 text-center border-r border-border/60">
                                {rowIndex + 1}
                              </TableCell>
                              {visibleColumns.map((column) => (
                                <TableCell key={column} className="font-mono text-xs text-foreground/90 whitespace-nowrap px-4 py-2.5 border-r border-border/60 last:border-r-0">
                                  {row[column] === null || row[column] === undefined ? (
                                    <span className="text-muted-foreground/60 italic font-sans text-[11px]">null</span>
                                  ) : (
                                    String(row[column])
                                  )}
                                </TableCell>
                              ))}
                            </TableRow>
                          );
                        })
                      ) : (
                        <TableRow>
                          <TableCell colSpan={visibleColumns.length + 1} className="text-center py-8 text-xs font-mono text-muted-foreground">
                            No records matched your search query &ldquo;{rowSearchQuery}&rdquo;
                          </TableCell>
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="quality" className="space-y-6">
          <Card className="border border-border bg-card shadow-premium rounded-2xl overflow-hidden">
            <CardHeader className="border-b border-border/60 pb-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                  <CardTitle className="text-lg font-display font-bold text-foreground">
                    Column Health & Audit Ledger
                  </CardTitle>
                  <CardDescription className="text-xs font-sans mt-0.5">
                    Detailed per-variable breakdown of missing data, formatting issues, and quality flags
                  </CardDescription>
                </div>

                <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2">
                  <div className="flex items-center gap-1 bg-muted/40 p-1 rounded-xl border border-border/60">
                    <button
                      type="button"
                      onClick={() => setQualityFilter("all")}
                      className={`px-3 py-1 rounded-lg text-xs font-mono font-semibold transition-all ${
                        qualityFilter === "all" ? "bg-white text-foreground shadow-2xs" : "text-muted-foreground"
                      }`}
                    >
                      All ({info.columns.length})
                    </button>
                    <button
                      type="button"
                      onClick={() => setQualityFilter("issues")}
                      className={`px-3 py-1 rounded-lg text-xs font-mono font-semibold transition-all ${
                        qualityFilter === "issues" ? "bg-amber-100 text-amber-800 shadow-2xs" : "text-muted-foreground"
                      }`}
                    >
                      Flags ({totalIssuesCount})
                    </button>
                    <button
                      type="button"
                      onClick={() => setQualityFilter("clean")}
                      className={`px-3 py-1 rounded-lg text-xs font-mono font-semibold transition-all ${
                        qualityFilter === "clean" ? "bg-emerald-100 text-emerald-800 shadow-2xs" : "text-muted-foreground"
                      }`}
                    >
                      Clean ({info.columns.length - totalIssuesCount})
                    </button>
                  </div>

                  <div className="relative w-full sm:w-48">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
                    <Input
                      placeholder="Filter columns..."
                      value={colSearchQuery}
                      onChange={(e) => setColSearchQuery(e.target.value)}
                      className="pl-9 h-8 text-xs rounded-xl bg-white border-border/80"
                    />
                  </div>
                </div>
              </div>
            </CardHeader>

            <CardContent className="p-6">
              <div className="grid gap-3.5">
                {qualityColumns.length > 0 ? (
                  qualityColumns.map((col) => {
                    const missing = info.missing_values?.[col] || { count: 0, percentage: 0 };
                    const issues = analysis.column_quality_flags?.[col] || [];
                    const hasIssues = issues.length > 0 || missing.count > 0;

                    return (
                      <div
                        key={col}
                        className={`flex flex-col sm:flex-row sm:items-center justify-between p-4 border rounded-xl transition-all duration-200 bg-white ${
                          hasIssues ? 'border-amber-300 shadow-sm' : 'border-border hover:bg-muted/10'
                        }`}
                      >
                        <div className="mb-2 sm:mb-0">
                          <div className="flex items-center gap-3">
                            <span className="font-display font-bold text-base text-foreground">{col}</span>
                            <Badge variant="outline" className="text-[10px] font-mono px-2 py-0.5 bg-muted/20 border-border rounded-full">
                              {info.dtypes[col] || "unknown"}
                            </Badge>
                            {!hasIssues && (
                              <span className="inline-flex items-center gap-1 text-[10px] font-mono text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded-full border border-emerald-200">
                                <CheckCircle2 className="h-3 w-3" /> Healthy
                              </span>
                            )}
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
                            <span className="text-muted-foreground text-[10px] font-mono uppercase tracking-wider">Missing Values</span>
                            <span className={`font-mono text-xs font-semibold mt-0.5 ${
                              missing.count > 0
                                ? 'text-destructive bg-destructive/5 px-2 py-0.5 rounded-full border border-destructive/20'
                                : 'text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded-full border border-emerald-200'
                            }`}>
                              {missing.count > 0 ? `${missing.count.toLocaleString()} (${missing.percentage.toFixed(1)}%)` : "0 (0%)"}
                            </span>
                          </div>
                        </div>
                      </div>
                    );
                  })
                ) : (
                  <div className="text-center py-8 text-xs font-mono text-muted-foreground">
                    No columns match the selected filters.
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};
