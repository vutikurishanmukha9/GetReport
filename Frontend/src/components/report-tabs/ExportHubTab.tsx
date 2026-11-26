import { BookOpen, Download, FileSpreadsheet, FileText, ShieldCheck, Table2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { api } from "@/services/api";
import type { ExportFormat } from "@/types/api";

interface ExportHubTabProps {
  taskId: string | null;
  filename: string;
  isDownloadingPdf: boolean;
  onDownloadPdf: () => void;
  onDownloadExport: (format: ExportFormat) => void;
  toast: (opts: { title: string; description: string; variant?: "default" | "destructive" }) => void;
}

export function ExportHubTab({
  taskId,
  filename,
  isDownloadingPdf,
  onDownloadPdf,
  onDownloadExport,
  toast,
}: ExportHubTabProps) {
  const handleGxDownload = async () => {
    if (!taskId) return;
    try {
      await api.downloadGxSuite(taskId, filename || "dataset");
      toast({
        title: "Expectation Suite Exported",
        description: "Downloaded Great Expectations data contract suite (.json)",
      });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Could not download GX suite.";
      toast({ title: "Export Failed", description: msg, variant: "destructive" });
    }
  };

  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      <Card className="border border-border/80 bg-card shadow-premium rounded-3xl p-6 flex flex-col justify-between hover:border-primary/40 transition-all">
        <div className="space-y-3">
          <div className="h-10 w-10 rounded-xl bg-primary/10 text-primary flex items-center justify-center">
            <FileText className="h-5 w-5" />
          </div>
          <div>
            <h4 className="font-display font-bold text-base text-foreground">Publication PDF Report</h4>
            <p className="text-xs text-muted-foreground mt-1">
              Full executive briefing featuring confidence seal, visual plots, multicollinearity audits, and AI narrative synthesis.
            </p>
          </div>
        </div>
        <div className="pt-6">
          <Button
            className="w-full rounded-xl bg-primary text-primary-foreground font-semibold text-xs shadow-2xs gap-2"
            disabled={isDownloadingPdf}
            onClick={onDownloadPdf}
          >
            <Download className="h-3.5 w-3.5" />
            <span>{isDownloadingPdf ? "Downloading..." : "Download PDF"}</span>
          </Button>
        </div>
      </Card>

      <Card className="border border-border/80 bg-card shadow-premium rounded-3xl p-6 flex flex-col justify-between hover:border-primary/40 transition-all">
        <div className="space-y-3">
          <div className="h-10 w-10 rounded-xl bg-emerald-500/10 text-emerald-600 flex items-center justify-center">
            <BookOpen className="h-5 w-5" />
          </div>
          <div>
            <h4 className="font-display font-bold text-base text-foreground">Standalone HTML Dossier</h4>
            <p className="text-xs text-muted-foreground mt-1">
              Self-contained offline document containing interactive plots, schema dictionaries, and responsive formatting.
            </p>
          </div>
        </div>
        <div className="pt-6">
          <Button
            variant="outline"
            className="w-full rounded-xl border-border/80 bg-secondary/80 hover:bg-secondary text-foreground font-semibold text-xs shadow-xs gap-2 transition-all"
            onClick={() => onDownloadExport("html")}
          >
            <Download className="h-3.5 w-3.5" />
            <span>Download HTML</span>
          </Button>
        </div>
      </Card>

      <Card className="border border-border/80 bg-card shadow-premium rounded-3xl p-6 flex flex-col justify-between hover:border-primary/40 transition-all">
        <div className="space-y-3">
          <div className="h-10 w-10 rounded-xl bg-emerald-500/10 text-emerald-600 flex items-center justify-center">
            <FileSpreadsheet className="h-5 w-5" />
          </div>
          <div>
            <h4 className="font-display font-bold text-base text-foreground">Excel Workbook (.xlsx)</h4>
            <p className="text-xs text-muted-foreground mt-1">
              Spreadsheet workbook export of the cleaned, type-standardized dataset ready for Excel and BI tools.
            </p>
          </div>
        </div>
        <div className="pt-6">
          <Button
            variant="outline"
            className="w-full rounded-xl border-border/80 bg-secondary/80 hover:bg-secondary text-foreground font-semibold text-xs shadow-xs gap-2 transition-all"
            onClick={() => onDownloadExport("xlsx")}
          >
            <Download className="h-3.5 w-3.5" />
            <span>Export Excel (.xlsx)</span>
          </Button>
        </div>
      </Card>

      <Card className="border border-border/80 bg-card shadow-premium rounded-3xl p-6 flex flex-col justify-between hover:border-primary/40 transition-all">
        <div className="space-y-3">
          <div className="h-10 w-10 rounded-xl bg-amber-500/10 text-amber-600 flex items-center justify-center">
            <FileSpreadsheet className="h-5 w-5" />
          </div>
          <div>
            <h4 className="font-display font-bold text-base text-foreground">Cleaned CSV Ledger</h4>
            <p className="text-xs text-muted-foreground mt-1">
              Standard comma-separated file with duplicate records dropped, missing values imputed, and sanitized headers.
            </p>
          </div>
        </div>
        <div className="pt-6">
          <Button
            variant="outline"
            className="w-full rounded-xl border-border/80 bg-secondary/80 hover:bg-secondary text-foreground font-semibold text-xs shadow-xs gap-2 transition-all"
            onClick={() => onDownloadExport("csv")}
          >
            <Download className="h-3.5 w-3.5" />
            <span>Export Cleaned CSV</span>
          </Button>
        </div>
      </Card>

      <Card className="border border-border/80 bg-card shadow-premium rounded-3xl p-6 flex flex-col justify-between hover:border-primary/40 transition-all">
        <div className="space-y-3">
          <div className="h-10 w-10 rounded-xl bg-purple-500/10 text-purple-600 flex items-center justify-center">
            <Table2 className="h-5 w-5" />
          </div>
          <div>
            <h4 className="font-display font-bold text-base text-foreground">Apache Parquet Storage</h4>
            <p className="text-xs text-muted-foreground mt-1">
              Snappy-compressed columnar format optimized for Spark, DuckDB, Snowflake, and production ML pipelines.
            </p>
          </div>
        </div>
        <div className="pt-6">
          <Button
            variant="outline"
            className="w-full rounded-xl border-border/80 bg-secondary/80 hover:bg-secondary text-foreground font-semibold text-xs shadow-xs gap-2 transition-all"
            onClick={() => onDownloadExport("parquet")}
          >
            <Download className="h-3.5 w-3.5" />
            <span>Export Parquet</span>
          </Button>
        </div>
      </Card>

      <Card className="border border-border/80 bg-card shadow-premium rounded-3xl p-6 flex flex-col justify-between hover:border-primary/40 transition-all">
        <div className="space-y-3">
          <div className="h-10 w-10 rounded-xl bg-stone-200 text-stone-700 dark:bg-white/[0.05] dark:text-zinc-300 flex items-center justify-center">
            <ShieldCheck className="h-5 w-5" />
          </div>
          <div>
            <h4 className="font-display font-bold text-base text-foreground">Great Expectations (GX) Suite</h4>
            <p className="text-xs text-muted-foreground mt-1">
              Automated JSON data quality contract suite ready for CI/CD pipeline assertion and orchestrators.
            </p>
          </div>
        </div>
        <div className="pt-6">
          <Button
            variant="outline"
            className="w-full rounded-xl border-border/80 bg-secondary/80 hover:bg-secondary text-foreground font-semibold text-xs shadow-xs gap-2 transition-all"
            onClick={() => void handleGxDownload()}
          >
            <Download className="h-3.5 w-3.5" />
            <span>Export GX Suite (.json)</span>
          </Button>
        </div>
      </Card>
    </div>
  );
}
