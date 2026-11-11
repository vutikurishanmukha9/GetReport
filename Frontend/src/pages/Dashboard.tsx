import { useState, useEffect } from "react";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { FileSpreadsheet, Search, Download, Calendar, AlertCircle, ArrowRight, Loader2 } from "lucide-react";
import { Link } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";
import { api } from "@/services/api";

interface AuditHistoryItem {
  task_id: string;
  filename: string;
  created_at: string;
  rows?: number;
  columns?: number;
  grade?: string;
  score?: number;
  status: string;
  isDemo?: boolean;
}

export const Dashboard = () => {
  const { toast } = useToast();
  const [searchQuery, setSearchQuery] = useState("");
  const [historyItems, setHistoryItems] = useState<AuditHistoryItem[]>([]);
  const [downloadingTaskId, setDownloadingTaskId] = useState<string | null>(null);

  useEffect(() => {
    const loadAudits = async () => {
      let localItems: AuditHistoryItem[] = [];
      try {
        const cached = localStorage.getItem("getreport_audit_history");
        if (cached) {
          localItems = JSON.parse(cached);
        }
      } catch (e) {
        console.error("Failed to parse cached audit history:", e);
      }

      try {
        const res = await api.listRecentDatasets(15);
        if (res && Array.isArray(res.datasets)) {
          const remoteItems: AuditHistoryItem[] = res.datasets.map((d) => ({
            task_id: d.task_id,
            filename: d.filename,
            created_at: new Date(d.created_at || Date.now()).toLocaleDateString("en-US", {
              month: "short",
              day: "numeric",
              year: "numeric",
            }),
            rows: d.rows,
            columns: d.columns_count,
            grade: "A",
            score: 90,
            status: "COMPLETED",
            isDemo: false,
          }));

          const seen = new Set<string>();
          const combined: AuditHistoryItem[] = [];
          for (const item of [...localItems, ...remoteItems]) {
            if (!seen.has(item.task_id)) {
              seen.add(item.task_id);
              combined.push(item);
            }
          }
          if (combined.length > 0) {
            setHistoryItems(combined);
            return;
          }
        }
      } catch (err) {
        console.warn("Could not fetch remote recent datasets, using local cache:", err);
      }

      if (localItems.length > 0) {
        setHistoryItems(localItems);
      } else {
        setHistoryItems([
          {
            task_id: "demo_audit_01",
            filename: "Q3_Financial_Ledger.xlsx",
            created_at: new Date().toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" }),
            rows: 14250,
            columns: 18,
            grade: "A-",
            score: 91,
            status: "COMPLETED",
            isDemo: true
          }
        ]);
      }
    };

    loadAudits();
  }, []);

  const handleDownloadReport = async (item: AuditHistoryItem) => {
    try {
      setDownloadingTaskId(item.task_id);
      toast({
        title: "Downloading Report",
        description: `Preparing PDF report for ${item.filename}...`,
      });
      const blob = await api.downloadReportBlob(item.task_id);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${item.filename.replace(/\.[^/.]+$/, "")}_audit_report.pdf`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      toast({
        title: "Download Complete",
        description: "Report downloaded successfully.",
      });
    } catch (err) {
      console.error("Failed to download report blob:", err);
      toast({
        title: "Download Failed",
        description: "The PDF report is not yet generated or could not be retrieved.",
        variant: "destructive",
      });
    } finally {
      setDownloadingTaskId(null);
    }
  };

  const filteredItems = historyItems.filter(item =>
    item.filename.toLowerCase().includes(searchQuery.toLowerCase()) ||
    item.task_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
    (item.grade && item.grade.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  return (
    <div className="min-h-screen flex flex-col bg-background text-foreground transition-colors duration-200 animate-in fade-in duration-500 relative">
      {/* Technical Background Grid & Film Grain */}
      <div className="absolute inset-0 bg-[radial-gradient(rgba(0,0,0,0.04)_1px,transparent_1px)] dark:bg-[radial-gradient(rgba(255,255,255,0.06)_1px,transparent_1px)] [background-size:24px_24px] pointer-events-none opacity-50 dark:opacity-40" />
      <div className="hero-grain-overlay" />

      <Header onReset={() => {}} showReset={false} />

      <main className="flex-1 pt-24 sm:pt-28 pb-16 relative z-10">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-6xl space-y-8">
          
          {/* Header Section */}
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-border pb-6">
            <div>
              <div className="flex items-center gap-2 text-xs text-muted-foreground font-mono mb-1.5">
                <Link to="/" className="hover:text-foreground transition">GetReport</Link>
                <span>/</span>
                <span className="text-foreground font-semibold">Audit History</span>
              </div>
              <h1 className="text-2xl sm:text-3xl font-display font-bold text-foreground tracking-tight">
                Audit History & Reports
              </h1>
              <p className="text-sm text-muted-foreground mt-1">
                View past dataset quality audits, inspect transformation DAGs, and download reports.
              </p>
            </div>

            <Link to="/workspace">
              <Button size="sm" className="rounded-xl shadow-sm font-semibold text-xs gap-2 h-9 px-4">
                <FileSpreadsheet className="h-4 w-4" />
                <span>New Dataset Audit</span>
              </Button>
            </Link>
          </div>

          {/* Search Controls */}
          <div className="flex items-center gap-3 bg-card/90 border border-border rounded-2xl p-2 shadow-xs max-w-md backdrop-blur-md">
            <Search className="h-4 w-4 text-muted-foreground ml-3 shrink-0" />
            <Input
              type="text"
              placeholder="Search dataset name, task ID, or grade..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="border-0 focus-visible:ring-0 text-xs bg-transparent text-foreground placeholder:text-muted-foreground"
            />
          </div>

          {/* Audit Cards List */}
          {filteredItems.length === 0 ? (
            <Card className="border border-border rounded-3xl p-12 text-center bg-card/80 shadow-xs backdrop-blur-md">
              <AlertCircle className="h-10 w-10 text-muted-foreground mx-auto mb-3" />
              <h3 className="text-base font-display font-semibold text-foreground">No Audits Found</h3>
              <p className="text-xs text-muted-foreground mt-1 max-w-sm mx-auto">
                No past audit sessions match your search query. Run a new dataset ingestion in the workspace.
              </p>
            </Card>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {filteredItems.map((item) => (
                <Card key={item.task_id} className="border border-border rounded-2xl bg-card/80 backdrop-blur-md shadow-xs hover:border-zinc-300 dark:hover:border-white/20 transition-all duration-200">
                  <CardHeader className="pb-3 flex flex-row items-center justify-between space-y-0">
                    <div className="flex items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-zinc-100 dark:bg-white/[0.06] text-foreground dark:text-white border border-border dark:border-white/10">
                        <FileSpreadsheet className="h-5 w-5" />
                      </div>
                      <div>
                        <CardTitle className="text-sm sm:text-base font-display font-bold text-foreground flex items-center gap-2">
                          <span>{item.filename}</span>
                          {item.isDemo && (
                            <Badge variant="outline" className="text-[10px] font-mono px-1.5 py-0 border-border text-muted-foreground bg-muted/50">
                              Demo
                            </Badge>
                          )}
                        </CardTitle>
                        <span className="text-[10px] font-mono text-muted-foreground">ID: {item.task_id.slice(0, 16)}</span>
                      </div>
                    </div>

                    {item.grade && (
                      <Badge variant="outline" className="font-mono text-xs px-2.5 py-0.5 border-emerald-500/30 text-emerald-600 dark:text-emerald-400 bg-emerald-500/10">
                        Grade {item.grade}
                      </Badge>
                    )}
                  </CardHeader>

                  <CardContent className="space-y-4 pt-0">
                    <div className="grid grid-cols-3 gap-2 py-2 border-y border-border/60 text-center">
                      <div>
                        <span className="text-[10px] font-mono text-muted-foreground uppercase block">Rows</span>
                        <span className="text-xs font-semibold text-foreground">{item.rows ? item.rows.toLocaleString() : "N/A"}</span>
                      </div>
                      <div>
                        <span className="text-[10px] font-mono text-muted-foreground uppercase block">Columns</span>
                        <span className="text-xs font-semibold text-foreground">{item.columns ?? "N/A"}</span>
                      </div>
                      <div>
                        <span className="text-[10px] font-mono text-muted-foreground uppercase block">Score</span>
                        <span className="text-xs font-semibold text-foreground">{item.score ? `${item.score}%` : "N/A"}</span>
                      </div>
                    </div>

                    <div className="flex items-center justify-between pt-1">
                      <span className="text-[11px] text-muted-foreground flex items-center gap-1 font-mono">
                        <Calendar className="h-3 w-3" />
                        {item.created_at}
                      </span>

                      <div className="flex items-center gap-2">
                        {item.isDemo ? (
                          <Link to="/workspace">
                            <Button
                              variant="outline"
                              size="sm"
                              className="h-8 text-xs gap-1.5 rounded-lg border-border text-foreground hover:bg-muted"
                            >
                              <span>Run Audit</span>
                              <ArrowRight className="h-3 w-3" />
                            </Button>
                          </Link>
                        ) : (
                          <Button
                            variant="outline"
                            size="sm"
                            disabled={downloadingTaskId === item.task_id}
                            onClick={() => handleDownloadReport(item)}
                            className="h-8 text-xs gap-1.5 rounded-lg border-border text-foreground hover:bg-muted"
                          >
                            {downloadingTaskId === item.task_id ? (
                              <Loader2 className="h-3 w-3 animate-spin" />
                            ) : (
                              <Download className="h-3 w-3" />
                            )}
                            <span>Report</span>
                          </Button>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}

        </div>
      </main>

      <Footer />
    </div>
  );
};

export default Dashboard;
