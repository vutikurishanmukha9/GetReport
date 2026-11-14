import { useState, useEffect } from "react";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { FileSpreadsheet, Search, Download, Calendar, AlertCircle, ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";
import { api } from "@/services/api";
import { useToast } from "@/hooks/use-toast";

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

  useEffect(() => {
    // Load audit history from localStorage cache or fallback
    try {
      const cached = localStorage.getItem("getreport_audit_history");
      if (cached) {
        setHistoryItems(JSON.parse(cached));
      } else {
        // Sample preview record
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
    } catch (e) {
      console.error("Failed to load audit history:", e);
    }
  }, []);

  const filteredItems = historyItems.filter(item =>
    item.filename.toLowerCase().includes(searchQuery.toLowerCase()) ||
    item.task_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
    (item.grade && item.grade.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  return (
    <div className="min-h-screen flex flex-col bg-[#F5F2EB] text-stone-900 dark:bg-[#07080a] dark:text-zinc-100 animate-in fade-in duration-500 relative">
      {/* Technical Background Grid & Film Grain */}
      <div className="absolute inset-0 bg-[radial-gradient(rgba(0,0,0,0.05)_1px,transparent_1px)] dark:bg-[radial-gradient(rgba(255,255,255,0.06)_1px,transparent_1px)] [background-size:24px_24px] pointer-events-none opacity-40" />
      <div className="hero-grain-overlay" />

      <Header onReset={() => {}} showReset={false} />

      <main className="flex-1 pt-24 sm:pt-28 pb-16 relative z-10">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-6xl space-y-8">
          
          {/* Header Section */}
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-stone-200/90 dark:border-white/10 pb-6">
            <div>
              <div className="flex items-center gap-2 text-xs text-stone-500 dark:text-zinc-400 font-mono mb-1.5">
                <Link to="/" className="hover:text-stone-900 dark:hover:text-zinc-200 transition">GetReport</Link>
                <span>/</span>
                <span className="text-stone-900 dark:text-zinc-200">Audit History</span>
              </div>
              <h1 className="text-2xl sm:text-3xl font-display font-bold text-stone-900 dark:text-white tracking-tight">
                Audit History & Reports
              </h1>
              <p className="text-sm text-stone-600 dark:text-zinc-400 mt-1">
                View past dataset quality audits, inspect transformation DAGs, and download reports.
              </p>
            </div>

            <Link to="/workspace">
              <Button size="sm" className="rounded-xl shadow-sm bg-[#800000] text-white hover:bg-[#6b0000] dark:bg-white dark:hover:bg-zinc-200 dark:text-black font-semibold text-xs gap-2 border border-[#800000] dark:border-white h-9 px-4">
                <FileSpreadsheet className="h-4 w-4" />
                <span>New Dataset Audit</span>
              </Button>
            </Link>
          </div>

          {/* Search Controls */}
          <div className="flex items-center gap-3 bg-[#FAF8F5] border border-stone-300 dark:bg-[#0c0d12]/90 dark:border-white/10 rounded-2xl p-2 shadow-sm max-w-md backdrop-blur-md">
            <Search className="h-4 w-4 text-stone-500 dark:text-zinc-400 ml-3" />
            <Input
              type="text"
              placeholder="Search dataset name, task ID, or grade..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="border-0 focus-visible:ring-0 text-xs bg-transparent text-stone-900 placeholder:text-stone-500 dark:text-white dark:placeholder:text-zinc-500"
            />
          </div>

          {/* Audit Cards List */}
          {filteredItems.length === 0 ? (
            <Card className="border border-stone-200 bg-[#FAF8F5] dark:border-white/10 dark:bg-[#0c0d12]/80 rounded-3xl p-12 text-center shadow-sm backdrop-blur-md">
              <AlertCircle className="h-10 w-10 text-stone-400 dark:text-zinc-500 mx-auto mb-3" />
              <h3 className="text-base font-display font-semibold text-stone-900 dark:text-white">No Audits Found</h3>
              <p className="text-xs text-stone-600 dark:text-zinc-400 mt-1 max-w-sm mx-auto">
                No past audit sessions match your search query. Run a new dataset ingestion in the workspace.
              </p>
            </Card>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {filteredItems.map((item) => (
                <Card key={item.task_id} className="border border-stone-200/90 rounded-2xl bg-[#FAF8F5] dark:border-white/10 dark:bg-[#0c0d12]/80 backdrop-blur-md shadow-sm hover:border-stone-400 dark:hover:border-white/20 transition-all duration-200">
                  <CardHeader className="pb-3 flex flex-row items-center justify-between space-y-0">
                    <div className="flex items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-stone-200 text-stone-900 border border-stone-300 dark:bg-white/[0.06] dark:text-white dark:border-white/10 font-bold">
                        <FileSpreadsheet className="h-5 w-5" />
                      </div>
                      <div>
                        <CardTitle className="text-sm sm:text-base font-display font-bold text-stone-900 dark:text-white flex items-center gap-2">
                          <span>{item.filename}</span>
                          {item.isDemo && (
                            <Badge variant="outline" className="text-[10px] font-mono px-1.5 py-0 border-stone-300 text-stone-600 bg-stone-100 dark:border-white/10 dark:text-zinc-400 dark:bg-white/[0.04]">
                              Demo
                            </Badge>
                          )}
                        </CardTitle>
                        <span className="text-[10px] font-mono text-stone-500 dark:text-zinc-400">ID: {item.task_id.slice(0, 16)}</span>
                      </div>
                    </div>

                    {item.grade && (
                      <Badge variant="outline" className="font-mono text-xs px-2.5 py-0.5 border-emerald-300 text-emerald-900 bg-emerald-100 dark:border-emerald-500/30 dark:text-emerald-400 dark:bg-emerald-500/10 font-bold">
                        Grade {item.grade}
                      </Badge>
                    )}
                  </CardHeader>

                  <CardContent className="space-y-4 pt-0">
                    <div className="grid grid-cols-3 gap-2 py-2 border-y border-stone-200 dark:border-white/5 text-center">
                      <div>
                        <span className="text-[10px] font-mono text-stone-500 dark:text-zinc-400 uppercase block">Rows</span>
                        <span className="text-xs font-semibold text-stone-900 dark:text-white">{item.rows ? item.rows.toLocaleString() : "N/A"}</span>
                      </div>
                      <div>
                        <span className="text-[10px] font-mono text-stone-500 dark:text-zinc-400 uppercase block">Columns</span>
                        <span className="text-xs font-semibold text-stone-900 dark:text-white">{item.columns ?? "N/A"}</span>
                      </div>
                      <div>
                        <span className="text-[10px] font-mono text-stone-500 dark:text-zinc-400 uppercase block">Score</span>
                        <span className="text-xs font-semibold text-stone-900 dark:text-white">{item.score ? `${item.score}%` : "N/A"}</span>
                      </div>
                    </div>

                    <div className="flex items-center justify-between pt-1">
                      <span className="text-[11px] text-stone-500 dark:text-zinc-400 flex items-center gap-1 font-mono">
                        <Calendar className="h-3 w-3" />
                        {item.created_at}
                      </span>

                      <div className="flex items-center gap-2">
                        {item.isDemo ? (
                          <Link to="/workspace">
                            <Button
                              variant="outline"
                              size="sm"
                              className="h-8 text-xs gap-1.5 rounded-lg border-stone-300 bg-stone-200/80 text-stone-800 hover:bg-stone-300 dark:border-white/10 dark:bg-white/[0.04] dark:text-zinc-200 dark:hover:bg-white/[0.08]"
                            >
                              <span>Run Audit</span>
                              <ArrowRight className="h-3 w-3" />
                            </Button>
                          </Link>
                        ) : (
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-8 text-xs gap-1.5 text-stone-600 hover:text-stone-900 hover:bg-stone-200/80 dark:text-zinc-300 dark:hover:text-white dark:hover:bg-white/[0.06] rounded-lg"
                            onClick={async () => {
                              try {
                                const blob = await api.downloadReportBlob(item.task_id);
                                const url = URL.createObjectURL(blob);
                                const a = document.createElement("a");
                                a.href = url;
                                a.download = `${item.task_id}_${item.filename}.pdf`;
                                a.click();
                                URL.revokeObjectURL(url);
                              } catch {
                                toast({ title: "Session Expired", description: "Audit data in ephemeral RAM has expired. Please run a new audit.", variant: "destructive" });
                              }
                            }}
                          >
                            <Download className="h-3.5 w-3.5" />
                            PDF
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
