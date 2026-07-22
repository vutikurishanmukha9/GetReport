import { useState, useEffect } from "react";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { FileSpreadsheet, Search, Download, ExternalLink, Calendar, CheckCircle2, AlertCircle } from "lucide-react";
import { Link } from "react-router-dom";
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
}

export const Dashboard = () => {
  const [searchQuery, setSearchQuery] = useState("");
  const [historyItems, setHistoryItems] = useState<AuditHistoryItem[]>([]);

  useEffect(() => {
    // Load audit history from localStorage cache or fallback
    try {
      const cached = localStorage.getItem("getreport_audit_history");
      if (cached) {
        setHistoryItems(JSON.parse(cached));
      } else {
        // Demo item for preview
        setHistoryItems([
          {
            task_id: "demo_audit_01",
            filename: "Q3_Financial_Ledger.xlsx",
            created_at: new Date().toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" }),
            rows: 14250,
            columns: 18,
            grade: "A-",
            score: 91,
            status: "COMPLETED"
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
    <div className="min-h-screen flex flex-col bg-background">
      <Header onReset={() => {}} showReset={false} />

      <main className="flex-1 pt-24 pb-16">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-6xl space-y-8">
          
          {/* Header Section */}
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-border/80 pb-6">
            <div>
              <h1 className="text-3xl font-display font-bold text-foreground tracking-tight uppercase">
                Audit History Dashboard
              </h1>
              <p className="text-sm text-muted-foreground mt-1">
                View past dataset quality audits, inspect transformation DAGs, and download reports.
              </p>
            </div>

            <Link to="/workspace">
              <Button size="sm" className="rounded-full shadow-premium gap-2">
                <FileSpreadsheet className="h-4 w-4" />
                New Dataset Audit
              </Button>
            </Link>
          </div>

          {/* Search Controls */}
          <div className="flex items-center gap-3 bg-card border border-border/80 rounded-2xl p-2 shadow-xs max-w-md">
            <Search className="h-4 w-4 text-muted-foreground ml-3" />
            <Input
              type="text"
              placeholder="Search dataset name, task ID, or grade..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="border-0 focus-visible:ring-0 text-xs"
            />
          </div>

          {/* Audit Cards List */}
          {filteredItems.length === 0 ? (
            <Card className="border border-border/80 rounded-3xl p-12 text-center bg-card shadow-xs">
              <AlertCircle className="h-10 w-10 text-muted-foreground mx-auto mb-3" />
              <h3 className="text-base font-display font-semibold text-foreground">No Audits Found</h3>
              <p className="text-xs text-muted-foreground mt-1 max-w-sm mx-auto">
                No past audit sessions match your search query. Run a new dataset ingestion in the workspace.
              </p>
            </Card>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {filteredItems.map((item) => (
                <Card key={item.task_id} className="border border-border/80 rounded-2xl bg-card shadow-xs hover:border-primary/40 transition-all duration-200">
                  <CardHeader className="pb-3 flex flex-row items-center justify-between space-y-0">
                    <div className="flex items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 text-primary">
                        <FileSpreadsheet className="h-5 w-5" />
                      </div>
                      <div>
                        <CardTitle className="text-base font-display font-bold text-foreground">
                          {item.filename}
                        </CardTitle>
                        <span className="text-[10px] font-mono text-muted-foreground">ID: {item.task_id.slice(0, 16)}</span>
                      </div>
                    </div>

                    {item.grade && (
                      <Badge variant="outline" className="font-mono text-xs px-2.5 py-0.5 border-primary/30 text-primary bg-primary/5">
                        Grade {item.grade}
                      </Badge>
                    )}
                  </CardHeader>

                  <CardContent className="space-y-4 pt-0">
                    <div className="grid grid-cols-3 gap-2 py-2 border-y border-border/40 text-center">
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
                      <span className="text-[11px] text-muted-foreground flex items-center gap-1">
                        <Calendar className="h-3 w-3" />
                        {item.created_at}
                      </span>

                      <div className="flex items-center gap-2">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-8 text-xs gap-1.5"
                          onClick={async () => {
                            try {
                              const blob = await api.downloadReportBlob(item.task_id);
                              const url = URL.createObjectURL(blob);
                              const a = document.createElement("a");
                              a.href = url;
                              a.download = `${item.task_id}_${item.filename}.pdf`;
                              a.click();
                            } catch (e) {
                              alert("Report PDF not found for this task ID.");
                            }
                          }}
                        >
                          <Download className="h-3.5 w-3.5" />
                          PDF
                        </Button>
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
