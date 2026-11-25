import React, { Suspense, lazy } from "react";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";
import { SupersetExploreStudio } from "@/components/SupersetExploreStudio";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { Database, Activity, BarChart3, Terminal, Loader2 } from "lucide-react";
import { Link, useSearchParams } from "react-router-dom";
import { cn } from "@/lib/utils";

const DuckDBSqlStudio = lazy(() => import("@/components/DuckDBSqlStudio"));

export const Analytics: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const initialTaskId = searchParams.get("taskId") || undefined;
  const studioParam = searchParams.get("studio");
  const studioMode: "bi" | "sql" = studioParam === "sql" ? "sql" : "bi";
  const initialSql = searchParams.get("sql") || undefined;

  const switchStudioMode = (nextMode: "bi" | "sql") => {
    const nextParams = new URLSearchParams(searchParams);
    nextParams.set("studio", nextMode);
    if (nextMode !== "sql") {
      nextParams.delete("sql");
    }
    setSearchParams(nextParams, { replace: true });
  };

  return (
    <div className="min-h-screen bg-[#F5F2EB] dark:bg-[#07080a] flex flex-col text-stone-900 dark:text-zinc-100 font-sans">
      <Header onReset={() => { }} showReset={false} />

      <main className="flex-1 container max-w-7xl mx-auto px-3.5 sm:px-6 pt-20 sm:pt-24 pb-12 sm:pb-16 space-y-6 sm:space-y-8">

        {/* Header Breadcrumb & Title */}
        <div className="space-y-3 border-b border-stone-300 dark:border-white/10 pb-5 sm:pb-6">
          <div className="flex items-center gap-2 text-xs text-stone-600 dark:text-zinc-400 font-mono">
            <Link to="/" className="hover:text-stone-900 dark:hover:text-zinc-200 transition">GetReport</Link>
            <span>/</span>
            <span className="text-stone-900 dark:text-zinc-200">
              {studioMode === "sql" ? "DuckDB SQL Studio" : "Business Intelligence Studio"}
            </span>
          </div>

          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
            <div>
              <h1 className="text-xl xs:text-2xl sm:text-3xl font-display font-bold tracking-tight text-stone-900 dark:text-white">
                {studioMode === "sql" ? "DuckDB Analytical SQL Studio" : "Business Intelligence Studio"}
              </h1>
              <p className="text-xs sm:text-sm text-stone-600 dark:text-zinc-400 mt-1 max-w-2xl leading-relaxed">
                {studioMode === "sql"
                  ? "Query audited Arrow/Parquet datasets directly with schema introspection, Golden KPIs, and safe read-only SQL execution."
                  : "Visual slice-and-dice analytics and dynamic exploratory charting for certified datasets."}
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-2.5">
              {/* Studio Workbench Mode Switcher */}
              <div className="inline-flex items-center p-1 rounded-xl bg-[#EDE8DE] border border-[#DCD6C9] dark:bg-[#11131A] dark:border-white/10">
                <button
                  type="button"
                  onClick={() => switchStudioMode("bi")}
                  className={cn(
                    "px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer",
                    studioMode === "bi"
                      ? "bg-[#800000] text-white shadow-xs dark:bg-white dark:text-zinc-950"
                      : "text-stone-700 hover:text-stone-950 dark:text-zinc-400 dark:hover:text-white",
                  )}
                >
                  <BarChart3 className="h-3.5 w-3.5" />
                  <span>Visual BI Studio</span>
                </button>
                <button
                  type="button"
                  onClick={() => switchStudioMode("sql")}
                  className={cn(
                    "px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer",
                    studioMode === "sql"
                      ? "bg-[#800000] text-white shadow-xs dark:bg-white dark:text-zinc-950"
                      : "text-stone-700 hover:text-stone-950 dark:text-zinc-400 dark:hover:text-white",
                  )}
                >
                  <Terminal className="h-3.5 w-3.5" />
                  <span>DuckDB SQL Studio</span>
                </button>
              </div>

              <Link
                to="/workspace"
                className="inline-flex items-center justify-center gap-1.5 px-4 py-2 rounded-xl border border-stone-300 bg-stone-200/90 text-xs font-semibold text-stone-900 hover:bg-stone-300 dark:border-white/10 dark:bg-white/[0.05] dark:text-zinc-200 dark:hover:bg-white/[0.09] dark:hover:text-white transition shadow-sm"
              >
                <Database className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />
                <span>Upload New Dataset</span>
              </Link>
            </div>
          </div>
        </div>

        {/* Active Studio Workbench */}
        <div className="w-full">
          {studioMode === "sql" ? (
            <ErrorBoundary fallbackTitle="DuckDB SQL Studio Error">
              <Suspense
                fallback={
                  <div className="flex flex-col items-center justify-center p-12 space-y-3 rounded-2xl border border-stone-300 dark:border-white/10 bg-[#FAF8F5] dark:bg-[#0B0C10]">
                    <Loader2 className="h-7 w-7 text-[#800000] dark:text-emerald-400 animate-spin" />
                    <span className="text-xs font-mono text-stone-600 dark:text-zinc-400 uppercase tracking-widest">
                      Initializing DuckDB SQL Studio...
                    </span>
                  </div>
                }
              >
                <DuckDBSqlStudio
                  taskId={initialTaskId}
                  allowDatasetSwitching={true}
                  initialSql={initialSql}
                  onTaskChange={(nextTaskId) => {
                    const nextParams = new URLSearchParams(searchParams);
                    nextParams.set("taskId", nextTaskId);
                    nextParams.set("studio", "sql");
                    setSearchParams(nextParams, { replace: true });
                  }}
                />
              </Suspense>
            </ErrorBoundary>
          ) : (
            <ErrorBoundary fallbackTitle="Business Intelligence Studio Error">
              <SupersetExploreStudio taskId={initialTaskId} allowDatasetSwitching={true} />
            </ErrorBoundary>
          )}
        </div>

        {/* Architecture & Features Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4 pt-2">
          <div className="rounded-xl border border-stone-200/90 bg-[#FAF8F5] dark:border-white/10 dark:bg-[#0c0d12] p-4 sm:p-5 shadow-sm">
            <div className="flex items-center gap-2.5 mb-2 text-stone-900 dark:text-zinc-200 font-semibold text-sm">
              <Database className="h-4 w-4 text-emerald-600 dark:text-emerald-400 shrink-0" />
              <span>Direct Upload Binding</span>
            </div>
            <p className="text-xs text-stone-600 dark:text-zinc-400 leading-relaxed">
              Every audited dataset is immediately ready for exploration. Switch between uploaded files without re-importing.
            </p>
          </div>

          <div className="rounded-xl border border-stone-200/90 bg-[#FAF8F5] dark:border-white/10 dark:bg-[#0c0d12] p-4 sm:p-5 shadow-sm">
            <div className="flex items-center gap-2.5 mb-2 text-stone-900 dark:text-zinc-200 font-semibold text-sm">
              <Activity className="h-4 w-4 text-amber-600 dark:text-amber-400 shrink-0" />
              <span>In-Memory Aggregation</span>
            </div>
            <p className="text-xs text-stone-600 dark:text-zinc-400 leading-relaxed">
              Executes grouped aggregations, metric calculations, and dimension breakdowns in memory in milliseconds.
            </p>
          </div>

          <div className="rounded-xl border border-stone-200/90 bg-[#FAF8F5] dark:border-white/10 dark:bg-[#0c0d12] p-4 sm:p-5 shadow-sm sm:col-span-2 lg:col-span-1">
            <div className="flex items-center gap-2.5 mb-2 text-stone-900 dark:text-zinc-200 font-semibold text-sm">
              <BarChart3 className="h-4 w-4 text-emerald-600 dark:text-emerald-400 shrink-0" />
              <span>Apache ECharts Visualizer</span>
            </div>
            <p className="text-xs text-stone-600 dark:text-zinc-400 leading-relaxed">
              Interactive visual specifications for grouped bars, trendlines, area fills, treemaps, and proportional breakdowns.
            </p>
          </div>
        </div>

      </main>

      <Footer />
    </div>
  );
};

export default Analytics;