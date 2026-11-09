import React from "react";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";
import { SupersetExploreStudio } from "@/components/SupersetExploreStudio";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { Database, Activity, BarChart3 } from "lucide-react";
import { Link } from "react-router-dom";

export const Analytics: React.FC = () => {
  return (
    <div className="min-h-screen bg-background flex flex-col text-foreground font-sans transition-colors duration-200">
      <Header onReset={() => {}} showReset={false} />

      <main className="flex-1 container max-w-6xl mx-auto px-3.5 sm:px-6 pt-20 sm:pt-24 pb-12 sm:pb-16 space-y-6 sm:space-y-8">
        
        {/* Header Breadcrumb & Title */}
        <div className="space-y-2.5 border-b border-border pb-5 sm:pb-6">
          <div className="flex items-center gap-2 text-xs text-muted-foreground font-mono">
            <Link to="/" className="hover:text-foreground transition">GetReport</Link>
            <span>/</span>
            <span className="text-foreground font-semibold">Analytics</span>
          </div>

          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <h1 className="text-xl xs:text-2xl sm:text-3xl font-display font-bold tracking-tight text-foreground">
                Business Intelligence Studio
              </h1>
              <p className="text-xs sm:text-sm text-muted-foreground mt-1 max-w-2xl leading-relaxed">
                Visual slice-and-dice analytics and dynamic exploratory charting for certified datasets.
              </p>
            </div>

            <div className="flex items-center gap-2 sm:gap-3 w-full sm:w-auto">
              <Link
                to="/workspace"
                className="w-full sm:w-auto inline-flex items-center justify-center gap-1.5 px-4 py-2 rounded-xl border border-border bg-card text-xs font-medium text-foreground hover:bg-muted transition shadow-xs"
              >
                <Database className="h-3.5 w-3.5 text-emerald-500" />
                <span>Upload New Dataset</span>
              </Link>
            </div>
          </div>
        </div>

        {/* Native Superset Explore Studio with Dataset Selector */}
        <div className="w-full">
          <ErrorBoundary fallbackTitle="Business Intelligence Studio Error">
            <SupersetExploreStudio allowDatasetSwitching={true} />
          </ErrorBoundary>
        </div>

        {/* Architecture & Features Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4 pt-2">
          <div className="rounded-xl border border-border bg-card p-4 sm:p-5 shadow-xs">
            <div className="flex items-center gap-2.5 mb-2 text-foreground font-semibold text-sm">
              <Database className="h-4 w-4 text-emerald-500 shrink-0" />
              <span>Direct Upload Binding</span>
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Every audited dataset is immediately ready for exploration. Switch between uploaded files without re-importing.
            </p>
          </div>

          <div className="rounded-xl border border-border bg-card p-4 sm:p-5 shadow-xs">
            <div className="flex items-center gap-2.5 mb-2 text-foreground font-semibold text-sm">
              <Activity className="h-4 w-4 text-amber-500 shrink-0" />
              <span>In-Memory Aggregation</span>
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Executes grouped aggregations, metric calculations, and dimension breakdowns in memory in milliseconds.
            </p>
          </div>

          <div className="rounded-xl border border-border bg-card p-4 sm:p-5 shadow-xs sm:col-span-2 lg:col-span-1">
            <div className="flex items-center gap-2.5 mb-2 text-foreground font-semibold text-sm">
              <BarChart3 className="h-4 w-4 text-emerald-500 shrink-0" />
              <span>Apache ECharts Visualizer</span>
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
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
