import React from "react";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";
import { SupersetExploreStudio } from "@/components/SupersetExploreStudio";
import { Database, Activity, Cpu, BarChart3 } from "lucide-react";
import { Link } from "react-router-dom";

export const Analytics: React.FC = () => {
  return (
    <div className="min-h-screen bg-[#07080a] flex flex-col text-zinc-100 font-sans">
      <Header onReset={() => {}} showReset={false} />

      <main className="flex-1 container max-w-6xl mx-auto px-4 sm:px-6 pt-24 pb-16 space-y-8">
        
        {/* Header Breadcrumb & Title */}
        <div className="space-y-2 border-b border-white/10 pb-6">
          <div className="flex items-center gap-2 text-xs text-zinc-400 font-mono">
            <Link to="/" className="hover:text-zinc-200 transition">GetReport</Link>
            <span>/</span>
            <span className="text-zinc-200">Analytics</span>
          </div>

          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <h1 className="text-2xl sm:text-3xl font-display font-bold tracking-tight text-white">
                Business Intelligence Studio
              </h1>
              <p className="text-sm text-zinc-400 mt-1">
                Visual slice-and-dice analytics and dynamic exploratory charting for certified datasets.
              </p>
            </div>

            <div className="flex items-center gap-3">
              <Link
                to="/workspace"
                className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl border border-white/10 bg-white/[0.05] text-xs font-medium text-zinc-200 hover:bg-white/[0.09] hover:text-white transition"
              >
                <span>Upload New Dataset</span>
              </Link>
            </div>
          </div>
        </div>

        {/* Native Superset Explore Studio with Dataset Selector */}
        <div>
          <SupersetExploreStudio allowDatasetSwitching={true} />
        </div>

        {/* Architecture & Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4">
          <div className="rounded-xl border border-white/10 bg-[#0c0d12] p-5 shadow-sm">
            <div className="flex items-center gap-2.5 mb-2 text-zinc-200 font-semibold text-sm">
              <Database className="h-4 w-4 text-emerald-400" />
              <span>Direct Upload Binding</span>
            </div>
            <p className="text-xs text-zinc-400 leading-relaxed">
              Every audited dataset is immediately ready for exploration. Switch between uploaded files without re-importing.
            </p>
          </div>

          <div className="rounded-xl border border-white/10 bg-[#0c0d12] p-5 shadow-sm">
            <div className="flex items-center gap-2.5 mb-2 text-zinc-200 font-semibold text-sm">
              <Activity className="h-4 w-4 text-amber-400" />
              <span>In-Memory Aggregation</span>
            </div>
            <p className="text-xs text-zinc-400 leading-relaxed">
              Executes grouped aggregations, metric calculations, and dimension breakdowns in memory in milliseconds.
            </p>
          </div>

          <div className="rounded-xl border border-white/10 bg-[#0c0d12] p-5 shadow-sm">
            <div className="flex items-center gap-2.5 mb-2 text-zinc-200 font-semibold text-sm">
              <BarChart3 className="h-4 w-4 text-emerald-400" />
              <span>Apache ECharts Visualizer</span>
            </div>
            <p className="text-xs text-zinc-400 leading-relaxed">
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