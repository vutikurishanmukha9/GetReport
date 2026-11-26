import { useMemo, useState } from "react";
import {
  BookMarked,
  Check,
  ChevronDown,
  ChevronUp,
  Database,
  History,
  Loader2,
  Pencil,
  RotateCcw,
  Search,
  Trash2,
  X,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import type { GoldenQueryItem } from "@/types/api";
import type {
  InspectorTab,
  QueryHistoryItem,
  SchemaTypeFilter,
  SqlColumnProfile,
  SqlSchemaColumn,
  SqlSchemaTable,
} from "./types";
import { classifySqlColumnType, formatSqlLiteral, isFiniteNumber, quoteSqlIdentifier } from "./utils";

const KIND_BADGE_MAP = {
  numeric: {
    tag: "NUM",
    cls: "bg-sky-500/10 text-sky-800 border-sky-500/25 dark:text-sky-300 dark:border-sky-500/30",
  },
  text: {
    tag: "TXT",
    cls: "bg-emerald-500/10 text-emerald-800 border-emerald-500/25 dark:text-emerald-300 dark:border-emerald-500/30",
  },
  temporal: {
    tag: "DATE",
    cls: "bg-amber-500/10 text-amber-800 border-amber-500/25 dark:text-amber-300 dark:border-amber-500/30",
  },
  all: {
    tag: "COL",
    cls: "bg-stone-100 text-stone-700 border-stone-300 dark:bg-white/10 dark:text-zinc-300 dark:border-white/15",
  },
} as const;

function ColumnProfileMiniCard({
  profile,
  loading,
}: {
  profile?: SqlColumnProfile;
  loading: boolean;
}) {
  if (loading && !profile) {
    return (
      <div className="flex items-center gap-1.5 py-1 text-[10px] font-mono text-stone-500 dark:text-zinc-400">
        <Loader2 className="h-3 w-3 animate-spin" />
        Profiling column...
      </div>
    );
  }
  if (!profile) return null;

  const total = Math.max(1, profile.total_records);
  const fillPct = Math.round((profile.non_null_count / total) * 100);

  return (
    <div className="p-2 rounded-lg border border-stone-200/80 dark:border-white/10 bg-white dark:bg-[#0D0F16] space-y-1.5 text-[10px] font-mono">
      <div className="flex items-center justify-between text-stone-600 dark:text-zinc-300 tabular-nums">
        <span>Fill: {fillPct}%</span>
        <span>~{profile.approx_distinct.toLocaleString()} distinct</span>
      </div>
      <div className="h-1.5 w-full rounded-full bg-stone-200 dark:bg-white/10 overflow-hidden">
        <div
          className="h-full rounded-full bg-[#800000] dark:bg-emerald-400 transition-all"
          style={{ width: `${Math.min(100, Math.max(0, fillPct))}%` }}
        />
      </div>
      <div className="grid grid-cols-2 gap-1 text-stone-500 dark:text-zinc-400 tabular-nums">
        <span className="truncate" title={String(profile.min_value ?? "—")}>
          Min: <strong className="text-stone-800 dark:text-zinc-200">{String(profile.min_value ?? "—")}</strong>
        </span>
        <span className="truncate" title={String(profile.max_value ?? "—")}>
          Max: <strong className="text-stone-800 dark:text-zinc-200">{String(profile.max_value ?? "—")}</strong>
        </span>
        {profile.mean_value !== null && isFiniteNumber(profile.mean_value) && (
          <span className="col-span-2 truncate">
            Mean: <strong className="text-stone-800 dark:text-zinc-200">{profile.mean_value.toFixed(2)}</strong>
          </span>
        )}
      </div>
    </div>
  );
}

function SchemaColumnCard({
  col,
  isExpanded,
  profile,
  loadingProfile,
  onToggleExpand,
  onInsertSnippet,
  onRunPresetSql,
}: {
  col: SqlSchemaColumn;
  isExpanded: boolean;
  profile?: SqlColumnProfile;
  loadingProfile: boolean;
  onToggleExpand: () => void;
  onInsertSnippet: (snippet: string) => void;
  onRunPresetSql: (sql: string) => void;
}) {
  const colKind = classifySqlColumnType(col.type);
  const badgeStyle = KIND_BADGE_MAP[colKind];
  const quoted = quoteSqlIdentifier(col.name);

  return (
    <div className="rounded-xl border border-stone-200/90 dark:border-white/10 bg-white dark:bg-[#131620] hover:border-[#800000]/40 dark:hover:border-white/25 transition-colors overflow-hidden">
      <div className="px-2.5 py-2 flex items-center justify-between gap-1.5">
        <button
          type="button"
          onClick={() => onInsertSnippet(quoted)}
          className="flex items-center gap-2 min-w-0 text-left cursor-pointer bg-transparent border-0 p-0 flex-1"
          title={`Insert ${quoted} at cursor`}
        >
          <span className={cn("px-1.5 py-0.5 rounded text-[9px] font-mono font-bold uppercase shrink-0 border", badgeStyle.cls)}>
            {badgeStyle.tag}
          </span>
          <span className="font-mono text-xs font-medium text-stone-900 dark:text-zinc-100 truncate">
            {col.name}
          </span>
        </button>

        <button
          type="button"
          onClick={onToggleExpand}
          className="h-5 w-5 rounded flex items-center justify-center text-stone-500 hover:text-stone-900 hover:bg-stone-100 dark:text-zinc-400 dark:hover:text-white dark:hover:bg-white/10 cursor-pointer shrink-0"
          aria-label={`Inspect column ${col.name}`}
          title="Inspect stats & sample values"
        >
          {isExpanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
        </button>
      </div>

      {isExpanded && (
        <div className="px-2.5 pb-2.5 pt-2 border-t border-stone-200/70 dark:border-white/10 bg-[#FAF8F5] dark:bg-black/25 space-y-2 text-[11px]">
          <div className="flex items-center justify-between font-mono text-[10px] text-stone-600 dark:text-zinc-400">
            <span>{col.type}</span>
            <span>{col.nullable ? "NULLABLE" : "NOT NULL"}</span>
          </div>

          <ColumnProfileMiniCard profile={profile} loading={loadingProfile} />

          {col.sample_values && col.sample_values.length > 0 && (
            <div className="space-y-1">
              <span className="text-[10px] text-stone-500 dark:text-zinc-400 font-sans block">
                Click sample to insert literal:
              </span>
              <div className="flex flex-wrap gap-1">
                {col.sample_values.slice(0, 3).map((val) => (
                  <button
                    key={`${col.name}-sample-${val}`}
                    type="button"
                    onClick={() => onInsertSnippet(formatSqlLiteral(val, col.type))}
                    className="px-1.5 py-0.5 rounded bg-stone-200/70 hover:bg-stone-300 text-stone-800 dark:bg-white/10 dark:hover:bg-white/20 dark:text-zinc-200 font-mono text-[10px] truncate max-w-[115px] cursor-pointer"
                    title={`Insert ${formatSqlLiteral(val, col.type)}`}
                  >
                    {val}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="flex items-center gap-1 flex-wrap pt-0.5">
            <button
              type="button"
              onClick={() =>
                onRunPresetSql(
                  `SELECT ${quoted},\n  COUNT(*) AS count\nFROM dataset\nGROUP BY ${quoted}\nORDER BY count DESC\nLIMIT 15;`,
                )
              }
              className="px-2 py-0.5 rounded-md text-[10px] font-medium bg-stone-200/80 hover:bg-stone-300 text-stone-800 dark:bg-white/10 dark:hover:bg-white/20 dark:text-zinc-200 cursor-pointer"
            >
              Top Values
            </button>
            {colKind === "numeric" && (
              <button
                type="button"
                onClick={() =>
                  onRunPresetSql(
                    `SELECT\n  ROUND(MIN(${quoted}), 2) AS min_val,\n  ROUND(AVG(${quoted}), 2) AS avg_val,\n  ROUND(MEDIAN(${quoted}), 2) AS median_val,\n  ROUND(MAX(${quoted}), 2) AS max_val\nFROM dataset;`,
                  )
                }
                className="px-2 py-0.5 rounded-md text-[10px] font-medium bg-stone-200/80 hover:bg-stone-300 text-stone-800 dark:bg-white/10 dark:hover:bg-white/20 dark:text-zinc-200 cursor-pointer"
              >
                Summary Stats
              </button>
            )}
            <button
              type="button"
              onClick={() =>
                onRunPresetSql(
                  `SELECT\n  COUNT(*) AS total_rows,\n  COUNT(${quoted}) AS non_null,\n  COUNT(*) - COUNT(${quoted}) AS missing\nFROM dataset;`,
                )
              }
              className="px-2 py-0.5 rounded-md text-[10px] font-medium bg-stone-200/80 hover:bg-stone-300 text-stone-800 dark:bg-white/10 dark:hover:bg-white/20 dark:text-zinc-200 cursor-pointer"
            >
              Null Check
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

interface SqlInspectorSidebarProps {
  selectedTaskId: string;
  activeTableMeta?: SqlSchemaTable;
  schemaColumnsList: SqlSchemaColumn[];
  resolvedRowCount: number;
  loadingSchema: boolean;
  schemaError: string | null;
  columnProfiles: Record<string, SqlColumnProfile>;
  loadingProfileCol: string | null;
  onLoadColumnProfile: (columnName: string) => void;
  savedQueries: GoldenQueryItem[];
  loadingSaved: boolean;
  queryHistory: QueryHistoryItem[];
  onRefreshSchema: (taskId: string) => void;
  onInsertSnippet: (snippet: string) => void;
  onLoadSql: (sql: string, isSaved: boolean) => void;
  onRunSql: (sql: string, markSaved: boolean) => void;
  onRenameSaved: (queryId: string, question: string) => void;
  onDeleteSaved: (queryId: string) => void;
}

export function SqlInspectorSidebar({
  selectedTaskId,
  activeTableMeta,
  schemaColumnsList,
  resolvedRowCount,
  loadingSchema,
  schemaError,
  columnProfiles,
  loadingProfileCol,
  onLoadColumnProfile,
  savedQueries,
  loadingSaved,
  queryHistory,
  onRefreshSchema,
  onInsertSnippet,
  onLoadSql,
  onRunSql,
  onRenameSaved,
  onDeleteSaved,
}: SqlInspectorSidebarProps) {
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>("schema");
  const [schemaSearch, setSchemaSearch] = useState<string>("");
  const [schemaTypeFilter, setSchemaTypeFilter] = useState<SchemaTypeFilter>("all");
  const [expandedSchemaCol, setExpandedSchemaCol] = useState<string | null>(null);
  const [editingKpiId, setEditingKpiId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState<string>("");

  const filteredSchemaColumns = useMemo(() => {
    const searchLower = schemaSearch.toLowerCase().trim();
    return schemaColumnsList.filter((col) => {
      const matchesSearch =
        !searchLower || col.name.toLowerCase().includes(searchLower) || col.type.toLowerCase().includes(searchLower);
      if (!matchesSearch) return false;
      if (schemaTypeFilter === "all") return true;
      return classifySqlColumnType(col.type) === schemaTypeFilter;
    });
  }, [schemaColumnsList, schemaSearch, schemaTypeFilter]);

  return (
    <aside className="h-full border-b lg:border-b-0 lg:border-r border-stone-300/80 dark:border-white/10 bg-[#F3EFE6]/70 dark:bg-[#0E1016] p-3.5 flex flex-col space-y-3">
      <div className="grid grid-cols-3 gap-1 p-1 rounded-xl bg-stone-200/80 dark:bg-black/40 border border-stone-300/80 dark:border-white/10 text-[11px] font-sans font-semibold">
        <button
          type="button"
          onClick={() => setInspectorTab("schema")}
          className={cn(
            "py-1.5 px-2 rounded-lg flex items-center justify-center gap-1 transition-all cursor-pointer",
            inspectorTab === "schema"
              ? "bg-white text-stone-900 shadow-xs dark:bg-zinc-800 dark:text-white"
              : "text-stone-600 hover:text-stone-900 dark:text-zinc-400 dark:hover:text-zinc-200",
          )}
        >
          <Database className="h-3 w-3 shrink-0" />
          <span>Schema</span>
        </button>
        <button
          type="button"
          onClick={() => setInspectorTab("saved")}
          className={cn(
            "py-1.5 px-2 rounded-lg flex items-center justify-center gap-1 transition-all cursor-pointer",
            inspectorTab === "saved"
              ? "bg-white text-stone-900 shadow-xs dark:bg-zinc-800 dark:text-white"
              : "text-stone-600 hover:text-stone-900 dark:text-zinc-400 dark:hover:text-zinc-200",
          )}
        >
          <BookMarked className="h-3 w-3 shrink-0" />
          <span>Saved</span>
          {savedQueries.length > 0 && (
            <span className="px-1 rounded bg-stone-200 dark:bg-white/10 text-[10px] font-mono tabular-nums">
              {savedQueries.length}
            </span>
          )}
        </button>
        <button
          type="button"
          onClick={() => setInspectorTab("history")}
          className={cn(
            "py-1.5 px-2 rounded-lg flex items-center justify-center gap-1 transition-all cursor-pointer",
            inspectorTab === "history"
              ? "bg-white text-stone-900 shadow-xs dark:bg-zinc-800 dark:text-white"
              : "text-stone-600 hover:text-stone-900 dark:text-zinc-400 dark:hover:text-zinc-200",
          )}
        >
          <History className="h-3 w-3 shrink-0" />
          <span>History</span>
        </button>
      </div>

      {inspectorTab === "schema" && (
        <div className="flex flex-col space-y-2.5 flex-1 min-h-0">
          <div className="p-2.5 rounded-xl bg-white dark:bg-[#131620] border border-stone-200 dark:border-white/10 flex items-center justify-between gap-2">
            <div className="min-w-0">
              <div className="flex items-center gap-1.5">
                <span className="font-mono text-xs font-bold text-stone-900 dark:text-white">
                  {activeTableMeta?.table_name || "dataset"}
                </span>
                <Badge variant="outline" className="text-[10px] font-mono px-1.5 py-0 border-stone-300 dark:border-white/15">
                  {schemaColumnsList.length} cols
                </Badge>
              </div>
              <span className="text-[10px] font-mono text-stone-500 dark:text-zinc-400 tabular-nums block mt-0.5">
                {resolvedRowCount.toLocaleString()} rows in Arrow view
              </span>
            </div>
            <div className="flex items-center gap-1 shrink-0">
              <button
                type="button"
                onClick={() => onRunSql(`DESCRIBE ${activeTableMeta?.table_name || "dataset"};`, false)}
                disabled={!selectedTaskId}
                className="px-2 py-1 rounded-md text-[10px] font-mono font-semibold bg-stone-100 hover:bg-stone-200 text-stone-800 border border-stone-300 dark:bg-white/[0.06] dark:hover:bg-white/[0.12] dark:text-zinc-200 dark:border-white/10 cursor-pointer disabled:opacity-40"
                title="Run DESCRIBE on active table"
              >
                DESCRIBE
              </button>
              {selectedTaskId && (
                <button
                  type="button"
                  onClick={() => onRefreshSchema(selectedTaskId)}
                  disabled={loadingSchema}
                  className="p-1 rounded-md text-stone-500 hover:text-stone-900 hover:bg-stone-100 dark:text-zinc-400 dark:hover:text-white dark:hover:bg-white/10 cursor-pointer"
                  title="Refresh DuckDB schema metadata"
                  aria-label="Refresh schema"
                >
                  <RotateCcw className={cn("h-3 w-3", loadingSchema && "animate-spin")} />
                </button>
              )}
            </div>
          </div>

          <div className="relative">
            <Search className="h-3.5 w-3.5 absolute left-2.5 top-2 text-stone-400 dark:text-zinc-500" />
            <Input
              type="text"
              placeholder="Filter columns..."
              aria-label="Filter schema columns"
              value={schemaSearch}
              onChange={(e) => setSchemaSearch(e.target.value)}
              className="h-7.5 pl-8 text-xs font-mono bg-white dark:bg-[#131620] border-stone-300 dark:border-white/10"
            />
          </div>

          <div className="flex items-center gap-1 flex-wrap">
            {(["all", "numeric", "text", "temporal"] as const).map((filterKey) => (
              <button
                key={filterKey}
                type="button"
                onClick={() => setSchemaTypeFilter(filterKey)}
                className={cn(
                  "px-2 py-0.5 rounded-md text-[10px] font-mono font-semibold transition-colors cursor-pointer uppercase",
                  schemaTypeFilter === filterKey
                    ? "bg-[#800000] text-white dark:bg-white dark:text-zinc-950"
                    : "bg-stone-200/70 text-stone-700 hover:bg-stone-300/70 dark:bg-white/[0.05] dark:text-zinc-400 dark:hover:text-white",
                )}
              >
                {filterKey}
              </button>
            ))}
          </div>

          {schemaError && (
            <div className="p-2.5 rounded-xl bg-amber-500/10 border border-amber-500/30 text-[11px] font-sans text-amber-900 dark:text-amber-300">
              {schemaError}
            </div>
          )}

          <div className="overflow-y-auto space-y-1.5 pr-1 max-h-[480px]">
            {loadingSchema ? (
              <div className="space-y-2 py-2" aria-label="Loading schema columns">
                {[1, 2, 3, 4, 5].map((item) => (
                  <div key={item} className="h-8 rounded-lg bg-stone-200/70 dark:bg-white/[0.05] animate-pulse" />
                ))}
              </div>
            ) : filteredSchemaColumns.length === 0 ? (
              <div className="py-8 text-center text-xs text-stone-500 dark:text-zinc-400 font-sans">
                {selectedTaskId ? "No columns match filter." : "No active dataset selected."}
              </div>
            ) : (
              filteredSchemaColumns.map((col) => {
                const isExpanded = expandedSchemaCol === col.name;
                return (
                  <SchemaColumnCard
                    key={col.name}
                    col={col}
                    isExpanded={isExpanded}
                    profile={columnProfiles[`${selectedTaskId}:${col.name}`]}
                    loadingProfile={loadingProfileCol === col.name}
                    onToggleExpand={() => {
                      const nextCol = isExpanded ? null : col.name;
                      setExpandedSchemaCol(nextCol);
                      if (nextCol) onLoadColumnProfile(col.name);
                    }}
                    onInsertSnippet={onInsertSnippet}
                    onRunPresetSql={(sql) => onRunSql(sql, false)}
                  />
                );
              })
            )}
          </div>
        </div>
      )}

      {inspectorTab === "saved" && (
        <div className="space-y-2 overflow-y-auto max-h-[540px] pr-1">
          {loadingSaved ? (
            <div className="py-8 text-center text-xs text-stone-500 dark:text-zinc-400">
              <Loader2 className="h-4 w-4 animate-spin mx-auto mb-2" />
              Loading saved KPIs…
            </div>
          ) : savedQueries.length === 0 ? (
            <div className="p-4 rounded-xl border border-dashed border-stone-300 dark:border-white/15 text-center space-y-1.5">
              <BookMarked className="h-5 w-5 mx-auto text-stone-400 dark:text-zinc-500" />
              <p className="text-xs font-medium text-stone-700 dark:text-zinc-300">No Saved Golden KPIs Yet</p>
              <p className="text-[11px] text-stone-500 dark:text-zinc-400 leading-relaxed">
                Run any analytical query and click &ldquo;Save KPI&rdquo; to pin it here.
              </p>
            </div>
          ) : (
            savedQueries.map((gq) => {
              const isEditing = editingKpiId === gq.id;
              return (
                <div
                  key={gq.id}
                  className="p-2.5 rounded-xl border border-stone-200 dark:border-white/10 bg-white dark:bg-[#131620] space-y-2"
                >
                  <div className="flex items-start justify-between gap-2">
                    {isEditing ? (
                      <div className="flex items-center gap-1 flex-1">
                        <Input
                          value={editingTitle}
                          onChange={(e) => setEditingTitle(e.target.value)}
                          className="h-6 text-xs px-2"
                          aria-label="Rename Golden KPI"
                        />
                        <button
                          type="button"
                          onClick={() => {
                            onRenameSaved(gq.id, editingTitle);
                            setEditingKpiId(null);
                          }}
                          className="p-1 text-emerald-700 hover:bg-emerald-50 rounded cursor-pointer"
                          title="Save title"
                        >
                          <Check className="h-3 w-3" />
                        </button>
                        <button
                          type="button"
                          onClick={() => setEditingKpiId(null)}
                          className="p-1 text-stone-500 hover:bg-stone-100 rounded cursor-pointer"
                          title="Cancel"
                        >
                          <X className="h-3 w-3" />
                        </button>
                      </div>
                    ) : (
                      <>
                        <span className="text-xs font-semibold text-stone-900 dark:text-white leading-snug">
                          {gq.question}
                        </span>
                        <div className="flex items-center gap-0.5 shrink-0">
                          <button
                            type="button"
                            onClick={() => {
                              setEditingKpiId(gq.id);
                              setEditingTitle(gq.question);
                            }}
                            aria-label={`Rename benchmark ${gq.question}`}
                            title="Rename benchmark"
                            className="text-stone-400 hover:text-stone-800 dark:hover:text-zinc-200 p-0.5 rounded cursor-pointer"
                          >
                            <Pencil className="h-3 w-3" />
                          </button>
                          <button
                            type="button"
                            onClick={() => onDeleteSaved(gq.id)}
                            aria-label={`Delete benchmark ${gq.question}`}
                            title="Delete saved benchmark"
                            className="text-stone-400 hover:text-destructive p-0.5 rounded cursor-pointer"
                          >
                            <Trash2 className="h-3 w-3" />
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                  <pre className="p-2 rounded-lg bg-[#FAF8F5] dark:bg-black/30 border border-stone-200 dark:border-white/5 text-[10px] font-mono text-stone-700 dark:text-zinc-300 overflow-x-auto max-h-20 whitespace-pre-wrap break-all">
                    {gq.sql_query}
                  </pre>
                  <div className="flex items-center gap-1.5">
                    <button
                      type="button"
                      onClick={() => onLoadSql(gq.sql_query, true)}
                      className="px-2 py-1 rounded-md text-[10px] font-medium bg-stone-100 hover:bg-stone-200 text-stone-800 dark:bg-white/10 dark:hover:bg-white/15 dark:text-zinc-200 cursor-pointer"
                    >
                      Load
                    </button>
                    <button
                      type="button"
                      onClick={() => onRunSql(gq.sql_query, true)}
                      className="px-2 py-1 rounded-md text-[10px] font-semibold bg-[#800000] text-white hover:bg-[#6b0000] dark:bg-white dark:text-zinc-950 dark:hover:bg-zinc-200 cursor-pointer"
                    >
                      Run
                    </button>
                  </div>
                </div>
              );
            })
          )}
        </div>
      )}

      {inspectorTab === "history" && (
        <div className="space-y-2 overflow-y-auto max-h-[540px] pr-1">
          {queryHistory.map((item) => (
            <div
              key={item.id}
              className="p-2.5 rounded-xl border border-stone-200 dark:border-white/10 bg-white dark:bg-[#131620] space-y-1.5"
            >
              <div className="flex items-center justify-between gap-2 text-[10px] font-mono">
                <span className="text-stone-500 dark:text-zinc-400">{item.executedAt}</span>
                <div className="flex items-center gap-1.5 tabular-nums">
                  {item.latencyMs !== undefined && (
                    <span className="text-emerald-700 dark:text-emerald-400 font-semibold">{item.latencyMs}ms</span>
                  )}
                  {item.rowCount !== undefined && (
                    <span className="text-stone-500 dark:text-zinc-400">{item.rowCount} rows</span>
                  )}
                </div>
              </div>
              <pre className="text-[11px] font-mono text-stone-800 dark:text-zinc-200 whitespace-pre-wrap break-all line-clamp-3">
                {item.sql}
              </pre>
              <div className="flex items-center gap-1.5 pt-0.5">
                <button
                  type="button"
                  onClick={() => onLoadSql(item.sql, false)}
                  className="px-2 py-0.5 rounded-md text-[10px] font-medium bg-stone-100 hover:bg-stone-200 text-stone-800 dark:bg-white/10 dark:hover:bg-white/15 dark:text-zinc-200 cursor-pointer"
                >
                  Load
                </button>
                <button
                  type="button"
                  onClick={() => onRunSql(item.sql, false)}
                  className="px-2 py-0.5 rounded-md text-[10px] font-semibold bg-stone-900 text-white hover:bg-stone-800 dark:bg-white dark:text-zinc-950 cursor-pointer"
                >
                  Re-run
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </aside>
  );
}
