const DEFAULT_GRADE_COLOR =
  "bg-stone-200/80 text-stone-800 border-stone-300 dark:bg-white/[0.04] dark:text-zinc-300 dark:border-white/10";

const EMERALD_GRADE_COLOR =
  "bg-emerald-500/15 text-emerald-800 border-emerald-500/30 dark:bg-emerald-950/40 dark:text-emerald-400 dark:border-emerald-500/30";

const GRADE_COLOR_MAP = new Map<string, string>([
  ["A", EMERALD_GRADE_COLOR],
  ["B", EMERALD_GRADE_COLOR],
  ["C", "bg-amber-500/15 text-amber-800 border-amber-500/30 dark:bg-amber-950/40 dark:text-amber-400 dark:border-amber-500/30"],
  ["D", "bg-orange-500/15 text-orange-800 border-orange-500/30 dark:bg-orange-950/40 dark:text-orange-400 dark:border-orange-500/30"],
  ["F", "bg-rose-500/15 text-rose-800 border-rose-500/30 dark:bg-rose-950/40 dark:text-rose-400 dark:border-rose-500/30"],
]);

export function getGradeColorClass(grade: string): string {
  return GRADE_COLOR_MAP.get((grade || "").toUpperCase()) ?? DEFAULT_GRADE_COLOR;
}

interface SealGradeConfig {
  container: string;
  label: string;
  grade: string;
  score: string;
}

const SEAL_EMERALD_CONFIG: SealGradeConfig = {
  container:
    "bg-emerald-50 border-emerald-300/80 text-emerald-900 dark:bg-emerald-950/40 dark:border-emerald-500/30 dark:text-emerald-400",
  label: "text-emerald-800 dark:text-emerald-400/80",
  grade: "text-emerald-900 dark:text-emerald-400",
  score: "text-emerald-800 dark:text-emerald-300/90",
};

const SEAL_GRADE_MAP = new Map<string, SealGradeConfig>([
  ["A", SEAL_EMERALD_CONFIG],
  ["B", SEAL_EMERALD_CONFIG],
  [
    "C",
    {
      container:
        "bg-amber-50 border-amber-300/80 text-amber-900 dark:bg-amber-950/40 dark:border-amber-500/30 dark:text-amber-400",
      label: "text-amber-800 dark:text-amber-400/80",
      grade: "text-amber-900 dark:text-amber-400",
      score: "text-amber-800 dark:text-amber-300/90",
    },
  ],
  [
    "D",
    {
      container:
        "bg-orange-50 border-orange-300/80 text-orange-900 dark:bg-orange-950/40 dark:border-orange-500/30 dark:text-orange-400",
      label: "text-orange-800 dark:text-orange-400/80",
      grade: "text-orange-900 dark:text-orange-400",
      score: "text-orange-800 dark:text-orange-300/90",
    },
  ],
  [
    "F",
    {
      container:
        "bg-rose-50 border-rose-300/80 text-rose-900 dark:bg-rose-950/40 dark:border-rose-500/30 dark:text-rose-400",
      label: "text-rose-800 dark:text-rose-400/80",
      grade: "text-rose-900 dark:text-rose-400",
      score: "text-rose-800 dark:text-rose-300/90",
    },
  ],
]);

const DEFAULT_SEAL_CONFIG: SealGradeConfig = {
  container:
    "bg-stone-200/70 border-stone-300 text-stone-800 dark:bg-white/[0.04] dark:border-white/10 dark:text-zinc-300",
  label: "text-stone-700/80 dark:text-zinc-400/80",
  grade: "text-stone-800 dark:text-zinc-200",
  score: "text-stone-700 dark:text-zinc-400",
};

export function getSealGradeConfig(grade: string): SealGradeConfig {
  return SEAL_GRADE_MAP.get((grade || "B").toUpperCase()) ?? DEFAULT_SEAL_CONFIG;
}

const VALID_MAIN_TABS = new Set([
  "overview",
  "charts",
  "ledger",
  "stats",
  "sql",
  "bi_analytics",
  "export",
]);

export const MAIN_TAB_TRIGGER_CLASS =
  "group flex-auto justify-center min-h-[38px] rounded-xl text-xs px-3 py-2 font-sans font-semibold data-[state=active]:bg-[#800000] data-[state=active]:text-white data-[state=active]:border-[#800000] data-[state=active]:shadow-md text-stone-900 hover:text-stone-950 hover:bg-stone-300/70 dark:data-[state=active]:bg-white dark:data-[state=active]:text-zinc-950 dark:data-[state=active]:border-white dark:text-zinc-300 dark:hover:text-white dark:hover:bg-white/[0.08] transition-all inline-flex items-center gap-1.5 whitespace-nowrap cursor-pointer border border-transparent";

export const TAB_COUNT_BADGE_CLASS =
  "tab-count-badge inline-flex items-center justify-center h-[18px] min-w-[18px] px-1.5 rounded-full text-[10px] leading-none font-sans font-bold tabular-nums bg-stone-300 text-stone-900 border border-stone-400/40 group-data-[state=active]:bg-white group-data-[state=active]:text-[#800000] group-data-[state=active]:border-transparent dark:bg-white/10 dark:text-zinc-300 dark:group-data-[state=active]:bg-zinc-950 dark:group-data-[state=active]:text-white";

export function normalizeMainTab(tab?: string | null): string {
  if (tab && VALID_MAIN_TABS.has(tab)) return tab;
  return "overview";
}
