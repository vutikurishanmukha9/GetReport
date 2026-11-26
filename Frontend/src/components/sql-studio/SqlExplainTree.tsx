import { ChevronRight, Cpu } from "lucide-react";
import type { SqlExplainNodeDto } from "@/services/api";
import type { SqlExplainState } from "./types";

function buildChildKeys(nodes: SqlExplainNodeDto[], parentPath: string): Array<{ key: string; node: SqlExplainNodeDto }> {
  const seen = new Map<string, number>();
  return nodes.map((node) => {
    const prevCount = seen.get(node.name) ?? 0;
    const nextCount = prevCount + 1;
    seen.set(node.name, nextCount);
    return {
      key: `${parentPath}/${node.name}#${nextCount}`,
      node,
    };
  });
}

function ExplainBranch({
  node,
  nodePath,
  depth = 0,
}: {
  node: SqlExplainNodeDto;
  nodePath: string;
  depth?: number;
}) {
  const extraEntries = node.extra_info ? Object.entries(node.extra_info) : [];
  const keyedChildren = buildChildKeys(node.children ?? [], nodePath);

  return (
    <div className="space-y-1.5" style={{ marginLeft: depth > 0 ? "16px" : "0px" }}>
      <div className="rounded-lg border border-stone-200/90 dark:border-white/10 bg-white dark:bg-[#131620] px-3 py-2 space-y-1 shadow-2xs">
        <div className="flex items-center justify-between gap-2">
          <div className="inline-flex items-center gap-1.5 font-mono text-xs font-bold text-[#800000] dark:text-emerald-400">
            {depth > 0 && <ChevronRight className="h-3 w-3 text-stone-400 dark:text-zinc-500" />}
            <span>{node.name}</span>
          </div>
          {node.extra_info?.["Estimated Cardinality"] !== undefined && (
            <span className="px-1.5 py-0.5 rounded bg-stone-100 dark:bg-white/[0.06] text-[10px] font-mono text-stone-600 dark:text-zinc-400 tabular-nums">
              est. {String(node.extra_info["Estimated Cardinality"])} rows
            </span>
          )}
        </div>

        {extraEntries.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-0.5 text-[11px] font-mono text-stone-600 dark:text-zinc-400">
            {extraEntries
              .filter(([key]) => key !== "Estimated Cardinality")
              .slice(0, 4)
              .map(([key, val]) => (
                <div key={key} className="truncate">
                  <span className="text-stone-400 dark:text-zinc-500">{key}: </span>
                  <span className="text-stone-800 dark:text-zinc-200">
                    {Array.isArray(val) ? val.join(", ") : String(val)}
                  </span>
                </div>
              ))}
          </div>
        )}
      </div>

      {keyedChildren.map(({ key, node: child }) => (
        <ExplainBranch key={key} node={child} nodePath={key} depth={depth + 1} />
      ))}
    </div>
  );
}

export function SqlExplainTree({ explainState }: { explainState: SqlExplainState | null }) {
  if (!explainState) {
    return (
      <div className="rounded-xl border border-stone-300/80 dark:border-white/10 bg-white dark:bg-[#0D0F16] p-8 text-center text-xs font-mono text-stone-500 dark:text-zinc-400">
        Click Explain in the toolbar to inspect DuckDB&apos;s physical operator tree.
      </div>
    );
  }

  const keyedRoots = buildChildKeys(explainState.plan, "root");

  return (
    <div className="rounded-xl border border-stone-300/90 dark:border-white/10 bg-[#FAF8F5] dark:bg-[#0B0D13] overflow-hidden">
      <div className="px-3.5 py-2 border-b border-stone-200 dark:border-white/10 bg-white dark:bg-[#12151F] flex items-center justify-between text-xs font-mono">
        <span className="inline-flex items-center gap-1.5 font-semibold text-stone-800 dark:text-zinc-200">
          <Cpu className="h-3.5 w-3.5 text-[#800000] dark:text-emerald-400" />
          Physical Execution Plan
        </span>
        <span className="text-[11px] text-emerald-700 dark:text-emerald-400 tabular-nums">
          Planned in {explainState.latency_ms}ms
        </span>
      </div>

      <div className="p-3.5 space-y-2 max-h-[380px] overflow-auto">
        {keyedRoots.length > 0 ? (
          keyedRoots.map(({ key, node: rootNode }) => (
            <ExplainBranch key={key} node={rootNode} nodePath={key} depth={0} />
          ))
        ) : (
          <pre className="p-3 rounded-lg bg-white dark:bg-[#131620] text-[11px] font-mono text-stone-800 dark:text-zinc-200 overflow-x-auto whitespace-pre-wrap">
            {explainState.raw_plan || "Plan output unavailable for this statement type."}
          </pre>
        )}
      </div>
    </div>
  );
}
