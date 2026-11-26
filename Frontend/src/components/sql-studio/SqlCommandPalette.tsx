import { BookMarked, Code2, Columns3, Cpu, Download, History, Play } from "lucide-react";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
  CommandShortcut,
} from "@/components/ui/command";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import type { GoldenQueryItem } from "@/types/api";
import type { QueryHistoryItem, QueryRecipe, SqlSchemaColumn } from "./types";
import { quoteSqlIdentifier } from "./utils";

interface SqlCommandPaletteProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  columns: SqlSchemaColumn[];
  recipes: QueryRecipe[];
  savedQueries: GoldenQueryItem[];
  history: QueryHistoryItem[];
  onInsertSnippet: (snippet: string) => void;
  onLoadAndRunSql: (sql: string, isSaved: boolean) => void;
  onRunAction: (action: "run" | "format" | "explain" | "export" | "clear") => void;
}

export function SqlCommandPalette({
  open,
  onOpenChange,
  columns,
  recipes,
  savedQueries,
  history,
  onInsertSnippet,
  onLoadAndRunSql,
  onRunAction,
}: SqlCommandPaletteProps) {
  const closeAnd = (fn: () => void) => {
    onOpenChange(false);
    fn();
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="overflow-hidden p-0 shadow-xl max-w-xl border-stone-300 dark:border-white/15 bg-white dark:bg-[#11131B]">
        <DialogTitle className="sr-only">SQL Studio Command Palette</DialogTitle>
        <DialogDescription className="sr-only">
          Search DuckDB columns, SQL recipes, saved KPIs, query history, and studio commands.
        </DialogDescription>
        <Command className="[&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:font-mono [&_[cmdk-group-heading]]:text-[10px] [&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-wider [&_[cmdk-group-heading]]:text-stone-500 dark:[&_[cmdk-group-heading]]:text-zinc-400">
          <CommandInput placeholder="Search columns, recipes, saved KPIs, or actions..." className="font-mono text-xs" />
          <CommandList>
            <CommandEmpty>No matching SQL commands or columns.</CommandEmpty>

            <CommandGroup heading="Studio Actions">
              <CommandItem onSelect={() => closeAnd(() => onRunAction("run"))}>
                <Play className="mr-2 h-3.5 w-3.5 text-[#800000] dark:text-emerald-400" />
                <span>Run Query</span>
                <CommandShortcut>Ctrl+Enter</CommandShortcut>
              </CommandItem>
              <CommandItem onSelect={() => closeAnd(() => onRunAction("explain"))}>
                <Cpu className="mr-2 h-3.5 w-3.5" />
                <span>Explain Query Plan</span>
              </CommandItem>
              <CommandItem onSelect={() => closeAnd(() => onRunAction("format"))}>
                <Code2 className="mr-2 h-3.5 w-3.5" />
                <span>Format SQL</span>
              </CommandItem>
              <CommandItem onSelect={() => closeAnd(() => onRunAction("export"))}>
                <Download className="mr-2 h-3.5 w-3.5" />
                <span>Export CSV</span>
              </CommandItem>
            </CommandGroup>

            {columns.length > 0 && (
              <>
                <CommandSeparator />
                <CommandGroup heading="Insert Column Identifier">
                  {columns.map((col) => (
                    <CommandItem
                      key={`cmd-col-${col.name}`}
                      onSelect={() => closeAnd(() => onInsertSnippet(quoteSqlIdentifier(col.name)))}
                    >
                      <Columns3 className="mr-2 h-3.5 w-3.5 text-stone-500" />
                      <span className="font-mono text-xs">{col.name}</span>
                      <CommandShortcut>{col.type}</CommandShortcut>
                    </CommandItem>
                  ))}
                </CommandGroup>
              </>
            )}

            {recipes.length > 0 && (
              <>
                <CommandSeparator />
                <CommandGroup heading="Analytical Recipes">
                  {recipes.map((recipe) => (
                    <CommandItem
                      key={`cmd-recipe-${recipe.id}`}
                      onSelect={() => closeAnd(() => onLoadAndRunSql(recipe.sql, false))}
                    >
                      <Code2 className="mr-2 h-3.5 w-3.5 text-[#800000] dark:text-emerald-400" />
                      <span>{recipe.label}</span>
                    </CommandItem>
                  ))}
                </CommandGroup>
              </>
            )}

            {savedQueries.length > 0 && (
              <>
                <CommandSeparator />
                <CommandGroup heading="Saved Golden KPIs">
                  {savedQueries.map((gq) => (
                    <CommandItem
                      key={`cmd-kpi-${gq.id}`}
                      onSelect={() => closeAnd(() => onLoadAndRunSql(gq.sql_query, true))}
                    >
                      <BookMarked className="mr-2 h-3.5 w-3.5 text-amber-600 dark:text-amber-400" />
                      <span className="truncate">{gq.question}</span>
                    </CommandItem>
                  ))}
                </CommandGroup>
              </>
            )}

            {history.length > 0 && (
              <>
                <CommandSeparator />
                <CommandGroup heading="Recent History">
                  {history.slice(0, 6).map((item) => (
                    <CommandItem
                      key={`cmd-hist-${item.id}`}
                      onSelect={() => closeAnd(() => onLoadAndRunSql(item.sql, false))}
                    >
                      <History className="mr-2 h-3.5 w-3.5 text-stone-400" />
                      <span className="font-mono text-xs truncate">{item.sql}</span>
                    </CommandItem>
                  ))}
                </CommandGroup>
              </>
            )}
          </CommandList>
        </Command>
      </DialogContent>
    </Dialog>
  );
}
