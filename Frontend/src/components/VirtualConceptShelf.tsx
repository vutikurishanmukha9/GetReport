import React, { useState, useEffect } from "react";
import { 
  Boxes, 
  Plus, 
  Calculator, 
  CheckCircle2, 
  Loader2, 
  Clock,
  Layers
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { api } from "@/services/api";

interface ConceptItem {
  concept_name: string;
  formula_or_intent: string;
  description: string;
  node_id: string;
  timestamp: string;
  duration_ms: number;
  expression?: string;
}

interface VirtualConceptShelfProps {
  taskId: string;
  columns?: string[];
  onConceptDerived?: (conceptName: string) => void;
}

export const VirtualConceptShelf: React.FC<VirtualConceptShelfProps> = ({
  taskId,
  columns = [],
  onConceptDerived,
}) => {
  const { toast } = useToast();
  const [concepts, setConcepts] = useState<ConceptItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [deriving, setDeriving] = useState<boolean>(false);
  const [conceptName, setConceptName] = useState<string>("");
  const [formulaOrIntent, setFormulaOrIntent] = useState<string>("");
  const [description, setDescription] = useState<string>("");
  const [isExpanded, setIsExpanded] = useState<boolean>(true);

  // Fetch derived concepts on mount
  useEffect(() => {
    let isMounted = true;
    const loadConcepts = async () => {
      try {
        setLoading(true);
        const list = await api.getDerivedConcepts(taskId);
        if (isMounted) {
          setConcepts(list || []);
        }
      } catch (err: any) {
        console.error("Failed to load derived concepts:", err);
      } finally {
        if (isMounted) setLoading(false);
      }
    };
    if (taskId) {
      loadConcepts();
    }
    return () => {
      isMounted = false;
    };
  }, [taskId]);

  const handleDerive = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const cleanName = conceptName.trim();
    const cleanFormula = formulaOrIntent.trim();

    if (!cleanName || !cleanFormula) {
      toast({
        title: "Incomplete Concept Definition",
        description: "Please specify both a concept name and a formula or intent description.",
        variant: "destructive",
      });
      return;
    }

    try {
      setDeriving(true);
      const res = await api.deriveVirtualConcept(
        taskId,
        cleanName,
        cleanFormula,
        description.trim() || undefined
      );

      toast({
        title: "Virtual Concept Synthesized",
        description: `Successfully synthesized '${cleanName}' into dataset in ${res.node.duration_ms}ms. Dataset now has ${res.total_columns} columns.`,
      });

      // Update local list
      const newItem: ConceptItem = {
        concept_name: cleanName,
        formula_or_intent: cleanFormula,
        description: description.trim(),
        node_id: res.node.id,
        timestamp: res.node.timestamp,
        duration_ms: res.node.duration_ms,
        expression: res.node.parameters?.expression,
      };
      setConcepts((prev) => [newItem, ...prev]);

      // Reset form
      setConceptName("");
      setFormulaOrIntent("");
      setDescription("");

      if (onConceptDerived) {
        onConceptDerived(cleanName);
      }
    } catch (err: any) {
      toast({
        title: "Concept Derivation Failed",
        description: err.message || "Failed to synthesize expression into dataset.",
        variant: "destructive",
      });
    } finally {
      setDeriving(false);
    }
  };

  const handlePresetClick = (presetFormula: string, presetName: string) => {
    setFormulaOrIntent(presetFormula);
    setConceptName(presetName);
  };

  return (
    <Card className="border border-border bg-card shadow-premium rounded-2xl overflow-hidden animate-in fade-in duration-300">
      <CardHeader className="border-b border-border bg-muted/10 pb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="p-2 bg-primary/10 rounded-xl">
              <Boxes className="h-4 w-4 text-primary" />
            </div>
            <div>
              <CardTitle className="text-base font-display font-bold text-foreground flex items-center gap-2">
                <span>Virtual Concept Shelf</span>
                <Badge variant="outline" className="text-[10px] font-mono font-semibold bg-primary/5 text-primary border-primary/20">
                  Data-Formulator Engine
                </Badge>
              </CardTitle>
              <CardDescription className="text-xs">
                Synthesize custom derived metrics and transformations on-the-fly, tracked with full DAG provenance
              </CardDescription>
            </div>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsExpanded(!isExpanded)}
            className="text-xs text-muted-foreground hover:text-foreground h-8 px-2"
          >
            {isExpanded ? "Collapse" : "Expand"}
          </Button>
        </div>
      </CardHeader>

      {isExpanded && (
        <CardContent className="p-4 sm:p-6 space-y-6">
          {/* Concept Creation Form */}
          <form onSubmit={handleDerive} className="space-y-3 bg-muted/15 border border-border/60 rounded-xl p-4">
            <div className="flex items-center justify-between text-xs font-semibold text-foreground">
              <span className="flex items-center gap-1.5 font-display">
                <Plus className="h-3.5 w-3.5 text-primary" />
                Derive New Dataset Metric
              </span>
              <span className="text-[10px] font-mono text-muted-foreground">AST Validated & Sandboxed</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-12 gap-3">
              <div className="sm:col-span-4">
                <label className="block text-[10px] font-mono text-muted-foreground uppercase mb-1">Concept Name</label>
                <Input
                  value={conceptName}
                  onChange={(e) => setConceptName(e.target.value)}
                  placeholder="e.g. net_margin"
                  className="font-mono text-xs bg-white"
                  disabled={deriving}
                />
              </div>
              <div className="sm:col-span-8">
                <label className="block text-[10px] font-mono text-muted-foreground uppercase mb-1">Formula or Intent</label>
                <Input
                  value={formulaOrIntent}
                  onChange={(e) => setFormulaOrIntent(e.target.value)}
                  placeholder="e.g. (revenue - cost) / revenue  OR  pl.col('sales') * 1.15"
                  className="font-mono text-xs bg-white"
                  disabled={deriving}
                />
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-12 gap-3 items-end pt-1">
              <div className="sm:col-span-8">
                <label className="block text-[10px] font-mono text-muted-foreground uppercase mb-1">Business Description (Optional)</label>
                <Input
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="e.g. Operating profit ratio after core supplier deductions"
                  className="text-xs bg-white"
                  disabled={deriving}
                />
              </div>
              <div className="sm:col-span-4 flex justify-end">
                <Button
                  type="submit"
                  disabled={deriving || !conceptName.trim() || !formulaOrIntent.trim()}
                  className="w-full sm:w-auto h-9 text-xs font-semibold flex items-center justify-center gap-2 rounded-xl"
                >
                  {deriving ? (
                    <>
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      <span>Synthesizing...</span>
                    </>
                  ) : (
                    <>
                      <Calculator className="h-3.5 w-3.5" />
                      <span>Derive & Record DAG</span>
                    </>
                  )}
                </Button>
              </div>
            </div>

            {/* Quick helper chips */}
            {columns && columns.length >= 2 && (
              <div className="pt-2 border-t border-border/40 flex items-center gap-2 flex-wrap">
                <span className="text-[10px] font-mono text-muted-foreground flex items-center gap-1">
                  <Calculator className="h-3 w-3 text-primary" />
                  Quick templates:
                </span>
                <button
                  type="button"
                  onClick={() => handlePresetClick(`${columns[0]} / (${columns[1]} + 0.0001)`, `${columns[0]}_to_${columns[1]}_ratio`)}
                  className="text-[10px] font-mono bg-white hover:bg-primary/5 text-foreground border border-border/80 rounded-md px-2 py-0.5 cursor-pointer transition-colors"
                >
                  Ratio: {columns[0]} / {columns[1]}
                </button>
                <button
                  type="button"
                  onClick={() => handlePresetClick(`${columns[0]} - ${columns[1]}`, `${columns[0]}_diff_${columns[1]}`)}
                  className="text-[10px] font-mono bg-white hover:bg-primary/5 text-foreground border border-border/80 rounded-md px-2 py-0.5 cursor-pointer transition-colors"
                >
                  Spread: {columns[0]} - {columns[1]}
                </button>
              </div>
            )}
          </form>

          {/* Derived Concepts Shelf List */}
          <div className="space-y-3">
            <div className="flex items-center justify-between text-xs font-mono font-semibold text-muted-foreground">
              <span className="flex items-center gap-1.5">
                <Layers className="h-3.5 w-3.5 text-primary" />
                Active Derived Metrics ({concepts.length})
              </span>
              <span>Audit Trail Provenance</span>
            </div>

            {loading ? (
              <div className="flex items-center justify-center py-6 text-xs font-mono text-muted-foreground gap-2">
                <Loader2 className="h-4 w-4 animate-spin text-primary" />
                <span>Loading virtual shelf...</span>
              </div>
            ) : concepts.length === 0 ? (
              <div className="p-6 text-center border border-dashed border-border/80 rounded-xl bg-muted/5 space-y-1">
                <p className="text-xs font-sans text-muted-foreground">No virtual concepts derived yet.</p>
                <p className="text-[11px] font-mono text-muted-foreground/70">
                  Derive a calculated column above to dynamically transform your dataset with immutable DAG tracking.
                </p>
              </div>
            ) : (
              <div className="grid gap-2.5 sm:grid-cols-2">
                {concepts.map((item, idx) => (
                  <div
                    key={`${item.concept_name}-${idx}`}
                    className="p-3 bg-white border border-border/80 hover:border-primary/40 rounded-xl shadow-2xs space-y-2 transition-all"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-center gap-1.5">
                        <span className="font-mono text-xs font-bold text-foreground">{item.concept_name}</span>
                        <Badge variant="secondary" className="text-[9px] px-1.5 py-0 rounded font-mono bg-emerald-50 text-emerald-700 border border-emerald-200">
                          Polars Column
                        </Badge>
                      </div>
                      <span className="text-[9px] font-mono text-muted-foreground flex items-center gap-0.5">
                        <Clock className="h-2.5 w-2.5" />
                        {item.duration_ms}ms
                      </span>
                    </div>

                    <div className="bg-muted/20 border border-border/40 rounded-md p-1.5 font-mono text-[11px] text-foreground/90 overflow-x-auto">
                      <code>{item.expression || item.formula_or_intent}</code>
                    </div>

                    {item.description && (
                      <p className="text-[11px] text-muted-foreground font-sans line-clamp-1">
                        {item.description}
                      </p>
                    )}

                    <div className="flex items-center justify-between text-[9px] font-mono text-muted-foreground pt-1 border-t border-border/30">
                      <span>Node: {item.node_id ? item.node_id.slice(0, 8) : "dag-root"}</span>
                      <span className="text-emerald-600 flex items-center gap-0.5">
                        <CheckCircle2 className="h-2.5 w-2.5" />
                        Immutable
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </CardContent>
      )}
    </Card>
  );
};

export default VirtualConceptShelf;
