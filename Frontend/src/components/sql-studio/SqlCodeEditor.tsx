import { useEffect, useRef, useState } from "react";
import type { Compartment, Extension } from "@codemirror/state";
import type { EditorView, ViewUpdate } from "@codemirror/view";
import { Braces } from "lucide-react";
import { Input } from "@/components/ui/input";
import { useTheme } from "@/hooks/useTheme";

interface SqlCodeEditorProps {
  value: string;
  onChange: (value: string) => void;
  tableName: string;
  schemaColumnNames: string[];
  editorViewRef: React.MutableRefObject<EditorView | null>;
  onSelectionChange: (selectedText: string, line: number, col: number) => void;
  onExecute: () => void;
  detectedParams: string[];
  paramValues: Record<string, string>;
  onParamChange: (key: string, val: string) => void;
  minHeight?: string;
}

interface LoadedCodeMirrorModules {
  basicSetup: typeof import("@uiw/react-codemirror").basicSetup;
  autocompletion: typeof import("@codemirror/autocomplete").autocompletion;
  indentWithTab: typeof import("@codemirror/commands").indentWithTab;
  StandardSQL: typeof import("@codemirror/lang-sql").StandardSQL;
  sql: typeof import("@codemirror/lang-sql").sql;
  HighlightStyle: typeof import("@codemirror/language").HighlightStyle;
  syntaxHighlighting: typeof import("@codemirror/language").syntaxHighlighting;
  Annotation: typeof import("@codemirror/state").Annotation;
  Compartment: typeof import("@codemirror/state").Compartment;
  EditorState: typeof import("@codemirror/state").EditorState;
  Prec: typeof import("@codemirror/state").Prec;
  EditorView: typeof import("@codemirror/view").EditorView;
  keymap: typeof import("@codemirror/view").keymap;
  placeholder: typeof import("@codemirror/view").placeholder;
  tags: typeof import("@lezer/highlight").tags;
}

function buildThemeExtension(
  mods: LoadedCodeMirrorModules,
  isDark: boolean,
  minHeight: string,
): Extension {
  const { HighlightStyle, syntaxHighlighting, EditorView: CMView, tags: t } = mods;
  const highlight = HighlightStyle.define(
    isDark
      ? [
          { tag: t.keyword, color: "#34D399", fontWeight: "600" },
          { tag: [t.operatorKeyword, t.typeName], color: "#6EE7B7", fontWeight: "600" },
          { tag: t.string, color: "#FBBF24" },
          { tag: [t.number, t.bool, t.null], color: "#60A5FA" },
          { tag: [t.comment, t.lineComment, t.blockComment], color: "#71717A", fontStyle: "italic" },
          { tag: t.propertyName, color: "#F4F4F5" },
          { tag: t.punctuation, color: "#A1A1AA" },
        ]
      : [
          { tag: t.keyword, color: "#800000", fontWeight: "600" },
          { tag: [t.operatorKeyword, t.typeName], color: "#9A3412", fontWeight: "600" },
          { tag: t.string, color: "#047857" },
          { tag: [t.number, t.bool, t.null], color: "#1D4ED8" },
          { tag: [t.comment, t.lineComment, t.blockComment], color: "#78716C", fontStyle: "italic" },
          { tag: t.propertyName, color: "#1C1917" },
          { tag: t.punctuation, color: "#57534E" },
        ],
  );

  return [
    syntaxHighlighting(highlight),
    CMView.theme(
      {
        "&": {
          minHeight,
          backgroundColor: isDark ? "#0B0D13" : "#FFFFFF",
          color: isDark ? "#E4E4E7" : "#1C1917",
          fontSize: "12.5px",
          fontFamily: "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
        },
        ".cm-content": {
          minHeight,
          padding: "12px 10px",
          lineHeight: "1.6",
          caretColor: isDark ? "#34D399" : "#800000",
        },
        ".cm-gutters": {
          backgroundColor: isDark ? "#0F121B" : "#FAF8F5",
          color: isDark ? "#52525B" : "#A8A29E",
          borderRight: isDark ? "1px solid rgba(255,255,255,0.08)" : "1px solid #E7E5E4",
          minWidth: "2.5rem",
        },
        ".cm-lineNumbers .cm-gutterElement": {
          padding: "0 10px 0 6px",
          fontSize: "11px",
          lineHeight: "1.6",
        },
        ".cm-activeLine": {
          backgroundColor: isDark ? "rgba(255,255,255,0.03)" : "rgba(128,0,0,0.03)",
        },
        ".cm-activeLineGutter": {
          backgroundColor: isDark ? "rgba(255,255,255,0.05)" : "rgba(128,0,0,0.05)",
          color: isDark ? "#E4E4E7" : "#800000",
        },
        "&.cm-focused": {
          outline: "none",
        },
        ".cm-selectionBackground, ::selection": {
          backgroundColor: isDark ? "rgba(52,211,153,0.22) !important" : "rgba(128,0,0,0.14) !important",
        },
        ".cm-tooltip-autocomplete": {
          borderRadius: "8px",
          border: isDark ? "1px solid rgba(255,255,255,0.12)" : "1px solid #D6D3D1",
          backgroundColor: isDark ? "#12151F" : "#FFFFFF",
          boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.25)",
        },
      },
      { dark: isDark },
    ),
  ];
}

export function SqlCodeEditor({
  value,
  onChange,
  tableName,
  schemaColumnNames,
  editorViewRef,
  onSelectionChange,
  onExecute,
  detectedParams,
  paramValues,
  onParamChange,
  minHeight = "148px",
}: SqlCodeEditorProps) {
  const { theme } = useTheme();
  const isDark = theme === "dark";

  const containerRef = useRef<HTMLDivElement | null>(null);
  const modsRef = useRef<LoadedCodeMirrorModules | null>(null);
  const themeCompartmentRef = useRef<Compartment | null>(null);
  const schemaCompartmentRef = useRef<Compartment | null>(null);
  const externalSyncAnnotationRef = useRef<ReturnType<LoadedCodeMirrorModules["Annotation"]["define"]> | null>(null);

  const [editorReady, setEditorReady] = useState(false);

  const callbacksRef = useRef({ onChange, onSelectionChange, onExecute });
  useEffect(() => {
    callbacksRef.current = { onChange, onSelectionChange, onExecute };
  });

  const latestPropsRef = useRef({ value, isDark, minHeight, tableName, schemaColumnNames });
  useEffect(() => {
    latestPropsRef.current = { value, isDark, minHeight, tableName, schemaColumnNames };
  });

  useEffect(() => {
    const parent = containerRef.current;
    if (!parent) return undefined;

    let cancelled = false;
    let mountedView: EditorView | null = null;

    const mountEditor = async () => {
      const [cmReact, cmAuto, cmCmd, cmSql, cmLang, cmState, cmViewMod, cmHighlight] = await Promise.all([
        import("@uiw/react-codemirror"),
        import("@codemirror/autocomplete"),
        import("@codemirror/commands"),
        import("@codemirror/lang-sql"),
        import("@codemirror/language"),
        import("@codemirror/state"),
        import("@codemirror/view"),
        import("@lezer/highlight"),
      ]);

      if (cancelled) return;

      const mods: LoadedCodeMirrorModules = {
        basicSetup: cmReact.basicSetup,
        autocompletion: cmAuto.autocompletion,
        indentWithTab: cmCmd.indentWithTab,
        StandardSQL: cmSql.StandardSQL,
        sql: cmSql.sql,
        HighlightStyle: cmLang.HighlightStyle,
        syntaxHighlighting: cmLang.syntaxHighlighting,
        Annotation: cmState.Annotation,
        Compartment: cmState.Compartment,
        EditorState: cmState.EditorState,
        Prec: cmState.Prec,
        EditorView: cmViewMod.EditorView,
        keymap: cmViewMod.keymap,
        placeholder: cmViewMod.placeholder,
        tags: cmHighlight.tags,
      };
      modsRef.current = mods;

      const themeComp = new mods.Compartment();
      const schemaComp = new mods.Compartment();
      const syncAnnotation = mods.Annotation.define<boolean>();
      themeCompartmentRef.current = themeComp;
      schemaCompartmentRef.current = schemaComp;
      externalSyncAnnotationRef.current = syncAnnotation;

      const handleViewUpdate = (vu: ViewUpdate) => {
        const isExternal = vu.transactions.some((tr) => tr.annotation(syncAnnotation) === true);
        if (vu.docChanged && !isExternal) {
          callbacksRef.current.onChange(vu.state.doc.toString());
        }
        if (vu.selectionSet || vu.docChanged) {
          const sel = vu.state.selection.main;
          const lineInfo = vu.state.doc.lineAt(sel.head);
          const selectedText = sel.empty ? "" : vu.state.sliceDoc(sel.from, sel.to).trim();
          callbacksRef.current.onSelectionChange(selectedText, lineInfo.number, sel.head - lineInfo.from + 1);
        }
      };

      const snap = latestPropsRef.current;
      const state = mods.EditorState.create({
        doc: snap.value,
        extensions: [
          mods.basicSetup({
            tabSize: 2,
            lineNumbers: true,
            highlightActiveLineGutter: true,
            highlightActiveLine: true,
            foldGutter: true,
            bracketMatching: true,
            closeBrackets: true,
            autocompletion: false,
          }),
          mods.autocompletion(),
          mods.EditorState.tabSize.of(2),
          mods.EditorView.lineWrapping,
          mods.EditorView.contentAttributes.of({ "aria-label": "SQL Query Input" }),
          mods.placeholder("SELECT * FROM dataset WHERE ..."),
          mods.keymap.of([mods.indentWithTab]),
          mods.Prec.highest(
            mods.keymap.of([
              {
                key: "Mod-Enter",
                run: () => {
                  callbacksRef.current.onExecute();
                  return true;
                },
              },
            ]),
          ),
          schemaComp.of(
            mods.sql({
              dialect: mods.StandardSQL,
              upperCaseKeywords: true,
              schema: { [snap.tableName]: snap.schemaColumnNames },
              defaultTable: snap.tableName,
            }),
          ),
          themeComp.of(buildThemeExtension(mods, snap.isDark, snap.minHeight)),
          mods.EditorView.updateListener.of(handleViewUpdate),
        ],
      });

      mountedView = new mods.EditorView({ state, parent });
      editorViewRef.current = mountedView;
      setEditorReady(true);
    };

    void mountEditor();

    return () => {
      cancelled = true;
      if (mountedView) {
        mountedView.destroy();
        if (editorViewRef.current === mountedView) {
          editorViewRef.current = null;
        }
      }
      setEditorReady(false);
    };
  }, [editorViewRef]);

  useEffect(() => {
    const view = editorViewRef.current;
    const syncAnnotation = externalSyncAnnotationRef.current;
    if (!editorReady || !view || !syncAnnotation) return;
    const current = view.state.doc.toString();
    if (value !== current) {
      view.dispatch({
        changes: { from: 0, to: current.length, insert: value },
        annotations: syncAnnotation.of(true),
      });
    }
  }, [value, editorReady, editorViewRef]);

  useEffect(() => {
    const view = editorViewRef.current;
    const mods = modsRef.current;
    const comp = themeCompartmentRef.current;
    if (!editorReady || !view || !mods || !comp) return;
    view.dispatch({
      effects: comp.reconfigure(buildThemeExtension(mods, isDark, minHeight)),
    });
  }, [isDark, minHeight, editorReady, editorViewRef]);

  useEffect(() => {
    const view = editorViewRef.current;
    const mods = modsRef.current;
    const comp = schemaCompartmentRef.current;
    if (!editorReady || !view || !mods || !comp) return;
    view.dispatch({
      effects: comp.reconfigure(
        mods.sql({
          dialect: mods.StandardSQL,
          upperCaseKeywords: true,
          schema: { [tableName]: schemaColumnNames },
          defaultTable: tableName,
        }),
      ),
    });
  }, [tableName, schemaColumnNames, editorReady, editorViewRef]);

  return (
    <div className="bg-white dark:bg-[#0B0D13]">
      {detectedParams.length > 0 && (
        <div className="px-3.5 py-2 border-b border-stone-200 dark:border-white/10 bg-[#FAF8F5] dark:bg-[#10131C] flex flex-wrap items-center gap-2.5">
          <span className="inline-flex items-center gap-1 text-[10px] font-mono font-semibold uppercase tracking-wider text-[#800000] dark:text-emerald-400">
            <Braces className="h-3 w-3" />
            Parameters
          </span>
          {detectedParams.map((paramKey) => (
            <div key={paramKey} className="inline-flex items-center gap-1.5">
              <label
                htmlFor={`sql-param-${paramKey}`}
                className="text-[11px] font-mono text-stone-600 dark:text-zinc-400"
              >
                {`{{ ${paramKey} }}`}
              </label>
              <Input
                id={`sql-param-${paramKey}`}
                type="text"
                value={paramValues[paramKey] ?? ""}
                onChange={(e) => onParamChange(paramKey, e.target.value)}
                placeholder="value..."
                className="h-6.5 w-32 px-2 text-[11px] font-mono bg-white dark:bg-[#0B0D13] border-stone-300 dark:border-white/15 rounded-[5px]"
              />
            </div>
          ))}
        </div>
      )}

      <div ref={containerRef} className="min-h-[140px] max-h-[380px] overflow-auto">
        {!editorReady && (
          <pre className="p-3 font-mono text-xs leading-6 text-stone-600 dark:text-zinc-400 whitespace-pre-wrap">
            {value}
          </pre>
        )}
      </div>
    </div>
  );
}
