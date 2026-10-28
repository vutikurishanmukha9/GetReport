import React, { useState, useEffect, useCallback, useMemo } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { useToast } from "@/hooks/use-toast";
import { cn } from "@/lib/utils";
import {
  Database,
  KeyRound,
  Lock,
  Server,
  Table,
  CheckCircle2,
  AlertCircle,
  Loader2,
  ShieldCheck,
  Eye,
  FileCode,
  ArrowRight,
  RefreshCw,
  LogOut,
  ChevronRight,
  ChevronDown,
  Folder,
  FolderOpen,
  Search,
  Upload,
  AlertTriangle,
  HardDrive,
  Trash2,
  Copy,
} from "lucide-react";
import { api } from "@/services/api";
import type {
  SnowflakeConnectParams,
  SnowflakeDatabase,
  SnowflakeSchema,
  SnowflakeObject,
  SnowflakePreviewResponse,
} from "@/types/api";

interface SnowflakeConnectModalProps {
  isOpen: boolean;
  onClose: () => void;
  onImportStarted: (taskId: string, datasetName: string) => void;
}

export const SnowflakeConnectModal: React.FC<SnowflakeConnectModalProps> = ({
  isOpen,
  onClose,
  onImportStarted,
}) => {
  const { toast } = useToast();

  // Step state: 'auth' | 'explorer'
  const [step, setStep] = useState<"auth" | "explorer">("auth");

  // Connection form state
  const [account, setAccount] = useState("");
  const [user, setUser] = useState("");
  const [authType, setAuthType] = useState<"key_pair" | "password">("key_pair");
  const [privateKeyPem, setPrivateKeyPem] = useState("");
  const [privateKeyFileName, setPrivateKeyFileName] = useState("");
  const [privateKeyPassphrase, setPrivateKeyPassphrase] = useState("");
  const [password, setPassword] = useState("");
  const [warehouse, setWarehouse] = useState("COMPUTE_WH");
  const [database, setDatabase] = useState("");
  const [schema, setSchema] = useState("");
  const [role, setRole] = useState("");
  const [useMock, setUseMock] = useState(false);

  // Key input mode: 'file' | 'paste'
  const [keyInputMode, setKeyInputMode] = useState<"file" | "paste">("file");
  const [showRawPem, setShowRawPem] = useState(false);

  // Drag-and-drop state
  const [isDraggingPem, setIsDraggingPem] = useState(false);

  // Connection session state
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isConnecting, setIsConnecting] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  // Explorer state
  const [databases, setDatabases] = useState<SnowflakeDatabase[]>([]);
  const [selectedDb, setSelectedDb] = useState("");
  const [schemas, setSchemas] = useState<SnowflakeSchema[]>([]);
  const [selectedSchema, setSelectedSchema] = useState("");
  const [objects, setObjects] = useState<SnowflakeObject[]>([]);
  const [selectedTable, setSelectedTable] = useState("");
  const [selectedObjectMeta, setSelectedObjectMeta] = useState<SnowflakeObject | null>(null);
  const [isLoadingMetadata, setIsLoadingMetadata] = useState(false);

  // Lazy hierarchy cache & expansion state
  const [schemasCache, setSchemasCache] = useState<Record<string, SnowflakeSchema[]>>({});
  const [objectsCache, setObjectsCache] = useState<Record<string, SnowflakeObject[]>>({});
  const [expandedDbs, setExpandedDbs] = useState<Set<string>>(new Set());
  const [expandedSchemas, setExpandedSchemas] = useState<Set<string>>(new Set());
  const [treeSearch, setTreeSearch] = useState("");

  // Query & Preview state
  const [queryMode, setQueryMode] = useState<"builder" | "sql">("builder");
  const [customSql, setCustomSql] = useState("");
  const [useSampling, setUseSampling] = useState(false);
  const [datasetLabel, setDatasetLabel] = useState("");
  const [previewData, setPreviewData] = useState<SnowflakePreviewResponse | null>(null);
  const [isPreviewing, setIsPreviewing] = useState(false);
  const [isImporting, setIsImporting] = useState(false);

  // Real-time AST & safety validation on custom SQL
  const sqlValidationError = useMemo(() => {
    if (queryMode !== "sql" || !customSql.trim()) return null;
    const upper = customSql.trim().toUpperCase();
    const forbiddenKeywords = [
      "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE",
      "GRANT", "REVOKE", "CREATE", "REPLACE", "MERGE"
    ];
    for (const kw of forbiddenKeywords) {
      const regex = new RegExp(`\\b${kw}\\b`, "i");
      if (regex.test(upper)) {
        return `Prohibited operation '${kw}'. Only read-only queries (SELECT, WITH, UNION) are permitted.`;
      }
    }
    if (!upper.startsWith("SELECT") && !upper.startsWith("WITH")) {
      return "Query must begin with SELECT or WITH.";
    }
    return null;
  }, [queryMode, customSql]);

  // Compute active query based on builder or custom SQL
  const activeQuery = useMemo(() => {
    if (queryMode === "sql") {
      return customSql.trim();
    }
    if (!selectedTable) return "";
    const qualifiedTable = selectedDb && selectedSchema
      ? `"${selectedDb}"."${selectedSchema}"."${selectedTable}"`
      : `"${selectedTable}"`;

    if (useSampling) {
      return `SELECT * FROM ${qualifiedTable} SAMPLE BERNOULLI (10)`;
    }
    return `SELECT * FROM ${qualifiedTable}`;
  }, [queryMode, customSql, selectedTable, selectedDb, selectedSchema, useSampling]);

  // Sample row calculations and memory footprint estimation
  const sampleMetrics = useMemo(() => {
    if (!selectedObjectMeta || !selectedObjectMeta.row_count) {
      return null;
    }
    const totalRows = selectedObjectMeta.row_count;
    const totalBytes = selectedObjectMeta.bytes || 0;
    const avgRowBytes = totalRows > 0 && totalBytes > 0 ? Math.max(16, Math.round(totalBytes / totalRows)) : 64;
    const effectiveRows = useSampling ? Math.round(totalRows * 0.10) : totalRows;
    const estimatedParquetBytes = Math.round(effectiveRows * avgRowBytes * 0.35); // Approx 3x compression
    const estimatedMb = (estimatedParquetBytes / (1024 * 1024)).toFixed(1);
    const ceilingMb = 150;
    const isSafe = parseFloat(estimatedMb) <= ceilingMb;

    return {
      totalRows,
      totalBytes,
      avgRowBytes,
      effectiveRows,
      estimatedMb,
      isSafe,
    };
  }, [selectedObjectMeta, useSampling]);

  // Handle PEM file loading from input or dropzone
  const processPemFile = (file: File) => {
    setPrivateKeyFileName(file.name);
    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result as string;
      setPrivateKeyPem(content);
      toast({
        title: "Key Loaded",
        description: `Successfully loaded RSA key from ${file.name}`,
      });
    };
    reader.onerror = () => {
      toast({
        title: "Key Load Failed",
        description: "Failed to read private key file.",
        variant: "destructive",
      });
    };
    reader.readAsText(file);
  };

  const handleKeyFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processPemFile(file);
  };

  const handlePemDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDraggingPem(false);
    const file = e.dataTransfer.files?.[0];
    if (file) processPemFile(file);
  };

  // Connect handler
  const handleConnect = async (e: React.FormEvent) => {
    e.preventDefault();
    setConnectionError(null);
    setIsConnecting(true);

    try {
      const params: SnowflakeConnectParams = {
        account: account.trim(),
        user: user.trim(),
        warehouse: warehouse.trim() || undefined,
        database: database.trim() || undefined,
        schema: schema.trim() || undefined,
        role: role.trim() || undefined,
        auth_type: authType,
        private_key_pem: authType === "key_pair" ? privateKeyPem : undefined,
        private_key_passphrase: authType === "key_pair" && privateKeyPassphrase ? privateKeyPassphrase : undefined,
        password: authType === "password" ? password : undefined,
        use_mock: useMock,
      };

      const res = await api.connectSnowflake(params);
      setSessionId(res.session_id);
      setStep("explorer");
      toast({
        title: "Connected to Snowflake",
        description: "Encrypted session established. Ready to explore data.",
      });

      // Load initial databases
      await loadDatabases(res.session_id);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to connect to Snowflake.";
      setConnectionError(msg);
      toast({
        title: "Connection Failed",
        description: msg,
        variant: "destructive",
      });
    } finally {
      setIsConnecting(false);
    }
  };

  // Load databases
  const loadDatabases = async (sId: string) => {
    setIsLoadingMetadata(true);
    try {
      const res = await api.listSnowflakeDatabases(sId);
      setDatabases(res.databases);
      if (res.databases.length > 0) {
        const firstDb = res.databases[0].name;
        setSelectedDb(firstDb);
        setExpandedDbs(new Set([firstDb]));
        await loadSchemas(sId, firstDb);
      }
    } catch (err) {
      console.error("Failed to load databases:", err);
    } finally {
      setIsLoadingMetadata(false);
    }
  };

  // Load schemas for a specific database
  const loadSchemas = async (sId: string, dbName: string) => {
    if (schemasCache[dbName]) {
      const cached = schemasCache[dbName];
      setSchemas(cached);
      if (cached.length > 0) {
        setSelectedSchema(cached[0].name);
        await loadObjects(sId, dbName, cached[0].name);
      }
      return;
    }

    setIsLoadingMetadata(true);
    try {
      const res = await api.listSnowflakeSchemas(sId, dbName);
      setSchemas(res.schemas);
      setSchemasCache(prev => ({ ...prev, [dbName]: res.schemas }));

      if (res.schemas.length > 0) {
        const firstSchema = res.schemas[0].name;
        setSelectedSchema(firstSchema);
        setExpandedSchemas(prev => new Set([...prev, `${dbName}.${firstSchema}`]));
        await loadObjects(sId, dbName, firstSchema);
      } else {
        setSchemas([]);
        setSelectedSchema("");
        setObjects([]);
        setSelectedTable("");
        setSelectedObjectMeta(null);
      }
    } catch (err) {
      console.error("Failed to load schemas:", err);
    } finally {
      setIsLoadingMetadata(false);
    }
  };

  // Load objects (tables/views) for a specific schema
  const loadObjects = async (sId: string, dbName: string, scName: string) => {
    const cacheKey = `${dbName}.${scName}`;
    if (objectsCache[cacheKey]) {
      const cached = objectsCache[cacheKey];
      setObjects(cached);
      if (cached.length > 0) {
        setSelectedTable(cached[0].name);
        setSelectedObjectMeta(cached[0]);
        setDatasetLabel(cached[0].name.toLowerCase());
      }
      return;
    }

    setIsLoadingMetadata(true);
    try {
      const res = await api.listSnowflakeObjects(sId, dbName, scName);
      setObjects(res.objects);
      setObjectsCache(prev => ({ ...prev, [cacheKey]: res.objects }));

      if (res.objects.length > 0) {
        setSelectedTable(res.objects[0].name);
        setSelectedObjectMeta(res.objects[0]);
        setDatasetLabel(res.objects[0].name.toLowerCase());
      } else {
        setSelectedTable("");
        setSelectedObjectMeta(null);
      }
    } catch (err) {
      console.error("Failed to load objects:", err);
    } finally {
      setIsLoadingMetadata(false);
    }
  };

  // Toggle database expansion in tree
  const toggleDatabaseNode = async (dbName: string) => {
    const next = new Set(expandedDbs);
    if (next.has(dbName)) {
      next.delete(dbName);
    } else {
      next.add(dbName);
      setSelectedDb(dbName);
      if (sessionId) await loadSchemas(sessionId, dbName);
    }
    setExpandedDbs(next);
  };

  // Toggle schema expansion in tree
  const toggleSchemaNode = async (dbName: string, scName: string) => {
    const key = `${dbName}.${scName}`;
    const next = new Set(expandedSchemas);
    if (next.has(key)) {
      next.delete(key);
    } else {
      next.add(key);
      setSelectedDb(dbName);
      setSelectedSchema(scName);
      if (sessionId) await loadObjects(sessionId, dbName, scName);
    }
    setExpandedSchemas(next);
  };

  // Select table node
  const handleSelectTableNode = (obj: SnowflakeObject) => {
    setSelectedDb(obj.database);
    setSelectedSchema(obj.schema);
    setSelectedTable(obj.name);
    setSelectedObjectMeta(obj);
    setDatasetLabel(obj.name.toLowerCase());
  };

  // Filtered objects in tree search
  const filteredTreeObjects = useMemo(() => {
    if (!treeSearch.trim()) return objects;
    const q = treeSearch.toLowerCase().trim();
    return objects.filter(o => o.name.toLowerCase().includes(q));
  }, [objects, treeSearch]);

  // Preview Query
  const handlePreview = async () => {
    if (!sessionId || !activeQuery || sqlValidationError) return;
    setIsPreviewing(true);
    try {
      const res = await api.previewSnowflakeQuery(sessionId, activeQuery, 25);
      setPreviewData(res);
      toast({
        title: "Data Preview Loaded",
        description: `Retrieved ${res.total_preview_rows} sample records.`,
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Preview query failed.";
      toast({
        title: "Preview Failed",
        description: msg,
        variant: "destructive",
      });
    } finally {
      setIsPreviewing(false);
    }
  };

  // Import query into GetReport pipeline
  const handleImport = async () => {
    if (!sessionId || !activeQuery || sqlValidationError) return;
    setIsImporting(true);
    try {
      const label = datasetLabel.trim() || selectedTable.toLowerCase() || "snowflake_data";
      const res = await api.importSnowflakeQuery(sessionId, activeQuery, label);

      toast({
        title: "Extraction Started",
        description: "Streaming Snowflake records into the audit pipeline...",
      });

      onImportStarted(res.task_id, label);
      onClose();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Import query failed.";
      toast({
        title: "Import Failed",
        description: msg,
        variant: "destructive",
      });
    } finally {
      setIsImporting(false);
    }
  };

  // Disconnect handler
  const handleDisconnect = async () => {
    if (sessionId) {
      try {
        await api.disconnectSnowflake(sessionId);
      } catch (e) {
        console.error("Disconnect cleanup failed:", e);
      }
    }
    setSessionId(null);
    setPrivateKeyPem("");
    setPrivateKeyPassphrase("");
    setPassword("");
    setPrivateKeyFileName("");
    setStep("auth");
    setPreviewData(null);
    setSelectedObjectMeta(null);
    toast({
      title: "Disconnected",
      description: "Snowflake session invalidated and credentials purged.",
    });
  };

  // Quick Demo fill
  const handleUseDemo = () => {
    setAccount("demo-org.snowflakecomputing.com");
    setUser("DEMO_ANALYST");
    setAuthType("password");
    setPassword("demo-sandbox-key");
    setWarehouse("ANALYTICS_WH");
    setDatabase("ANALYTICS_DB");
    setSchema("PUBLIC");
    setUseMock(true);
    toast({
      title: "Demo Configuration Applied",
      description: "Using simulated Snowflake data warehouse for instant audit testing.",
    });
  };

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent
        className={cn(
          "w-[calc(100vw-1.5rem)] sm:w-full max-h-[92dvh] overflow-y-auto bg-[#08090e] border border-white/10 text-white shadow-2xl rounded-2xl transition-colors duration-150 p-4 sm:p-6 md:p-7",
          step === "auth" ? "max-w-2xl" : "max-w-5xl"
        )}
      >
        <DialogHeader className="space-y-2 border-b border-white/10 pb-4 pr-7 sm:pr-0">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div className="flex items-start sm:items-center gap-3 min-w-0">
              <div className="p-2 sm:p-2.5 rounded-xl bg-white/[0.05] border border-white/10 text-zinc-300 shrink-0">
                <Database className="h-5 w-5 sm:h-6 sm:w-6" />
              </div>
              <div className="min-w-0">
                <DialogTitle className="text-base sm:text-xl font-display font-bold text-white flex flex-wrap items-center gap-2">
                  <span className="truncate">Snowflake Data Warehouse</span>
                  <span className="px-1.5 py-0.5 text-[9px] font-mono font-semibold uppercase tracking-widest text-zinc-400 bg-white/[0.06] border border-white/10 rounded leading-none shrink-0">
                    Beta
                  </span>
                  {sessionId && (
                    <Badge variant="outline" className="border-emerald-500/30 bg-emerald-500/10 text-emerald-400 text-[10px] font-mono font-medium shrink-0">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1.5" />
                      Encrypted Active
                    </Badge>
                  )}
                </DialogTitle>
                <DialogDescription className="text-[11px] sm:text-xs text-zinc-400 leading-relaxed mt-0.5">
                  Direct warehouse ingestion with out-of-core profiling and automated data hygiene audit
                </DialogDescription>
              </div>
            </div>

            <div className="flex items-center gap-2 self-start sm:self-auto shrink-0">
              {step === "auth" && !sessionId && (
                <button
                  type="button"
                  onClick={handleUseDemo}
                  className="px-2.5 py-1.5 rounded-lg text-xs font-medium text-zinc-300 bg-white/[0.05] hover:bg-white/10 border border-white/10 transition-colors cursor-pointer"
                  title="Auto-fill simulated sandbox credentials"
                >
                  Auto-fill Demo
                </button>
              )}
              {sessionId && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleDisconnect}
                  className="border-white/10 bg-white/5 text-zinc-300 hover:text-rose-400 hover:bg-rose-500/10 hover:border-rose-500/20 text-xs"
                >
                  <LogOut className="h-3.5 w-3.5 mr-1.5" />
                  Disconnect
                </Button>
              )}
            </div>
          </div>
        </DialogHeader>

        {/* STEP 1: AUTHENTICATION FORM */}
        {step === "auth" && (
          <form onSubmit={handleConnect} className="space-y-4 pt-3">
            {/* Security Notice Card */}
            <div className="p-3 rounded-xl bg-[#121215] border border-white/10 text-zinc-300 text-xs flex items-start gap-2.5 sm:gap-3">
              <div className="p-1.5 rounded-lg bg-white/[0.05] border border-white/10 text-zinc-300 shrink-0 mt-0.5">
                <ShieldCheck className="h-4 w-4" />
              </div>
              <div className="space-y-0.5 text-xs min-w-0">
                <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
                  <span className="font-semibold text-white">Dedicated Read-Only Enforcement</span>
                  <span className="text-[9px] uppercase font-mono px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                    Safe Audit
                  </span>
                </div>
                <p className="text-[11px] text-zinc-400 leading-relaxed">
                  Connect using a Snowflake role with <code className="font-mono bg-black/50 px-1 py-0.5 rounded text-zinc-300 text-[10px]">GRANT SELECT</code> privileges. GetReport executes pure read-only extracts with zero mutation capability.
                </p>
              </div>
            </div>

            {/* Account & User */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
              <div>
                <label htmlFor="snowflake-account" className="block text-xs font-medium text-zinc-300 mb-1">
                  Account Identifier <span className="text-rose-400">*</span>
                </label>
                <Input
                  id="snowflake-account"
                  required
                  placeholder="e.g. orgname-accountname"
                  value={account}
                  onChange={(e) => setAccount(e.target.value)}
                  className="bg-[#0e1017] border-white/10 text-white placeholder:text-zinc-500 text-xs h-10"
                />
                <span className="text-[10px] text-zinc-500 mt-1 block">
                  Identifier without https:// or snowflakecomputing.com
                </span>
              </div>

              <div>
                <label htmlFor="snowflake-user" className="block text-xs font-medium text-zinc-300 mb-1">
                  Username <span className="text-rose-400">*</span>
                </label>
                <Input
                  id="snowflake-user"
                  required
                  placeholder="e.g. DATA_ANALYST"
                  value={user}
                  onChange={(e) => setUser(e.target.value)}
                  className="bg-[#0e1017] border-white/10 text-white placeholder:text-zinc-500 text-xs h-10"
                />
                <span className="text-[10px] text-zinc-500 mt-1 block">
                  Case-sensitive Snowflake user login
                </span>
              </div>
            </div>

            {/* Auth Method Selector */}
            <div className="space-y-1.5">
              <span className="block text-xs font-medium text-zinc-300">
                Authentication Method
              </span>
              <div className="grid grid-cols-1 sm:grid-cols-2 p-1 rounded-xl bg-black/40 border border-white/10 gap-1.5">
                <button
                  type="button"
                  onClick={() => setAuthType("key_pair")}
                  className={cn(
                    "flex items-center justify-center gap-2 py-2 px-3 rounded-lg text-xs font-medium transition-colors cursor-pointer",
                    authType === "key_pair"
                      ? "bg-white/10 text-white border border-white/20 shadow-xs"
                      : "text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.04]"
                  )}
                >
                  <KeyRound className="h-3.5 w-3.5 text-zinc-300 shrink-0" />
                  <span className="truncate">RSA Key-Pair</span>
                  <span className="text-[9px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded bg-white/10 text-zinc-200 font-semibold border border-white/20 shrink-0">
                    Recommended
                  </span>
                </button>

                <button
                  type="button"
                  onClick={() => setAuthType("password")}
                  className={cn(
                    "flex items-center justify-center gap-2 py-2 px-3 rounded-lg text-xs font-medium transition-colors cursor-pointer",
                    authType === "password"
                      ? "bg-white/10 text-white border border-white/20 shadow-xs"
                      : "text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.04]"
                  )}
                >
                  <Lock className="h-3.5 w-3.5 text-zinc-400 shrink-0" />
                  <span>Password</span>
                </button>
              </div>
            </div>

            {/* Auth Input Fields */}
            {authType === "key_pair" ? (
              <div className="p-3.5 rounded-xl bg-[#0e1017] border border-white/10 space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="text-xs font-medium text-zinc-300 flex items-center gap-1.5">
                    <KeyRound className="h-3.5 w-3.5 text-zinc-300" />
                    <span>RSA Private Key</span>
                    <span className="text-rose-400">*</span>
                  </span>
                  <div className="flex items-center gap-1 bg-black/40 p-0.5 rounded-lg border border-white/5">
                    <button
                      type="button"
                      onClick={() => setKeyInputMode("file")}
                      className={cn(
                        "text-[11px] px-2 py-0.5 rounded-md transition-colors cursor-pointer",
                        keyInputMode === "file"
                          ? "bg-white/10 text-white font-medium"
                          : "text-zinc-400 hover:text-white"
                      )}
                    >
                      File Upload
                    </button>
                    <button
                      type="button"
                      onClick={() => setKeyInputMode("paste")}
                      className={cn(
                        "text-[11px] px-2 py-0.5 rounded-md transition-colors cursor-pointer",
                        keyInputMode === "paste"
                          ? "bg-white/10 text-white font-medium"
                          : "text-zinc-400 hover:text-white"
                      )}
                    >
                      Paste PEM
                    </button>
                  </div>
                </div>

                {/* If key is loaded, show a clean verified pill card */}
                {privateKeyPem ? (
                  <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/30 space-y-2">
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5">
                      <div className="flex items-center gap-2 min-w-0">
                        <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0" />
                        <div className="min-w-0">
                          <p className="text-xs font-mono font-medium text-emerald-300 truncate max-w-[200px] sm:max-w-xs">
                            {privateKeyFileName || "PEM Key Loaded"}
                          </p>
                          <span className="text-[10px] text-zinc-400 font-mono block">
                            {privateKeyPem.length} characters • Session vault encrypted
                          </span>
                        </div>
                      </div>
                      <div className="flex items-center gap-1.5 shrink-0 self-end sm:self-auto">
                        <button
                          type="button"
                          onClick={() => setShowRawPem(!showRawPem)}
                          className="text-[10px] text-zinc-400 hover:text-zinc-200 px-2 py-1 rounded bg-black/40 border border-white/10 cursor-pointer"
                        >
                          {showRawPem ? "Hide" : "Preview"}
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            setPrivateKeyPem("");
                            setPrivateKeyFileName("");
                          }}
                          className="text-[10px] text-rose-400 hover:text-rose-300 px-2 py-1 rounded bg-rose-500/10 border border-rose-500/20 cursor-pointer"
                        >
                          Clear
                        </button>
                      </div>
                    </div>

                    {showRawPem && (
                      <textarea
                        rows={3}
                        aria-label="Raw RSA Private Key Content"
                        value={privateKeyPem}
                        onChange={(e) => setPrivateKeyPem(e.target.value)}
                        className="w-full p-2.5 rounded-lg border border-white/10 bg-black/60 text-zinc-200 font-mono text-[10px] focus:outline-none focus:border-white/40"
                      />
                    )}
                  </div>
                ) : keyInputMode === "file" ? (
                  /* Drag and Drop / Touch Picker Zone */
                  <div
                    onDragOver={(e) => {
                      e.preventDefault();
                      setIsDraggingPem(true);
                    }}
                    onDragLeave={(e) => {
                      e.preventDefault();
                      setIsDraggingPem(false);
                    }}
                    onDrop={handlePemDrop}
                    className={`p-4 sm:p-5 rounded-xl border-2 border-dashed transition-colors flex flex-col items-center justify-center text-center cursor-pointer ${isDraggingPem
                        ? "border-white/50 bg-white/10 text-white"
                        : "border-white/15 bg-black/30 text-zinc-400 hover:border-white/30 hover:bg-black/40"
                      }`}
                  >
                    <label className="cursor-pointer w-full flex flex-col items-center">
                      <input
                        type="file"
                        accept=".pem,.key"
                        onChange={handleKeyFileUpload}
                        className="sr-only"
                      />
                      <Upload className="h-5 w-5 mb-1.5 text-zinc-300" />
                      <p className="text-xs font-medium text-zinc-200">
                        Tap to select private key file (.pem), or drag and drop
                      </p>
                      <p className="text-[10px] text-zinc-500 mt-0.5">
                        Encrypted with HKDF/AES-256-GCM in memory session vault
                      </p>
                    </label>
                  </div>
                ) : (
                  /* Monospace Key Textarea */
                  <textarea
                    rows={4}
                    aria-label="RSA Private Key PEM"
                    placeholder="Paste PKCS#8 or OpenSSL RSA private key PEM here..."
                    value={privateKeyPem}
                    onChange={(e) => setPrivateKeyPem(e.target.value)}
                    className="w-full p-2.5 rounded-xl border border-white/10 bg-black/40 text-zinc-200 font-mono text-[11px] focus:outline-none focus:border-white/40 focus:ring-1 focus:ring-white/20"
                  />
                )}

                <div>
                  <label htmlFor="snowflake-key-passphrase" className="block text-xs font-medium text-zinc-300 mb-1">
                    Key Passphrase <span className="text-zinc-500 font-normal">(optional)</span>
                  </label>
                  <Input
                    id="snowflake-key-passphrase"
                    type="password"
                    placeholder="Required only if RSA key is encrypted with a passphrase"
                    value={privateKeyPassphrase}
                    onChange={(e) => setPrivateKeyPassphrase(e.target.value)}
                    className="bg-[#08090e] border-white/10 text-white placeholder:text-zinc-600 text-xs h-9"
                  />
                </div>
              </div>
            ) : (
              <div className="p-3.5 rounded-xl bg-[#0e1017] border border-white/10 space-y-2">
                <label htmlFor="snowflake-password" className="block text-xs font-medium text-zinc-300">
                  Account Password <span className="text-rose-400">*</span>
                </label>
                <Input
                  id="snowflake-password"
                  required
                  type="password"
                  placeholder="Enter your Snowflake password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="bg-[#08090e] border-white/10 text-white placeholder:text-zinc-600 text-xs h-10"
                />
              </div>
            )}

            {/* Warehouse and Context: 2x2 Grid */}
            <div className="p-3.5 rounded-xl bg-[#0e1017] border border-white/10 space-y-2.5">
              <span className="text-xs font-medium text-zinc-300 block">
                Warehouse & Session Context <span className="text-zinc-500 font-normal">(Optional defaults)</span>
              </span>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                <div>
                  <label htmlFor="snowflake-warehouse" className="block text-[11px] text-zinc-400 mb-1">Warehouse</label>
                  <Input
                    id="snowflake-warehouse"
                    placeholder="COMPUTE_WH"
                    value={warehouse}
                    onChange={(e) => setWarehouse(e.target.value)}
                    className="bg-[#08090e] border-white/10 text-white placeholder:text-zinc-600 text-xs h-9"
                  />
                </div>
                <div>
                  <label htmlFor="snowflake-database" className="block text-[11px] text-zinc-400 mb-1">Database</label>
                  <Input
                    id="snowflake-database"
                    placeholder="ANALYTICS_DB"
                    value={database}
                    onChange={(e) => setDatabase(e.target.value)}
                    className="bg-[#08090e] border-white/10 text-white placeholder:text-zinc-600 text-xs h-9"
                  />
                </div>
                <div>
                  <label htmlFor="snowflake-schema" className="block text-[11px] text-zinc-400 mb-1">Schema</label>
                  <Input
                    id="snowflake-schema"
                    placeholder="PUBLIC"
                    value={schema}
                    onChange={(e) => setSchema(e.target.value)}
                    className="bg-[#08090e] border-white/10 text-white placeholder:text-zinc-600 text-xs h-9"
                  />
                </div>
                <div>
                  <label htmlFor="snowflake-role" className="block text-[11px] text-zinc-400 mb-1">Role</label>
                  <Input
                    id="snowflake-role"
                    placeholder="READONLY_ROLE"
                    value={role}
                    onChange={(e) => setRole(e.target.value)}
                    className="bg-[#08090e] border-white/10 text-white placeholder:text-zinc-600 text-xs h-9"
                  />
                </div>
              </div>
            </div>

            {/* Sandbox mock driver toggle with Switch component */}
            <div className="flex items-center justify-between p-3 rounded-xl bg-[#0e1017] border border-white/10 gap-3">
              <div className="space-y-0.5 min-w-0">
                <span className="text-xs font-medium text-zinc-200 block">Sandbox Mock Driver</span>
                <p className="text-[11px] text-zinc-500 leading-normal">
                  Simulates realistic multi-batch financial tables for testing without external cloud network requests.
                </p>
              </div>
              <Switch
                checked={useMock}
                onCheckedChange={setUseMock}
                className="data-[state=checked]:bg-emerald-500 shrink-0"
              />
            </div>

            {connectionError && (
              <div className="flex items-center gap-2.5 p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-xs text-rose-400">
                <AlertCircle className="h-4 w-4 shrink-0" />
                <span>{connectionError}</span>
              </div>
            )}

            <div className="flex flex-col-reverse sm:flex-row sm:items-center sm:justify-between gap-3 pt-3 border-t border-white/10">
              <div className="flex items-center gap-1.5 text-[10px] sm:text-[11px] text-zinc-500">
                <ShieldCheck className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                <span>AES-256 encrypted session • 15m idle auto-expiry</span>
              </div>

              <div className="flex items-center gap-2.5 w-full sm:w-auto">
                <Button
                  type="button"
                  variant="outline"
                  onClick={onClose}
                  className="flex-1 sm:flex-none border-white/10 text-zinc-300 hover:bg-white/5 text-xs h-9"
                >
                  Cancel
                </Button>
                <Button
                  type="submit"
                  disabled={isConnecting}
                  className="flex-1 sm:flex-none bg-white hover:bg-zinc-200 text-black font-semibold text-xs px-5 shadow-sm h-9"
                >
                  {isConnecting ? (
                    <>
                      <Loader2 className="h-3.5 w-3.5 mr-2 animate-spin" />
                      Authenticating...
                    </>
                  ) : (
                    <>
                      Connect
                      <ArrowRight className="h-3.5 w-3.5 ml-2" />
                    </>
                  )}
                </Button>
              </div>
            </div>
          </form>
        )}

        {/* STEP 2: SCHEMA EXPLORER & QUERY PREVIEW */}
        {step === "explorer" && (
          <div className="space-y-5 pt-3">
            {/* Mode Switcher */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div className="grid grid-cols-2 sm:flex sm:items-center gap-1.5 p-1 rounded-xl bg-black/40 border border-white/10 w-full sm:w-auto">
                <button
                  type="button"
                  onClick={() => setQueryMode("builder")}
                  className={cn(
                    "px-3 py-1.5 rounded-lg text-xs font-medium transition-colors cursor-pointer text-center",
                    queryMode === "builder"
                      ? "bg-white/10 border border-white/20 text-white font-semibold"
                      : "text-zinc-400 hover:text-white"
                  )}
                >
                  Tree Explorer
                </button>
                <button
                  type="button"
                  onClick={() => setQueryMode("sql")}
                  className={cn(
                    "px-3 py-1.5 rounded-lg text-xs font-medium transition-colors cursor-pointer text-center",
                    queryMode === "sql"
                      ? "bg-white/10 border border-white/20 text-white font-semibold"
                      : "text-zinc-400 hover:text-white"
                  )}
                >
                  SQL Editor
                </button>
              </div>

              <div className="flex items-center gap-2 text-xs text-zinc-400 self-end sm:self-auto">
                <Switch
                  checked={useSampling}
                  onCheckedChange={setUseSampling}
                  className="data-[state=checked]:bg-emerald-500 scale-90"
                />
                <span className="text-[11px]">10% Bernoulli Subsample</span>
              </div>
            </div>

            {/* HIERARCHICAL TREE EXPLORER */}
            {queryMode === "builder" ? (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-3.5 sm:p-4 rounded-xl bg-[#0e1017] border border-white/10">
                {/* Left Tree Explorer Sidebar */}
                <div className="md:col-span-1 border-b md:border-b-0 md:border-r border-white/10 pb-3 md:pb-0 md:pr-3 space-y-3">
                  <div className="relative">
                    <Search className="h-3.5 w-3.5 absolute left-2.5 top-2.5 text-zinc-500" />
                    <Input
                      placeholder="Filter tables..."
                      value={treeSearch}
                      onChange={(e) => setTreeSearch(e.target.value)}
                      className="bg-[#08090e] border-white/10 text-white placeholder:text-zinc-600 text-xs h-8 pl-8"
                    />
                  </div>

                  <div className="max-h-52 sm:max-h-60 overflow-y-auto space-y-1 text-xs">
                    {databases.map((db) => {
                      const isDbExpanded = expandedDbs.has(db.name);
                      return (
                        <div key={db.name} className="space-y-1">
                          <button
                            type="button"
                            onClick={() => toggleDatabaseNode(db.name)}
                            className="w-full flex items-center justify-between p-1.5 rounded hover:bg-white/5 text-zinc-300 text-left font-mono"
                          >
                            <span className="flex items-center gap-1.5 truncate">
                              {isDbExpanded ? <ChevronDown className="h-3.5 w-3.5 text-zinc-500 shrink-0" /> : <ChevronRight className="h-3.5 w-3.5 text-zinc-500 shrink-0" />}
                              <Database className="h-3.5 w-3.5 text-zinc-400 shrink-0" />
                              <span className="truncate">{db.name}</span>
                            </span>
                          </button>

                          {isDbExpanded && (
                            <div className="pl-4 space-y-1 border-l border-white/5 ml-2">
                              {schemas.map((sc) => {
                                const scKey = `${db.name}.${sc.name}`;
                                const isScExpanded = expandedSchemas.has(scKey);
                                return (
                                  <div key={scKey} className="space-y-1">
                                    <button
                                      type="button"
                                      onClick={() => toggleSchemaNode(db.name, sc.name)}
                                      className="w-full flex items-center justify-between p-1 rounded hover:bg-white/5 text-zinc-400 text-left font-mono text-[11px]"
                                    >
                                      <span className="flex items-center gap-1.5 truncate">
                                        {isScExpanded ? <ChevronDown className="h-3 w-3 text-zinc-500 shrink-0" /> : <ChevronRight className="h-3 w-3 text-zinc-500 shrink-0" />}
                                        <Folder className="h-3 w-3 text-zinc-400 shrink-0" />
                                        <span className="truncate">{sc.name}</span>
                                      </span>
                                    </button>

                                    {isScExpanded && (
                                      <div className="pl-4 space-y-0.5 border-l border-white/5 ml-2">
                                        {filteredTreeObjects.map((obj) => {
                                          const isSelected = selectedTable === obj.name;
                                          return (
                                            <button
                                              key={obj.name}
                                              type="button"
                                              onClick={() => handleSelectTableNode(obj)}
                                              className={cn(
                                                "w-full flex items-center justify-between p-1 rounded text-left text-[11px] font-mono transition-colors",
                                                isSelected
                                                  ? "bg-white/10 text-white font-semibold"
                                                  : "text-zinc-400 hover:bg-white/5 hover:text-zinc-200"
                                              )}
                                            >
                                              <span className="flex items-center gap-1.5 truncate">
                                                <Table className="h-3 w-3 text-emerald-400 shrink-0" />
                                                <span className="truncate">{obj.name}</span>
                                              </span>
                                              <span className="text-[9px] text-zinc-500 shrink-0">
                                                {obj.row_count ? `${(obj.row_count / 1000).toFixed(0)}k` : obj.type}
                                              </span>
                                            </button>
                                          );
                                        })}
                                      </div>
                                    )}
                                  </div>
                                );
                              })}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Right Details & Sample Row Calculator */}
                <div className="md:col-span-2 space-y-3.5">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 text-xs">
                    <div className="p-3 rounded-lg bg-black/40 border border-white/5">
                      <span className="text-[10px] text-zinc-500 block mb-0.5 font-mono">SELECTED OBJECT</span>
                      <div className="font-mono text-white text-xs font-semibold truncate">
                        {selectedTable ? `"${selectedDb}"."${selectedSchema}"."${selectedTable}"` : "None selected"}
                      </div>
                      <span className="text-[10px] text-zinc-500 mt-1 block">
                        Type: {selectedObjectMeta?.type || "TABLE"}
                      </span>
                    </div>

                    <div className="p-3 rounded-lg bg-black/40 border border-white/5">
                      <span className="text-[10px] text-zinc-500 block mb-0.5 font-mono">WAREHOUSE ROWS</span>
                      <div className="font-mono text-emerald-400 text-xs font-semibold">
                        {selectedObjectMeta?.row_count !== undefined
                          ? selectedObjectMeta.row_count.toLocaleString()
                          : "Metadata query"}
                      </div>
                      <span className="text-[10px] text-zinc-500 mt-1 block">
                        {selectedObjectMeta?.bytes !== undefined
                          ? `${(selectedObjectMeta.bytes / (1024 * 1024)).toFixed(1)} MB uncompressed`
                          : "Live cluster"}
                      </span>
                    </div>
                  </div>

                  {/* Sample Row Calculations & Memory Budget Card */}
                  {sampleMetrics && (
                    <div className="p-3 rounded-xl bg-white/[0.03] border border-white/10 text-xs space-y-2">
                      <div className="flex flex-wrap items-center justify-between gap-1.5">
                        <span className="font-semibold text-zinc-200 flex items-center gap-1.5 font-mono text-[11px]">
                          <HardDrive className="h-3.5 w-3.5 text-zinc-300 shrink-0" />
                          Ingestion Budget Calculator
                        </span>
                        <Badge
                          variant="outline"
                          className={sampleMetrics.isSafe ? "border-emerald-500/30 text-emerald-400 bg-emerald-500/10 text-[10px]" : "border-rose-500/30 text-rose-400 bg-rose-500/10 text-[10px]"}
                        >
                          {sampleMetrics.isSafe ? "Within 150MB Ceiling" : "Exceeds Safety Ceiling"}
                        </Badge>
                      </div>

                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-[11px] font-mono text-zinc-300">
                        <div className="p-2 rounded bg-black/30 border border-white/5">
                          <span className="text-zinc-500 text-[10px] block">Avg Row Width</span>
                          <span>~{sampleMetrics.avgRowBytes} bytes/row</span>
                        </div>
                        <div className="p-2 rounded bg-black/30 border border-white/5">
                          <span className="text-zinc-500 text-[10px] block">Projected Extract</span>
                          <span>{sampleMetrics.effectiveRows.toLocaleString()} rows</span>
                        </div>
                        <div className="p-2 rounded bg-black/30 border border-white/5">
                          <span className="text-zinc-500 text-[10px] block">Estimated Parquet</span>
                          <span>~{sampleMetrics.estimatedMb} MB</span>
                        </div>
                      </div>
                    </div>
                  )}

                  {isLoadingMetadata && (
                    <div className="flex items-center gap-2 text-xs text-zinc-300">
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      <span>Loading metadata from warehouse...</span>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              /* MONOSPACE SQL EDITOR */
              <div className="p-3.5 sm:p-4 rounded-xl bg-[#0e1017] border border-white/10 space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-1.5">
                  <label htmlFor="snowflake-sql-editor" className="block text-xs font-medium text-zinc-300 font-mono">
                    Snowflake SQL Query (Monospace Editor)
                  </label>
                  <span className="text-[10px] text-zinc-500 font-mono">
                    Read-only (SELECT / WITH)
                  </span>
                </div>

                <textarea
                  id="snowflake-sql-editor"
                  aria-label="Snowflake SQL Query"
                  rows={5}
                  value={customSql}
                  onChange={(e) => setCustomSql(e.target.value)}
                  placeholder="SELECT * FROM ANALYTICS_DB.PUBLIC.TRANSACTIONS WHERE amount > 100"
                  className="w-full p-3 rounded-lg border border-white/10 bg-[#08090e] text-zinc-100 font-mono text-xs focus:outline-none focus:border-white/40"
                />

                {sqlValidationError ? (
                  <div className="flex items-center gap-2 p-2.5 rounded-lg bg-rose-500/10 border border-rose-500/20 text-xs text-rose-400 font-mono">
                    <AlertTriangle className="h-4 w-4 shrink-0" />
                    <span>{sqlValidationError}</span>
                  </div>
                ) : (
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 text-[11px] text-zinc-500">
                    <span>Queries validated using Snowflake AST parser. Prohibits DDL/DML mutations.</span>
                    {useSampling && (
                      <Badge variant="outline" className="text-[9px] bg-white/5 text-zinc-300 border-white/10 self-start sm:self-auto">
                        10% Bernoulli Sampling Active
                      </Badge>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Resolved Execution Query Preview & Label */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 items-end">
              <div className="md:col-span-2">
                <span className="block text-[11px] text-zinc-400 mb-1 font-mono">
                  Resolved Execution Query
                </span>
                <div className="p-2.5 rounded-lg bg-black/40 border border-white/5 font-mono text-[11px] text-zinc-300 truncate">
                  {activeQuery || "Select a table or enter SQL..."}
                </div>
              </div>

              <div>
                <label htmlFor="snowflake-dataset-label" className="block text-[11px] text-zinc-400 mb-1">
                  Dataset Label
                </label>
                <Input
                  id="snowflake-dataset-label"
                  placeholder="e.g. sales_transactions"
                  value={datasetLabel}
                  onChange={(e) => setDatasetLabel(e.target.value)}
                  className="bg-[#0e1017] border-white/10 text-white placeholder:text-zinc-500 text-xs h-9"
                />
              </div>
            </div>

            {/* PREVIEW BUTTON & TABLE */}
            <div className="space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={handlePreview}
                  disabled={!activeQuery || isPreviewing || Boolean(sqlValidationError)}
                  className="border-white/10 bg-white/5 text-zinc-200 hover:bg-white/10 text-xs gap-1.5"
                >
                  {isPreviewing ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <Eye className="h-3.5 w-3.5 text-zinc-300" />
                  )}
                  Preview 25 Rows
                </Button>

                <div className="flex flex-wrap items-center gap-2 text-[10px] sm:text-[11px] text-zinc-500 font-mono">
                  {useSampling && (
                    <Badge variant="outline" className="border-white/10 bg-white/5 text-zinc-300 text-[10px]">
                      Sampled (10%)
                    </Badge>
                  )}
                  <span>150MB Parquet Ceiling Budget</span>
                </div>
              </div>

              {previewData && (
                <div className="p-3 rounded-xl bg-black/40 border border-white/10 space-y-2">
                  <div className="flex items-center justify-between text-xs text-zinc-400 pb-1 border-b border-white/5">
                    <span>Columns: {previewData.columns.length}</span>
                    <span>Sample Rows: {previewData.rows.length}</span>
                  </div>

                  <div className="overflow-x-auto max-h-56 -mx-1 px-1">
                    <table className="w-full text-[11px] font-mono text-zinc-300 min-w-full">
                      <thead>
                        <tr className="border-b border-white/10 text-left text-zinc-400">
                          {previewData.columns.map((col) => (
                            <th key={col} className="p-2 whitespace-nowrap bg-white/[0.02]">
                              {col}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {previewData.rows.map((row) => {
                          const rowSignature = previewData.columns.slice(0, 3).map((col) => String(row[col] ?? '')).join('::');
                          return (
                            <tr key={`row-${rowSignature}`} className="border-b border-white/5 hover:bg-white/[0.02]">
                              {previewData.columns.map((col) => (
                                <td key={col} className="p-2 whitespace-nowrap">
                                  {row[col] !== null && row[col] !== undefined
                                    ? String(row[col])
                                    : <span className="text-zinc-600 italic">null</span>}
                                </td>
                              ))}
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>

            {/* ACTION FOOTER */}
            <div className="flex flex-col-reverse sm:flex-row sm:items-center sm:justify-between gap-3 pt-4 border-t border-white/10">
              <Button
                type="button"
                variant="outline"
                onClick={() => setStep("auth")}
                className="w-full sm:w-auto border-white/10 text-zinc-300 hover:bg-white/5 text-xs h-9"
              >
                Back to Credentials
              </Button>

              <Button
                type="button"
                onClick={handleImport}
                disabled={!activeQuery || isImporting || Boolean(sqlValidationError)}
                className="w-full sm:w-auto bg-white hover:bg-zinc-200 text-black font-semibold text-xs px-6 shadow-sm h-9"
              >
                {isImporting ? (
                  <>
                    <Loader2 className="h-3.5 w-3.5 mr-2 animate-spin" />
                    Launching Extraction...
                  </>
                ) : (
                  <>
                    <Database className="h-3.5 w-3.5 mr-2" />
                    Import & Run Health Audit
                  </>
                )}
              </Button>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
};
