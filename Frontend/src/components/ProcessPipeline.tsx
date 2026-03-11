import { useMemo, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
    Upload,
    Search,
    Sparkles,
    BarChart3,
    Brain,
    FileText,
    CheckCircle2,
    Loader2,
} from "lucide-react";

// ─── Pipeline Stage Definitions ─────────────────────────────────────────────

export type StageStatus = "pending" | "active" | "completed" | "error";

interface PipelineStage {
    id: string;
    label: string;
    icon: React.ElementType;
    status: StageStatus;
}

const STAGE_DEFINITIONS = [
    { id: "upload", label: "Upload", icon: Upload },
    { id: "inspect", label: "Inspect", icon: Search },
    { id: "prepare", label: "Prepare", icon: Sparkles },
    { id: "analyze", label: "Analyze", icon: BarChart3 },
    { id: "insights", label: "Insights", icon: Brain },
    { id: "report", label: "Report", icon: FileText },
] as const;

const STAGE_INDEX_MAP: Record<string, number> = {};
STAGE_DEFINITIONS.forEach((s, i) => (STAGE_INDEX_MAP[s.id] = i));

// ─── Message → Stage Mapping ────────────────────────────────────────────────

function resolveCurrentStage(
    taskStatus: string,
    message: string,
    progress: number
): string {
    const msg = (message || "").toLowerCase();
    const status = (taskStatus || "").toUpperCase();

    if (status === "COMPLETED") return "report";
    if (status === "FAILED") return "error";
    if (status === "WAITING_FOR_USER") return "prepare";

    if (msg.includes("compiling report") || msg.includes("generating pdf") || msg.includes("rendering pdf") || progress >= 90)
        return "report";
    if (msg.includes("generating insights") || msg.includes("insight") || (progress >= 80 && progress < 90))
        return "insights";
    if (msg.includes("generating chart") || msg.includes("visualization") || msg.includes("analyzing") || msg.includes("statistical") || (progress >= 55 && progress < 80))
        return "analyze";
    if (msg.includes("cleaning") || (progress >= 45 && progress < 55))
        return "prepare";
    if (msg.includes("detecting") || msg.includes("issue") || msg.includes("inspecting") || msg.includes("quality") || msg.includes("loading") || (progress >= 5 && progress < 45))
        return "inspect";
    if (msg.includes("upload") || msg.includes("sending") || progress > 0)
        return "upload";

    return "upload";
}

// ─── Props ──────────────────────────────────────────────────────────────────

interface ProcessPipelineProps {
    taskStatus: string;
    message: string;
    progress: number;
    isActive: boolean;
    /** Optional: minimum stage index that should already be completed on mount. */
    minCompletedStage?: number;
}

// ─── Component ──────────────────────────────────────────────────────────────

export const ProcessPipeline = ({
    taskStatus,
    message,
    progress,
    isActive,
    minCompletedStage,
}: ProcessPipelineProps) => {
    // High-water-mark: pipeline NEVER goes backwards
    const highWaterRef = useRef<number>(minCompletedStage ?? -1);

    // Update high-water if minCompletedStage changes
    useEffect(() => {
        if (minCompletedStage !== undefined && minCompletedStage > highWaterRef.current) {
            highWaterRef.current = minCompletedStage;
        }
    }, [minCompletedStage]);

    const stages: PipelineStage[] = useMemo(() => {
        if (!isActive) {
            return STAGE_DEFINITIONS.map((s) => ({ ...s, status: "pending" as StageStatus }));
        }

        const isError = taskStatus?.toUpperCase() === "FAILED";
        const isComplete = taskStatus?.toUpperCase() === "COMPLETED";

        if (isComplete) {
            highWaterRef.current = STAGE_DEFINITIONS.length - 1;
            return STAGE_DEFINITIONS.map((stage) => ({
                ...stage,
                status: "completed" as StageStatus,
            }));
        }

        const rawStageId = resolveCurrentStage(taskStatus, message, progress);
        const rawIndex = STAGE_INDEX_MAP[rawStageId] ?? 0;
        const currentIndex = Math.max(rawIndex, highWaterRef.current);
        highWaterRef.current = currentIndex;

        return STAGE_DEFINITIONS.map((stage, i) => {
            if (isError && i === currentIndex) {
                return { ...stage, status: "error" as StageStatus };
            }
            if (i < currentIndex) {
                return { ...stage, status: "completed" as StageStatus };
            }
            if (i === currentIndex) {
                return { ...stage, status: "active" as StageStatus };
            }
            return { ...stage, status: "pending" as StageStatus };
        });
    }, [taskStatus, message, progress, isActive]);

    // Calculate realistic overall progress
    const displayProgress = useMemo(() => {
        const isComplete = taskStatus?.toUpperCase() === "COMPLETED";
        if (isComplete) return 100;
        
        const rawProgress = progress || 0;
        const currentIndex = Math.max(0, highWaterRef.current);
        const totalStages = STAGE_DEFINITIONS.length; // 6
        
        // If we are at stage index 4 (Insights), we are 4/6 = 66% done.
        // We map the rawProgress (0-100) to the remaining portion (33%).
        const baseProgressPercent = (currentIndex / totalStages) * 100;
        const remainingPercent = 100 - baseProgressPercent;
        
        // The relative progress within the current stage(s)
        const additionalProgress = (rawProgress / 100) * remainingPercent;
        
        return Math.min(99, Math.floor(baseProgressPercent + additionalProgress));
    }, [progress, taskStatus, highWaterRef.current]);

    if (!isActive) return null;

    const isComplete = taskStatus?.toUpperCase() === "COMPLETED";
    
    // Find the currently active or recently active stage for the status banner
    const activeStage = stages.find(s => s.status === "active" || s.status === "error") 
        || (isComplete ? stages[stages.length - 1] : stages[0]);

    return (
        <motion.div
            className="w-full mb-8"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.4, ease: "easeOut" }}
        >
            <div className="bg-card rounded-xl border border-border/50 shadow-sm overflow-hidden flex flex-col">
                
                {/* Header Section */}
                <div className="px-6 pt-5 pb-4 border-b border-border/50 flex items-center justify-between bg-muted/20">
                    <div className="flex items-center gap-3">
                        <div className="flex h-8 w-8 items-center justify-center rounded-md bg-primary/10 text-primary">
                            {isComplete ? (
                                <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                            ) : (
                                <Loader2 className="w-4 h-4 animate-spin text-cyan-500" />
                            )}
                        </div>
                        <div>
                            <h3 className="text-sm font-semibold tracking-tight">Processing your dataset</h3>
                            <p className="text-xs text-muted-foreground mt-0.5">
                                AI is analyzing and generating insights
                            </p>
                        </div>
                    </div>
                    <div className="flex text-right">
                        <span className="text-2xl font-semibold tracking-tight tabular-nums animate-in fade-in slide-in-from-bottom-2">
                            {displayProgress}
                            <span className="text-sm text-muted-foreground ml-0.5">%</span>
                        </span>
                    </div>
                </div>

                {/* Pipeline Stepper (Desktop) */}
                <div className="px-6 py-6 overflow-x-auto hidden md:block hide-scrollbar">
                    <SaaSStepper stages={stages} />
                </div>
                
                {/* Vertical Stepper (Mobile) */}
                <div className="px-6 py-6 block md:hidden">
                    <VerticalStepper stages={stages} />
                </div>

                {/* Live Status Subtext Line */}
                <div className="border-t border-border/50 bg-muted/10 px-6 py-3.5 flex items-center gap-3">
                    {/* Pulsing indicator */}
                    {!isComplete && (
                        <span className="relative flex h-2.5 w-2.5 shrink-0">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-cyan-500"></span>
                        </span>
                    )}
                    {isComplete && (
                         <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
                    )}
                    
                    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between w-full gap-1 sm:gap-4">
                        <div className="flex items-center gap-2 text-sm">
                            <span className="font-medium text-foreground">
                                {isComplete ? "Completed" : "Currently running:"}
                            </span>
                            <span className="text-muted-foreground truncate max-w-[200px] sm:max-w-md">
                                {message || `Processing ${activeStage.label.toLowerCase()}...`}
                            </span>
                        </div>
                        
                        {!isComplete && (
                            <span className="text-xs font-medium text-muted-foreground/70 shrink-0">
                                Please wait...
                            </span>
                        )}
                    </div>
                </div>

            </div>
        </motion.div>
    );
};

// ─── SaaS Stepper Components (Desktop) ──────────────────────────────────────

function SaaSStepper({ stages }: { stages: PipelineStage[] }) {
    return (
        <div className="flex items-center min-w-max w-full">
            {stages.map((stage, i) => {
                const isLast = i === stages.length - 1;
                return (
                    <div key={stage.id} className={`flex items-center ${isLast ? "flex-none" : "flex-1"}`}>
                        <StepNode stage={stage} />
                        
                        {/* Connector Line */}
                        {!isLast && (
                            <div className="flex-1 shrink-0 px-3">
                                <div className="h-[2px] w-full rounded-full bg-muted/40 relative overflow-hidden">
                                    <motion.div
                                        className="absolute inset-y-0 left-0 bg-gradient-to-r from-emerald-400 to-cyan-400"
                                        initial={{ width: "0%" }}
                                        animate={{
                                            width: getConnectorFill(stage.status, stages[i + 1]?.status),
                                        }}
                                        transition={{ duration: 0.6, ease: "easeInOut" }}
                                    />
                                    {/* Shimmer pulse for active running line */}
                                    {stage.status === "active" && stages[i + 1]?.status !== "active" && (
                                        <motion.div
                                            className="absolute inset-y-0 w-1/3 bg-gradient-to-r from-transparent via-white/40 to-transparent"
                                            animate={{ x: ["-100%", "300%"] }}
                                            transition={{ repeat: Infinity, duration: 1.5, ease: "linear" }}
                                        />
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                );
            })}
        </div>
    );
}

function StepNode({ stage }: { stage: PipelineStage }) {
    const Icon = stage.icon;
    const isCompleted = stage.status === "completed";
    const isActive = stage.status === "active";
    const isError = stage.status === "error";

    return (
        <div className="flex items-center gap-2 shrink-0">
            {/* Status Indicator */}
            <div className="relative flex items-center justify-center w-6 h-6 shrink-0">
                <AnimatePresence mode="popLayout">
                    {isCompleted && (
                        <motion.div
                            key="completed"
                            initial={{ scale: 0.5, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            className="text-emerald-500 bg-emerald-500/10 rounded-full p-0.5"
                        >
                            <CheckCircle2 className="w-4 h-4" />
                        </motion.div>
                    )}
                    {isActive && (
                        <motion.div
                            key="active"
                            initial={{ scale: 0.5, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            className="relative flex items-center justify-center"
                        >
                            <div className="absolute inset-0 rounded-full bg-cyan-400/20 blur-sm glow-effect" />
                            <div className="w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.8)] relative z-10" />
                        </motion.div>
                    )}
                    {isError && (
                        <motion.div
                            key="error"
                            initial={{ scale: 0.5, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            className="w-2.5 h-2.5 rounded-full bg-destructive shadow-[0_0_8px_rgba(239,68,68,0.8)]"
                        />
                    )}
                    {stage.status === "pending" && (
                        <motion.div
                            key="pending"
                            initial={{ scale: 0.5, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            className="w-2 h-2 rounded-full border border-muted-foreground/30"
                        />
                    )}
                </AnimatePresence>
            </div>

            {/* Label & Icon */}
            <div className={`flex items-center gap-1.5 transition-colors duration-300
                ${isCompleted ? "text-emerald-600 dark:text-emerald-400" :
                  isActive ? "text-foreground font-semibold" :
                  isError ? "text-destructive font-semibold" :
                  "text-muted-foreground/60 font-medium"}
            `}>
                <span className="text-sm tracking-tight">{stage.label}</span>
            </div>
        </div>
    );
}

// ─── Vertical Stepper (Mobile) ──────────────────────────────────────────────

function VerticalStepper({ stages }: { stages: PipelineStage[] }) {
    return (
        <div className="flex flex-col gap-0 w-full relative">
            <div className="absolute left-[11px] top-3 bottom-5 w-[2px] bg-muted/40 rounded-full" />
            
            {stages.map((stage, i) => {
                const isLast = i === stages.length - 1;
                const isCompleted = stage.status === "completed";
                const isActive = stage.status === "active";
                const isError = stage.status === "error";

                return (
                    <div key={stage.id} className="flex flex-col">
                        <div className="flex items-center gap-4 py-2 relative z-10">
                            {/* Vertical Status Indicator */}
                            <div className="relative flex items-center justify-center w-6 h-6 shrink-0 bg-card rounded-full">
                                {isCompleted && (
                                    <div className="text-emerald-500 bg-emerald-500/10 rounded-full p-0.5">
                                        <CheckCircle2 className="w-4 h-4" />
                                    </div>
                                )}
                                {isActive && (
                                    <div className="relative flex items-center justify-center">
                                        <div className="absolute inset-0 rounded-full bg-cyan-400/20 blur-sm glow-effect" />
                                        <div className="w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.8)] relative z-10" />
                                    </div>
                                )}
                                {isError && (
                                    <div className="w-2.5 h-2.5 rounded-full bg-destructive shadow-[0_0_8px_rgba(239,68,68,0.8)]" />
                                )}
                                {stage.status === "pending" && (
                                    <div className="w-2 h-2 rounded-full border border-muted-foreground/30 bg-card" />
                                )}
                            </div>

                            <span className={`text-sm tracking-tight transition-colors duration-300
                                ${isCompleted ? "text-emerald-600 dark:text-emerald-400" :
                                isActive ? "text-foreground font-semibold" :
                                isError ? "text-destructive font-semibold" :
                                "text-muted-foreground/60 font-medium"}
                            `}>
                                {stage.label}
                            </span>
                        </div>
                        
                        {/* Fill line for mobile */}
                        {!isLast && (
                            <div className="ml-[11px] h-6 w-[2px] relative -mt-2 -mb-2 z-0">
                                <motion.div
                                    className="absolute inset-x-0 top-0 bg-gradient-to-b from-emerald-400 to-cyan-400"
                                    initial={{ height: "0%" }}
                                    animate={{ height: getConnectorFill(stage.status, stages[i + 1]?.status) }}
                                    transition={{ duration: 0.6, ease: "easeInOut" }}
                                />
                            </div>
                        )}
                    </div>
                );
            })}
        </div>
    );
}

// Helper to determine fill percentage of the connector line
function getConnectorFill(currentStatus: StageStatus, nextStatus?: StageStatus): string {
    if (!nextStatus) return "0%";
    if (currentStatus === "completed" && (nextStatus === "completed" || nextStatus === "active")) {
        return "100%";
    }
    if (currentStatus === "active") {
        return "50%";
    }
    if (currentStatus === "completed") {
        return "100%";
    }
    return "0%";
}
