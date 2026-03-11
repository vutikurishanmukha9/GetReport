import { useMemo, useRef, useEffect } from "react";
import { motion } from "framer-motion";
import {
    Upload,
    Search,
    UserCheck,
    Sparkles,
    BarChart3,
    PieChart,
    Brain,
    FileText,
    CheckCircle2,
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
    { id: "review", label: "Review", icon: UserCheck },
    { id: "clean", label: "Clean", icon: Sparkles },
    { id: "analyze", label: "Analyze", icon: BarChart3 },
    { id: "visualize", label: "Visualize", icon: PieChart },
    { id: "insights", label: "AI Insights", icon: Brain },
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
    if (status === "WAITING_FOR_USER") return "review";

    if (msg.includes("compiling report") || msg.includes("generating pdf") || msg.includes("rendering pdf") || progress >= 90)
        return "report";
    if (msg.includes("generating insights") || msg.includes("insight") || (progress >= 80 && progress < 90))
        return "insights";
    if (msg.includes("generating chart") || msg.includes("visualization") || (progress >= 70 && progress < 80))
        return "visualize";
    if (msg.includes("analyzing") || msg.includes("statistical") || (progress >= 55 && progress < 70))
        return "analyze";
    if (msg.includes("cleaning") || (progress >= 45 && progress < 55))
        return "clean";
    if (msg.includes("detecting") || msg.includes("issue") || (progress >= 30 && progress < 45))
        return "inspect";
    if (msg.includes("inspecting") || msg.includes("quality") || (progress >= 20 && progress < 30))
        return "inspect";
    if (msg.includes("loading") || (progress >= 5 && progress < 20))
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
    /** Optional: minimum stage index that should already be completed on mount.
     *  When the pipeline remounts in a later phase (e.g. report generation),
     *  set this so earlier stages don't reset to pending. */
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

    // Update high-water if minCompletedStage changes (e.g. remount in later phase)
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

    if (!isActive) return null;

    const isComplete = taskStatus?.toUpperCase() === "COMPLETED";

    return (
        <motion.div
            className="w-full mb-6"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.4, ease: "easeOut" }}
        >
            <div className="relative rounded-2xl border border-border/40 bg-gradient-to-b from-card to-card/80 backdrop-blur-xl shadow-xl overflow-hidden">
                {/* Top accent line */}
                <div className="absolute inset-x-0 top-0 h-[2px] bg-gradient-to-r from-emerald-500/60 via-cyan-400/80 to-violet-500/60" />

                <div className="px-5 py-6 sm:px-8 sm:py-7">
                    {/* Header */}
                    <div className="flex items-center justify-between mb-6">
                        <div className="flex items-center gap-2.5">
                            {isComplete ? (
                                <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                            ) : (
                                <motion.div
                                    className="w-2.5 h-2.5 rounded-full bg-cyan-400"
                                    animate={{ opacity: [1, 0.4, 1] }}
                                    transition={{ repeat: Infinity, duration: 1.5 }}
                                />
                            )}
                            <h3 className="text-xs font-bold text-foreground/70 tracking-[0.2em] uppercase">
                                Processing Pipeline
                            </h3>
                        </div>
                        <span
                            className={`text-xs font-mono font-bold px-2.5 py-1 rounded-md ${
                                isComplete
                                    ? "bg-emerald-500/10 text-emerald-500 border border-emerald-500/20"
                                    : "bg-primary/10 text-primary border border-primary/20"
                            }`}
                        >
                            {progress}%
                        </span>
                    </div>

                    {/* Desktop: Horizontal pipeline */}
                    <div className="hidden md:block">
                        <HorizontalPipeline stages={stages} />
                    </div>

                    {/* Mobile: Vertical pipeline */}
                    <div className="block md:hidden">
                        <VerticalPipeline stages={stages} />
                    </div>

                    {/* Current message */}
                    {message && (
                        <motion.p
                            key={message}
                            className="mt-5 text-center text-xs sm:text-sm text-muted-foreground/80"
                            initial={{ opacity: 0, y: 4 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.3 }}
                        >
                            {isComplete ? "✅ " : ""}{message}
                        </motion.p>
                    )}
                </div>
            </div>
        </motion.div>
    );
};

// ─── Horizontal Pipeline (Desktop) ──────────────────────────────────────────
// The pipe connectors go BETWEEN circles (edge-to-edge), not through them.

const NODE_SIZE = 44; // px — diameter of each junction node

function HorizontalPipeline({ stages }: { stages: PipelineStage[] }) {
    return (
        <div className="flex items-start justify-between relative">
            {stages.map((stage, i) => (
                <div key={stage.id} className="flex items-center flex-1 last:flex-none">
                    {/* Junction Node */}
                    <HorizontalStageNode stage={stage} index={i} />

                    {/* Connector pipe BETWEEN this node and the next (not through them) */}
                    {i < stages.length - 1 && (
                        <div className="flex-1 relative h-[6px] mx-0">
                            {/* Pipe background track */}
                            <div className="absolute inset-0 rounded-full bg-muted/30" />

                            {/* Pipe fill — filled if BOTH this node and the next are completed, 
                                or this node is completed and next is active */}
                            <motion.div
                                className="absolute inset-y-0 left-0 rounded-full overflow-hidden"
                                initial={{ width: "0%" }}
                                animate={{
                                    width: getConnectorFill(stage.status, stages[i + 1]?.status),
                                }}
                                transition={{ duration: 0.5, ease: "easeInOut" }}
                            >
                                <div className="h-full w-full bg-gradient-to-r from-emerald-400 via-cyan-400 to-emerald-400" />
                                {/* Shimmer */}
                                <motion.div
                                    className="absolute inset-0 bg-gradient-to-r from-transparent via-white/25 to-transparent"
                                    animate={{ x: ["-100%", "200%"] }}
                                    transition={{ repeat: Infinity, duration: 2.5, ease: "linear" }}
                                />
                            </motion.div>
                        </div>
                    )}
                </div>
            ))}
        </div>
    );
}

function getConnectorFill(currentStatus: StageStatus, nextStatus?: StageStatus): string {
    if (!nextStatus) return "0%";
    // This connector is fully filled when both sides are completed
    if (currentStatus === "completed" && (nextStatus === "completed" || nextStatus === "active")) {
        return "100%";
    }
    // Half-filled when current is active (flow is reaching this connector)
    if (currentStatus === "active") {
        return "50%";
    }
    // Completed on left side only
    if (currentStatus === "completed") {
        return "100%";
    }
    return "0%";
}

// ─── Vertical Pipeline (Mobile) ─────────────────────────────────────────────

function VerticalPipeline({ stages }: { stages: PipelineStage[] }) {
    return (
        <div className="flex flex-col relative">
            {stages.map((stage, i) => (
                <div key={stage.id} className="flex flex-col items-start">
                    {/* Row: node + label */}
                    <div className="flex items-center gap-4">
                        <VerticalStageNode stage={stage} index={i} />
                        <span
                            className={`text-sm font-medium
                                ${stage.status === "completed"
                                    ? "text-emerald-500/90"
                                    : stage.status === "active"
                                    ? "text-cyan-400 font-semibold"
                                    : stage.status === "error"
                                    ? "text-destructive"
                                    : "text-muted-foreground/30"
                                }
                            `}
                        >
                            {stage.label}
                        </span>
                    </div>

                    {/* Vertical connector pipe between nodes */}
                    {i < stages.length - 1 && (
                        <div className="ml-[18px] w-[6px] h-6 relative">
                            <div className="absolute inset-0 rounded-full bg-muted/30" />
                            <motion.div
                                className="absolute inset-x-0 top-0 rounded-full overflow-hidden"
                                initial={{ height: "0%" }}
                                animate={{
                                    height: getConnectorFill(stage.status, stages[i + 1]?.status),
                                }}
                                transition={{ duration: 0.5, ease: "easeInOut" }}
                            >
                                <div className="h-full w-full bg-gradient-to-b from-emerald-400 via-cyan-400 to-emerald-400" />
                            </motion.div>
                        </div>
                    )}
                </div>
            ))}
        </div>
    );
}

// ─── Horizontal Stage Node ──────────────────────────────────────────────────

function HorizontalStageNode({
    stage,
    index,
}: {
    stage: PipelineStage;
    index: number;
}) {
    const Icon = stage.icon;

    return (
        <motion.div
            className="flex flex-col items-center gap-2.5 shrink-0"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.06, duration: 0.35 }}
        >
            {/* Pipe junction node */}
            <div className="relative">
                {/* Pulsing ring for active */}
                {stage.status === "active" && (
                    <motion.div
                        className="absolute -inset-2 rounded-full border-2 border-cyan-400/40"
                        animate={{ scale: [1, 1.15, 1], opacity: [0.5, 0.15, 0.5] }}
                        transition={{ repeat: Infinity, duration: 2, ease: "easeInOut" }}
                    />
                )}

                {/* Main circle */}
                <div
                    className={`
                        w-11 h-11 rounded-full border-[3px] flex items-center justify-center
                        transition-all duration-500 relative z-10 bg-card
                        ${stage.status === "completed"
                            ? "border-emerald-500/70 text-emerald-500"
                            : stage.status === "active"
                            ? "border-cyan-400/80 text-cyan-400"
                            : stage.status === "error"
                            ? "border-destructive/70 text-destructive"
                            : "border-muted-foreground/15 text-muted-foreground/25"
                        }
                    `}
                >
                    {stage.status === "completed" ? (
                        <motion.div
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            transition={{ type: "spring", stiffness: 500, damping: 15 }}
                        >
                            <CheckCircle2 className="w-5 h-5" />
                        </motion.div>
                    ) : stage.status === "active" ? (
                        <motion.div
                            animate={{ scale: [1, 0.85, 1] }}
                            transition={{ repeat: Infinity, duration: 1.5, ease: "easeInOut" }}
                        >
                            <Icon className="w-5 h-5" />
                        </motion.div>
                    ) : (
                        <Icon className="w-4 h-4" />
                    )}
                </div>
            </div>

            {/* Label */}
            <span
                className={`text-[10px] sm:text-[11px] text-center leading-tight font-medium tracking-wide
                    ${stage.status === "completed"
                        ? "text-emerald-500/90"
                        : stage.status === "active"
                        ? "text-cyan-400 font-semibold"
                        : stage.status === "error"
                        ? "text-destructive"
                        : "text-muted-foreground/30"
                    }
                `}
            >
                {stage.label}
            </span>
        </motion.div>
    );
}

// ─── Vertical Stage Node ────────────────────────────────────────────────────

function VerticalStageNode({
    stage,
    index,
}: {
    stage: PipelineStage;
    index: number;
}) {
    const Icon = stage.icon;

    return (
        <motion.div
            className="relative shrink-0"
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.05, duration: 0.3 }}
        >
            {stage.status === "active" && (
                <motion.div
                    className="absolute -inset-1.5 rounded-full border-2 border-cyan-400/40"
                    animate={{ scale: [1, 1.15, 1], opacity: [0.5, 0.15, 0.5] }}
                    transition={{ repeat: Infinity, duration: 2, ease: "easeInOut" }}
                />
            )}

            <div
                className={`
                    w-[38px] h-[38px] rounded-full border-[3px] flex items-center justify-center
                    transition-all duration-500 bg-card relative z-10
                    ${stage.status === "completed"
                        ? "border-emerald-500/70 text-emerald-500"
                        : stage.status === "active"
                        ? "border-cyan-400/80 text-cyan-400"
                        : stage.status === "error"
                        ? "border-destructive/70 text-destructive"
                        : "border-muted-foreground/15 text-muted-foreground/25"
                    }
                `}
            >
                {stage.status === "completed" ? (
                    <CheckCircle2 className="w-4 h-4" />
                ) : stage.status === "active" ? (
                    <motion.div
                        animate={{ scale: [1, 0.85, 1] }}
                        transition={{ repeat: Infinity, duration: 1.5, ease: "easeInOut" }}
                    >
                        <Icon className="w-4 h-4" />
                    </motion.div>
                ) : (
                    <Icon className="w-4 h-4" />
                )}
            </div>
        </motion.div>
    );
}
