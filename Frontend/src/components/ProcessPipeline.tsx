import { useMemo, useRef } from "react";
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
}

// ─── Component ──────────────────────────────────────────────────────────────

export const ProcessPipeline = ({
    taskStatus,
    message,
    progress,
    isActive,
}: ProcessPipelineProps) => {
    // High-water-mark: pipeline NEVER goes backwards
    const highWaterRef = useRef<number>(-1);

    const stages: PipelineStage[] = useMemo(() => {
        if (!isActive) {
            highWaterRef.current = -1;
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

function HorizontalPipeline({ stages }: { stages: PipelineStage[] }) {
    const fillPercent = getProgressWidth(stages);

    return (
        <div className="relative">
            {/* The pipe track (background) */}
            <div className="absolute top-[22px] left-[20px] right-[20px] h-[6px] rounded-full bg-muted/40 z-0" />

            {/* The flowing fill inside the pipe */}
            <motion.div
                className="absolute top-[22px] left-[20px] h-[6px] rounded-full z-[1] overflow-hidden"
                initial={{ width: 0 }}
                animate={{ width: `calc(${fillPercent}% - 0px)` }}
                style={{ maxWidth: "calc(100% - 40px)" }}
                transition={{ duration: 0.8, ease: "easeInOut" }}
            >
                {/* Gradient fill */}
                <div className="h-full w-full bg-gradient-to-r from-emerald-400 via-cyan-400 to-violet-500 rounded-full" />

                {/* Flowing shimmer animation */}
                <motion.div
                    className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent rounded-full"
                    animate={{ x: ["-100%", "200%"] }}
                    transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                />
            </motion.div>

            {/* Stage nodes */}
            <div className="flex items-start justify-between relative z-10">
                {stages.map((stage, i) => (
                    <HorizontalStageNode key={stage.id} stage={stage} index={i} total={stages.length} />
                ))}
            </div>
        </div>
    );
}

// ─── Vertical Pipeline (Mobile) ─────────────────────────────────────────────

function VerticalPipeline({ stages }: { stages: PipelineStage[] }) {
    const fillPercent = getProgressWidth(stages);

    return (
        <div className="relative pl-5">
            {/* The pipe track (background) */}
            <div className="absolute top-[20px] bottom-[20px] left-[18px] w-[6px] rounded-full bg-muted/40 z-0" />

            {/* The flowing fill inside the pipe */}
            <motion.div
                className="absolute top-[20px] left-[18px] w-[6px] rounded-full z-[1] overflow-hidden"
                initial={{ height: 0 }}
                animate={{ height: `${fillPercent}%` }}
                style={{ maxHeight: "calc(100% - 40px)" }}
                transition={{ duration: 0.8, ease: "easeInOut" }}
            >
                <div className="h-full w-full bg-gradient-to-b from-emerald-400 via-cyan-400 to-violet-500 rounded-full" />
                <motion.div
                    className="absolute inset-0 bg-gradient-to-b from-transparent via-white/30 to-transparent rounded-full"
                    animate={{ y: ["-100%", "200%"] }}
                    transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                />
            </motion.div>

            {/* Stage nodes */}
            <div className="flex flex-col gap-0 relative z-10">
                {stages.map((stage, i) => (
                    <VerticalStageNode key={stage.id} stage={stage} index={i} />
                ))}
            </div>
        </div>
    );
}

// ─── Horizontal Stage Node ──────────────────────────────────────────────────

function HorizontalStageNode({
    stage,
    index,
    total,
}: {
    stage: PipelineStage;
    index: number;
    total: number;
}) {
    const Icon = stage.icon;

    return (
        <motion.div
            className="flex flex-col items-center gap-2.5 flex-1"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.06, duration: 0.35 }}
        >
            {/* Pipe junction node */}
            <div className="relative">
                {/* Outer ring for active state */}
                {stage.status === "active" && (
                    <motion.div
                        className="absolute -inset-1.5 rounded-full border-2 border-cyan-400/40"
                        animate={{ scale: [1, 1.2, 1], opacity: [0.6, 0.2, 0.6] }}
                        transition={{ repeat: Infinity, duration: 2, ease: "easeInOut" }}
                    />
                )}

                {/* Main junction circle */}
                <div
                    className={`
                        w-11 h-11 rounded-full border-[3px] flex items-center justify-center
                        transition-all duration-500
                        ${stage.status === "completed"
                            ? "border-emerald-500/70 bg-emerald-500/15 text-emerald-500"
                            : stage.status === "active"
                            ? "border-cyan-400/80 bg-cyan-400/10 text-cyan-400"
                            : stage.status === "error"
                            ? "border-destructive/70 bg-destructive/10 text-destructive"
                            : "border-muted-foreground/15 bg-muted/20 text-muted-foreground/25"
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
                        <Icon className={`w-4 h-4 ${stage.status === "error" ? "w-5 h-5" : ""}`} />
                    )}
                </div>

                {/* Active glow dot */}
                {stage.status === "active" && (
                    <motion.div
                        className="absolute -bottom-0.5 left-1/2 -translate-x-1/2 w-1.5 h-1.5 rounded-full bg-cyan-400"
                        animate={{ opacity: [1, 0.3, 1] }}
                        transition={{ repeat: Infinity, duration: 1 }}
                    />
                )}
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
            className="flex items-center gap-4 py-2.5 relative"
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.05, duration: 0.3 }}
        >
            {/* Pipe junction */}
            <div className="relative">
                {stage.status === "active" && (
                    <motion.div
                        className="absolute -inset-1.5 -ml-[20px] rounded-full border-2 border-cyan-400/40"
                        animate={{ scale: [1, 1.2, 1], opacity: [0.6, 0.2, 0.6] }}
                        transition={{ repeat: Infinity, duration: 2, ease: "easeInOut" }}
                    />
                )}
                
                <div
                    className={`
                        w-[38px] h-[38px] rounded-full border-[3px] flex items-center justify-center
                        transition-all duration-500 -ml-[20px]
                        ${stage.status === "completed"
                            ? "border-emerald-500/70 bg-emerald-500/15 text-emerald-500"
                            : stage.status === "active"
                            ? "border-cyan-400/80 bg-cyan-400/10 text-cyan-400"
                            : stage.status === "error"
                            ? "border-destructive/70 bg-destructive/10 text-destructive"
                            : "border-muted-foreground/15 bg-muted/20 text-muted-foreground/25"
                        }
                    `}
                >
                    {stage.status === "completed" ? (
                        <motion.div
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            transition={{ type: "spring", stiffness: 500, damping: 15 }}
                        >
                            <CheckCircle2 className="w-4 h-4" />
                        </motion.div>
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
            </div>

            {/* Label */}
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
        </motion.div>
    );
}

// ─── Utilities ──────────────────────────────────────────────────────────────

function getProgressWidth(stages: PipelineStage[]): number {
    const allCompleted = stages.every((s) => s.status === "completed");
    if (allCompleted) return 100;

    const lastActiveOrComplete = stages.reduce(
        (acc, stage, i) => (stage.status === "completed" || stage.status === "active" ? i : acc),
        -1
    );
    if (lastActiveOrComplete < 0) return 0;
    return Math.min(((lastActiveOrComplete) / (stages.length - 1)) * 100, 100);
}
