import { useMemo } from "react";
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
    Check,
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
    { id: "review", label: "Review", icon: UserCheck },
    { id: "clean", label: "Clean", icon: Sparkles },
    { id: "analyze", label: "Analyze", icon: BarChart3 },
    { id: "visualize", label: "Visualize", icon: PieChart },
    { id: "insights", label: "AI Insights", icon: Brain },
    { id: "report", label: "Report", icon: FileText },
] as const;

// ─── Message → Stage Mapping ────────────────────────────────────────────────

function resolveCurrentStage(
    taskStatus: string,
    message: string,
    progress: number
): string {
    const msg = (message || "").toLowerCase();
    const status = (taskStatus || "").toUpperCase();

    // Terminal states
    if (status === "COMPLETED") return "report";
    if (status === "FAILED") return "error";

    // Review pause
    if (status === "WAITING_FOR_USER") return "review";

    // Match by message content
    if (msg.includes("compiling report") || msg.includes("generating pdf") || progress >= 90)
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
    /** Current task status from WebSocket (e.g. PROCESSING, WAITING_FOR_USER, COMPLETED) */
    taskStatus: string;
    /** Current progress message from backend */
    message: string;
    /** Progress percentage 0-100 */
    progress: number;
    /** If true, the pipeline is visible and active */
    isActive: boolean;
}

// ─── Component ──────────────────────────────────────────────────────────────

export const ProcessPipeline = ({
    taskStatus,
    message,
    progress,
    isActive,
}: ProcessPipelineProps) => {
    // Derive stages with statuses
    const stages: PipelineStage[] = useMemo(() => {
        if (!isActive) {
            return STAGE_DEFINITIONS.map((s) => ({ ...s, status: "pending" as StageStatus }));
        }

        const currentStageId = resolveCurrentStage(taskStatus, message, progress);
        const isError = taskStatus?.toUpperCase() === "FAILED";
        const isComplete = taskStatus?.toUpperCase() === "COMPLETED";

        let passedCurrent = false;

        return STAGE_DEFINITIONS.map((stage) => {
            if (isComplete) {
                return { ...stage, status: "completed" as StageStatus };
            }

            if (isError && stage.id === currentStageId) {
                return { ...stage, status: "error" as StageStatus };
            }

            if (stage.id === currentStageId) {
                passedCurrent = true;
                return { ...stage, status: isError ? "error" : "active" as StageStatus };
            }

            if (!passedCurrent) {
                return { ...stage, status: "completed" as StageStatus };
            }

            return { ...stage, status: "pending" as StageStatus };
        });
    }, [taskStatus, message, progress, isActive]);

    if (!isActive) return null;

    return (
        <motion.div
            className="w-full mb-6"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.4, ease: "easeOut" }}
        >
            {/* Pipeline Card */}
            <div className="relative rounded-xl border border-border/50 bg-card/80 backdrop-blur-md shadow-lg overflow-hidden">
                {/* Subtle top gradient accent */}
                <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-primary/50 to-transparent" />

                <div className="px-4 py-5 sm:px-6 sm:py-6">
                    {/* Header */}
                    <div className="flex items-center justify-between mb-5">
                        <h3 className="text-sm font-semibold text-foreground/80 tracking-wide uppercase">
                            Processing Pipeline
                        </h3>
                        <span className="text-xs text-muted-foreground font-mono">
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
                            className="mt-4 text-center text-xs sm:text-sm text-muted-foreground italic"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ duration: 0.3 }}
                        >
                            {message}
                        </motion.p>
                    )}
                </div>
            </div>
        </motion.div>
    );
};

// ─── Horizontal Pipeline (Desktop) ──────────────────────────────────────────

function HorizontalPipeline({ stages }: { stages: PipelineStage[] }) {
    return (
        <div className="flex items-center justify-between relative">
            {/* Background connector line */}
            <div className="absolute top-5 left-5 right-5 h-0.5 bg-border/30 z-0" />

            {/* Progress fill line */}
            <motion.div
                className="absolute top-5 left-5 h-0.5 bg-gradient-to-r from-emerald-500 via-primary to-primary z-[1]"
                initial={{ width: 0 }}
                animate={{
                    width: `${getProgressWidth(stages)}%`,
                }}
                transition={{ duration: 0.6, ease: "easeInOut" }}
            />

            {stages.map((stage, i) => (
                <StageNode key={stage.id} stage={stage} index={i} layout="horizontal" />
            ))}
        </div>
    );
}

// ─── Vertical Pipeline (Mobile) ─────────────────────────────────────────────

function VerticalPipeline({ stages }: { stages: PipelineStage[] }) {
    return (
        <div className="flex flex-col gap-0 relative pl-5">
            {/* Background connector line */}
            <div className="absolute top-5 bottom-5 left-[19px] w-0.5 bg-border/30 z-0" />

            {/* Progress fill line */}
            <motion.div
                className="absolute top-5 left-[19px] w-0.5 bg-gradient-to-b from-emerald-500 via-primary to-primary z-[1]"
                initial={{ height: 0 }}
                animate={{
                    height: `${getProgressWidth(stages)}%`,
                }}
                transition={{ duration: 0.6, ease: "easeInOut" }}
            />

            {stages.map((stage, i) => (
                <StageNode key={stage.id} stage={stage} index={i} layout="vertical" />
            ))}
        </div>
    );
}

// ─── Stage Node ─────────────────────────────────────────────────────────────

function StageNode({
    stage,
    index,
    layout,
}: {
    stage: PipelineStage;
    index: number;
    layout: "horizontal" | "vertical";
}) {
    const Icon = stage.icon;

    const nodeColors: Record<StageStatus, string> = {
        completed: "bg-emerald-500/15 border-emerald-500/60 text-emerald-500",
        active: "bg-primary/15 border-primary text-primary",
        pending: "bg-muted/50 border-border/50 text-muted-foreground/50",
        error: "bg-destructive/15 border-destructive/60 text-destructive",
    };

    const labelColors: Record<StageStatus, string> = {
        completed: "text-emerald-500/90",
        active: "text-primary font-semibold",
        pending: "text-muted-foreground/40",
        error: "text-destructive",
    };

    if (layout === "vertical") {
        return (
            <motion.div
                className="flex items-center gap-3 py-2 relative z-10"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05, duration: 0.3 }}
            >
                {/* Node circle */}
                <div className="relative">
                    <motion.div
                        className={`
              w-[38px] h-[38px] rounded-full border-2 flex items-center justify-center
              transition-colors duration-300 -ml-5
              ${nodeColors[stage.status]}
            `}
                        animate={stage.status === "active" ? { scale: [1, 1.08, 1] } : {}}
                        transition={
                            stage.status === "active"
                                ? { repeat: Infinity, duration: 1.5, ease: "easeInOut" }
                                : {}
                        }
                    >
                        {stage.status === "completed" ? (
                            <Check className="w-4 h-4" />
                        ) : stage.status === "active" ? (
                            <motion.div
                                animate={{ rotate: 360 }}
                                transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
                            >
                                <Loader2 className="w-4 h-4" />
                            </motion.div>
                        ) : (
                            <Icon className="w-4 h-4" />
                        )}
                    </motion.div>

                    {/* Active glow */}
                    {stage.status === "active" && (
                        <motion.div
                            className="absolute inset-0 -ml-5 rounded-full bg-primary/20 blur-md"
                            animate={{ opacity: [0.3, 0.6, 0.3] }}
                            transition={{ repeat: Infinity, duration: 2, ease: "easeInOut" }}
                        />
                    )}
                </div>

                {/* Label */}
                <span className={`text-sm ${labelColors[stage.status]}`}>
                    {stage.label}
                </span>
            </motion.div>
        );
    }

    // Horizontal layout
    return (
        <motion.div
            className="flex flex-col items-center gap-2 relative z-10 flex-1"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.06, duration: 0.3 }}
        >
            {/* Node circle */}
            <div className="relative">
                <motion.div
                    className={`
            w-10 h-10 rounded-full border-2 flex items-center justify-center
            transition-colors duration-300
            ${nodeColors[stage.status]}
          `}
                    animate={stage.status === "active" ? { scale: [1, 1.1, 1] } : {}}
                    transition={
                        stage.status === "active"
                            ? { repeat: Infinity, duration: 1.5, ease: "easeInOut" }
                            : {}
                    }
                >
                    {stage.status === "completed" ? (
                        <Check className="w-4 h-4" />
                    ) : stage.status === "active" ? (
                        <motion.div
                            animate={{ rotate: 360 }}
                            transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
                        >
                            <Loader2 className="w-4 h-4" />
                        </motion.div>
                    ) : (
                        <Icon className="w-4 h-4" />
                    )}
                </motion.div>

                {/* Active glow */}
                {stage.status === "active" && (
                    <motion.div
                        className="absolute inset-0 rounded-full bg-primary/20 blur-md"
                        animate={{ opacity: [0.3, 0.7, 0.3] }}
                        transition={{ repeat: Infinity, duration: 2, ease: "easeInOut" }}
                    />
                )}
            </div>

            {/* Label */}
            <span
                className={`text-[10px] sm:text-xs text-center leading-tight ${labelColors[stage.status]}`}
            >
                {stage.label}
            </span>
        </motion.div>
    );
}

// ─── Utilities ──────────────────────────────────────────────────────────────

function getProgressWidth(stages: PipelineStage[]): number {
    const lastCompleted = stages.reduce(
        (acc, stage, i) => (stage.status === "completed" || stage.status === "active" ? i : acc),
        -1
    );
    if (lastCompleted < 0) return 0;
    // Calculate percentage through the pipeline (accounting for spacing)
    return Math.min(((lastCompleted) / (stages.length - 1)) * 100, 100);
}
