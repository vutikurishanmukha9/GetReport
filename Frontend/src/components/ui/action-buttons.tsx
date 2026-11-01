import * as React from "react";
import { motion, AnimatePresence, useReducedMotion } from "framer-motion";
import {
  CheckCheck,
  Check,
  Trash2,
  X,
  XCircle,
  Download,
  Bookmark,
  BookmarkCheck,
  ArrowRight,
  Loader2,
  ChevronRight,
  AlertTriangle
} from "lucide-react";
import { cn } from "@/lib/utils";

// ─────────────────────────────────────────────────────────────────────────────
// 1. SAVE ALL / APPROVE ALL BUTTON
// Inspired by: 21st.dev + Magic UI Shimmer + Linear Enterprise Standard
// ─────────────────────────────────────────────────────────────────────────────

export interface SaveAllButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  count?: number;
  label?: string;
  loading?: boolean;
  completed?: boolean;
  size?: "sm" | "default" | "lg";
}

export const SaveAllButton = React.forwardRef<HTMLButtonElement, SaveAllButtonProps>(
  (
    {
      count,
      label = "Approve All",
      loading = false,
      completed = false,
      disabled = false,
      className,
      size = "default",
      onClick,
      ...props
    },
    ref
  ) => {
    const reduceMotion = useReducedMotion();
    const isDisabled = disabled || loading;

    const sizeClasses = {
      sm: "h-8 px-3 text-xs rounded-lg gap-2",
      default: "h-9 px-3.5 text-xs sm:text-[13px] rounded-xl gap-2",
      lg: "h-10.5 px-4.5 text-sm rounded-xl gap-2.5 font-semibold",
    };

    const iconSizes = {
      sm: "w-3.5 h-3.5",
      default: "w-4 h-4",
      lg: "w-4.5 h-4.5",
    };

    const podSizes = {
      sm: "w-4.5 h-4.5 rounded-md",
      default: "w-5 h-5 rounded-md",
      lg: "w-5.5 h-5.5 rounded-lg",
    };

    return (
      <motion.button
        ref={ref}
        type="button"
        disabled={isDisabled}
        onClick={onClick}
        whileHover={!isDisabled && !reduceMotion ? { scale: 1.02 } : undefined}
        whileTap={!isDisabled && !reduceMotion ? { scale: 0.97 } : undefined}
        transition={{ type: "spring", stiffness: 400, damping: 25 }}
        className={cn(
          "group relative inline-flex items-center justify-center font-medium select-none cursor-pointer overflow-hidden isolate focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 focus-visible:ring-offset-2 focus-visible:ring-offset-background transition-all duration-200",
          sizeClasses[size],
          !isDisabled
            ? "bg-gradient-to-b from-emerald-500/20 via-emerald-600/15 to-emerald-950/50 hover:from-emerald-500/30 hover:via-emerald-600/20 hover:to-emerald-950/70 text-emerald-100 hover:text-white border border-emerald-400/40 hover:border-emerald-400/70 shadow-[0_1px_3px_rgba(0,0,0,0.5),inset_0_1px_0_rgba(255,255,255,0.18),0_0_18px_-3px_rgba(16,185,129,0.3)] hover:shadow-[0_1px_3px_rgba(0,0,0,0.5),inset_0_1px_0_rgba(255,255,255,0.25),0_0_24px_-2px_rgba(16,185,129,0.45)]"
            : "bg-white/[0.03] text-zinc-500 border border-white/5 cursor-not-allowed opacity-50 shadow-none",
          className
        )}
        {...(props as any)}
      >
        {/* Magic UI Shimmer Sweep Ray */}
        {!isDisabled && !reduceMotion && (
          <span className="pointer-events-none absolute -inset-full w-[200%] h-[200%] -skew-x-12 bg-gradient-to-r from-transparent via-white/[0.12] to-transparent group-hover:translate-x-full transition-transform duration-700 ease-out" />
        )}

        {/* Ambient Emerald Aura */}
        {!isDisabled && (
          <span className="pointer-events-none absolute inset-0 bg-emerald-500/10 blur-sm opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
        )}

        {/* Integrated Icon Pod */}
        <span
          className={cn(
            "relative z-10 flex items-center justify-center shrink-0 transition-all duration-200",
            podSizes[size],
            !isDisabled
              ? "bg-emerald-400/20 text-emerald-300 group-hover:bg-emerald-400 group-hover:text-zinc-950"
              : "bg-white/5 text-zinc-600"
          )}
        >
          {loading ? (
            <Loader2 className={cn("animate-spin", iconSizes[size])} />
          ) : completed ? (
            <Check className={cn("stroke-[2.5]", iconSizes[size])} />
          ) : (
            <CheckCheck className={cn("stroke-[2.4] transition-transform duration-200 group-hover:scale-110", iconSizes[size])} />
          )}
        </span>

        {/* Label */}
        <span className="relative z-10 font-sans tracking-tight font-semibold">
          {loading ? "Processing..." : completed ? "All Approved" : label}
        </span>

        {/* Clean Integrated Counter Badge */}
        {!isDisabled && count !== undefined && count > 0 && (
          <span className="relative z-10 inline-flex items-center justify-center px-1.5 py-0.5 rounded-md text-[11px] font-mono font-bold bg-emerald-400/20 text-emerald-200 group-hover:bg-emerald-400 group-hover:text-zinc-950 transition-colors shadow-2xs">
            +{count}
          </span>
        )}
      </motion.button>
    );
  }
);

SaveAllButton.displayName = "SaveAllButton";

// ─────────────────────────────────────────────────────────────────────────────
// 2. ANIMATED DELETE BUTTON (WITH TRASH-LID MICRO-ANIMATION & TWO-STEP CONFIRM)
// Inspired by: Uiverse + 21st.dev + Mobbin Linear Standard
// ─────────────────────────────────────────────────────────────────────────────

export interface AnimatedDeleteButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  onDelete?: () => void | Promise<void>;
  label?: string;
  confirmLabel?: string;
  confirmRequired?: boolean;
  size?: "xs" | "sm" | "default" | "lg" | "icon";
  variant?: "danger" | "ghost" | "outline" | "solid";
  count?: number;
  loading?: boolean;
}

export const AnimatedDeleteButton = React.forwardRef<HTMLButtonElement, AnimatedDeleteButtonProps>(
  (
    {
      onDelete,
      label = "Delete",
      confirmLabel = "Confirm Delete?",
      confirmRequired = false,
      size = "default",
      variant = "danger",
      count,
      loading = false,
      disabled = false,
      className,
      onClick,
      ...props
    },
    ref
  ) => {
    const [isConfirming, setIsConfirming] = React.useState(false);
    const timeoutRef = React.useRef<NodeJS.Timeout | null>(null);
    const reduceMotion = useReducedMotion();
    const isDisabled = disabled || loading;

    // Auto-reset confirmation state after 3.5 seconds
    const startConfirmTimeout = React.useCallback(() => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      timeoutRef.current = setTimeout(() => {
        setIsConfirming(false);
      }, 3500);
    }, []);

    React.useEffect(() => {
      return () => {
        if (timeoutRef.current) clearTimeout(timeoutRef.current);
      };
    }, []);

    const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
      if (confirmRequired && !isConfirming) {
        e.preventDefault();
        e.stopPropagation();
        setIsConfirming(true);
        startConfirmTimeout();
        return;
      }

      setIsConfirming(false);
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      if (onDelete) {
        onDelete();
      } else if (onClick) {
        onClick(e);
      }
    };

    const sizeClasses = {
      xs: "h-7 px-2.5 text-[11px] rounded-lg gap-1.5",
      sm: "h-8 px-3 text-xs rounded-lg gap-2",
      default: "h-9 px-3.5 text-xs sm:text-[13px] rounded-xl gap-2",
      lg: "h-10.5 px-4.5 text-sm rounded-xl gap-2.5 font-semibold",
      icon: "h-9 w-9 p-0 rounded-xl justify-center",
    };

    const variantClasses = {
      danger: isConfirming
        ? "bg-rose-600 text-white border-rose-400 shadow-[0_0_25px_rgba(244,63,94,0.5),inset_0_1px_0_rgba(255,255,255,0.3)]"
        : "bg-gradient-to-b from-zinc-900/90 via-zinc-900/80 to-zinc-950/95 hover:from-rose-950/30 hover:via-zinc-900/90 hover:to-zinc-950/95 text-zinc-200 hover:text-rose-100 border-white/12 hover:border-rose-500/40 shadow-[0_1px_3px_rgba(0,0,0,0.5),inset_0_1px_0_rgba(255,255,255,0.08)] hover:shadow-[0_0_18px_-3px_rgba(244,63,94,0.3)]",
      ghost: isConfirming
        ? "bg-rose-600 text-white border border-rose-500"
        : "bg-transparent hover:bg-rose-500/10 text-muted-foreground hover:text-rose-200 border-transparent",
      outline: isConfirming
        ? "bg-rose-600 text-white border-rose-500 shadow-sm"
        : "border border-white/12 bg-zinc-900/60 hover:bg-rose-500/10 text-zinc-200 hover:text-rose-200 hover:border-rose-500/35",
      solid: isConfirming
        ? "bg-rose-600 text-white border-rose-400"
        : "bg-rose-500/15 hover:bg-rose-500 text-rose-300 hover:text-white border border-rose-500/30 hover:border-rose-400 shadow-2xs",
    };

    return (
      <motion.button
        ref={ref}
        type="button"
        disabled={isDisabled}
        onClick={handleClick}
        whileHover={!isDisabled && !reduceMotion ? { scale: 1.02 } : undefined}
        whileTap={!isDisabled && !reduceMotion ? { scale: 0.97 } : undefined}
        transition={{ type: "spring", stiffness: 400, damping: 25 }}
        className={cn(
          "group relative inline-flex items-center justify-center font-medium select-none cursor-pointer overflow-hidden isolate border transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-400 focus-visible:ring-offset-2 focus-visible:ring-offset-background",
          sizeClasses[size],
          variantClasses[variant],
          isDisabled && "opacity-50 cursor-not-allowed pointer-events-none",
          className
        )}
        {...(props as any)}
      >
        {/* Magic UI Shimmer Sweep Ray on Danger Hover */}
        {!isDisabled && !reduceMotion && variant === "danger" && !isConfirming && (
          <span className="pointer-events-none absolute -inset-full w-[200%] h-[200%] -skew-x-12 bg-gradient-to-r from-transparent via-rose-500/[0.08] to-transparent group-hover:translate-x-full transition-transform duration-700 ease-out" />
        )}

        <AnimatePresence mode="wait">
          {isConfirming ? (
            <motion.span
              key="confirm-state"
              initial={{ opacity: 0, y: 3 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -3 }}
              transition={{ duration: 0.15 }}
              className="inline-flex items-center gap-1.5 font-semibold text-white tracking-tight"
            >
              <AlertTriangle className="w-3.5 h-3.5 text-white animate-bounce shrink-0" />
              <span>{confirmLabel}</span>
            </motion.span>
          ) : (
            <motion.span
              key="normal-state"
              initial={{ opacity: 0, y: -3 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 3 }}
              transition={{ duration: 0.15 }}
              className="inline-flex items-center gap-2"
            >
              {/* Uiverse Animated Trash Pod with Kinetic Lid */}
              <span
                className={cn(
                  "relative flex items-center justify-center shrink-0 transition-colors",
                  size !== "icon" && "w-5 h-5 rounded-md bg-white/[0.06] group-hover:bg-rose-500/20 text-zinc-300 group-hover:text-rose-200"
                )}
              >
                {loading ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin text-rose-400" />
                ) : (
                  <div className="relative w-3.5 h-3.5 flex flex-col items-center justify-center">
                    <svg
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="w-3.5 h-3.5 overflow-visible"
                    >
                      {/* Trash Lid: Rotates up and left on hover */}
                      <path
                        d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"
                        className="transition-transform duration-200 ease-out origin-bottom-left group-hover:-translate-y-0.5 group-hover:-rotate-[28deg]"
                      />
                      {/* Trash Body */}
                      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" />
                      <line x1="10" y1="11" x2="10" y2="17" />
                      <line x1="14" y1="11" x2="14" y2="17" />
                    </svg>
                  </div>
                )}
              </span>

              {size !== "icon" && (
                <span className="font-sans tracking-tight font-semibold">
                  {label}
                </span>
              )}

              {count !== undefined && count > 0 && (
                <span className="inline-flex items-center justify-center px-1.5 py-0.5 rounded-md text-[11px] font-mono font-bold bg-white/[0.08] group-hover:bg-rose-500/20 text-zinc-400 group-hover:text-rose-200 transition-colors">
                  {count}
                </span>
              )}
            </motion.span>
          )}
        </AnimatePresence>
      </motion.button>
    );
  }
);

AnimatedDeleteButton.displayName = "AnimatedDeleteButton";

// ─────────────────────────────────────────────────────────────────────────────
// 3. TACTILE ROW MICRO-ACTION BUTTONS (APPROVE & REJECT)
// Inspired by: Mobbin (Linear / Raycast issue triaging)
// ─────────────────────────────────────────────────────────────────────────────

export interface TactileRowActionProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  actionType: "approve" | "reject";
  label?: string;
  isBusy?: boolean;
}

export const TactileRowAction = React.forwardRef<HTMLButtonElement, TactileRowActionProps>(
  ({ actionType, label, isBusy = false, disabled = false, className, onClick, ...props }, ref) => {
    const isApprove = actionType === "approve";
    const defaultLabel = isApprove ? "Approve" : "Reject";
    const displayLabel = label || defaultLabel;

    return (
      <motion.button
        ref={ref}
        type="button"
        disabled={disabled || isBusy}
        onClick={onClick}
        whileHover={!disabled && !isBusy ? { scale: 1.04 } : undefined}
        whileTap={!disabled && !isBusy ? { scale: 0.95 } : undefined}
        transition={{ type: "spring", stiffness: 450, damping: 25 }}
        className={cn(
          "group/row-btn inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-semibold tracking-tight transition-all duration-150 cursor-pointer select-none border",
          isApprove
            ? "bg-emerald-500/15 hover:bg-emerald-500 text-emerald-300 hover:text-zinc-950 border-emerald-500/35 hover:border-emerald-400 shadow-[0_1px_3px_rgba(0,0,0,0.4),inset_0_1px_0_rgba(255,255,255,0.18)]"
            : "bg-zinc-900/90 hover:bg-rose-500 text-zinc-300 hover:text-white border-white/10 hover:border-rose-400 shadow-[0_1px_3px_rgba(0,0,0,0.4),inset_0_1px_0_rgba(255,255,255,0.06)]",
          (disabled || isBusy) && "opacity-50 pointer-events-none cursor-not-allowed",
          className
        )}
        {...(props as any)}
      >
        <span
          className={cn(
            "w-4 h-4 rounded-md flex items-center justify-center transition-colors shrink-0",
            isApprove
              ? "bg-emerald-500/25 group-hover/row-btn:bg-zinc-950/20 text-emerald-300 group-hover/row-btn:text-zinc-950"
              : "bg-white/[0.06] group-hover/row-btn:bg-zinc-950/20 text-zinc-400 group-hover/row-btn:text-white"
          )}
        >
          {isBusy ? (
            <Loader2 className="w-3 h-3 animate-spin" />
          ) : isApprove ? (
            <Check className="w-3 h-3 stroke-[2.6]" />
          ) : (
            <X className="w-3 h-3 stroke-[2.6]" />
          )}
        </span>
        <span className="font-sans font-semibold">{displayLabel}</span>
      </motion.button>
    );
  }
);

TactileRowAction.displayName = "TactileRowAction";

// ─────────────────────────────────────────────────────────────────────────────
// 4. SHIMMER ACTION BUTTON (PRIMARY HERO / LAUNCH / EXPORT)
// Inspired by: Magic UI + 21st.dev Shimmer Button
// ─────────────────────────────────────────────────────────────────────────────

export interface ShimmerActionButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  shimmerColor?: string;
  icon?: React.ReactNode;
  trailingIcon?: React.ReactNode;
  loading?: boolean;
}

export const ShimmerActionButton = React.forwardRef<HTMLButtonElement, ShimmerActionButtonProps>(
  (
    {
      children,
      icon,
      trailingIcon,
      loading = false,
      disabled = false,
      className,
      onClick,
      ...props
    },
    ref
  ) => {
    const reduceMotion = useReducedMotion();
    const isDisabled = disabled || loading;

    return (
      <motion.button
        ref={ref}
        type="button"
        disabled={isDisabled}
        onClick={onClick}
        whileHover={!isDisabled && !reduceMotion ? { scale: 1.02 } : undefined}
        whileTap={!isDisabled && !reduceMotion ? { scale: 0.98 } : undefined}
        transition={{ type: "spring", stiffness: 400, damping: 25 }}
        className={cn(
          "group relative inline-flex items-center justify-center gap-2.5 px-5 py-2.5 rounded-xl text-sm font-bold text-zinc-950 select-none cursor-pointer overflow-hidden isolate shadow-[0_4px_20px_-4px_rgba(255,255,255,0.3),inset_0_1px_1px_rgba(255,255,255,0.9)] bg-white hover:bg-zinc-100 border border-white transition-all duration-200",
          isDisabled && "opacity-50 cursor-not-allowed pointer-events-none",
          className
        )}
        {...(props as any)}
      >
        {/* Shimmer Light Reflection Sweep */}
        {!isDisabled && !reduceMotion && (
          <span className="pointer-events-none absolute -inset-full w-[200%] h-[200%] -skew-x-12 bg-gradient-to-r from-transparent via-black/10 to-transparent group-hover:translate-x-full transition-transform duration-1000 ease-out" />
        )}

        {/* Leading Icon Pod */}
        {loading ? (
          <Loader2 className="w-4 h-4 animate-spin text-zinc-950 shrink-0" />
        ) : icon ? (
          <span className="shrink-0">{icon}</span>
        ) : null}

        {/* Text Children */}
        <span className="relative z-10 font-sans tracking-tight">{children}</span>

        {/* Trailing Icon Pod (Button-in-Button Apple/Linear pattern) */}
        {trailingIcon ? (
          <span className="relative z-10 w-5.5 h-5.5 rounded-md bg-black/10 flex items-center justify-center transition-transform group-hover:translate-x-0.5 shrink-0">
            {trailingIcon}
          </span>
        ) : (
          <span className="relative z-10 w-5.5 h-5.5 rounded-md bg-black/10 flex items-center justify-center transition-transform group-hover:translate-x-0.5 shrink-0">
            <ChevronRight className="w-3.5 h-3.5 text-zinc-950 stroke-[2.5]" />
          </span>
        )}
      </motion.button>
    );
  }
);

ShimmerActionButton.displayName = "ShimmerActionButton";

// ─────────────────────────────────────────────────────────────────────────────
// 5. SAVE / BOOKMARK KPI BUTTON
// Inspired by: 21st.dev + Mobbin (Verified KPI / Bookmark patterns)
// ─────────────────────────────────────────────────────────────────────────────

export interface SaveKpiButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  isSaved?: boolean;
  onToggleSave?: () => void;
  savedLabel?: string;
  defaultLabel?: string;
}

export const SaveKpiButton = React.forwardRef<HTMLButtonElement, SaveKpiButtonProps>(
  (
    {
      isSaved = false,
      onToggleSave,
      savedLabel = "Saved as KPI",
      defaultLabel = "Save as Golden KPI",
      disabled = false,
      className,
      ...props
    },
    ref
  ) => {
    return (
      <motion.button
        ref={ref}
        type="button"
        disabled={disabled}
        onClick={onToggleSave}
        whileHover={!disabled ? { scale: 1.02 } : undefined}
        whileTap={!disabled ? { scale: 0.96 } : undefined}
        transition={{ type: "spring", stiffness: 400, damping: 25 }}
        className={cn(
          "group inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold tracking-tight transition-all duration-200 cursor-pointer select-none border",
          isSaved
            ? "bg-emerald-500/20 text-emerald-200 border-emerald-500/40 shadow-[0_0_15px_-3px_rgba(16,185,129,0.3)]"
            : "bg-zinc-900/80 hover:bg-emerald-500/15 text-zinc-300 hover:text-emerald-200 border border-white/10 hover:border-emerald-500/40 shadow-[0_1px_3px_rgba(0,0,0,0.4),inset_0_1px_0_rgba(255,255,255,0.06)]",
          disabled && "opacity-50 cursor-not-allowed pointer-events-none",
          className
        )}
        {...(props as any)}
      >
        <span
          className={cn(
            "w-4.5 h-4.5 rounded-md flex items-center justify-center transition-colors shrink-0",
            isSaved ? "bg-emerald-500 text-zinc-950" : "bg-white/[0.06] group-hover:bg-emerald-500/20 text-emerald-400"
          )}
        >
          {isSaved ? <BookmarkCheck className="w-3.5 h-3.5 stroke-[2.5]" /> : <Bookmark className="w-3.5 h-3.5 stroke-[2.2]" />}
        </span>
        <span className="font-sans font-semibold">{isSaved ? savedLabel : defaultLabel}</span>
      </motion.button>
    );
  }
);

SaveKpiButton.displayName = "SaveKpiButton";
