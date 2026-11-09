import { forwardRef, type ComponentPropsWithoutRef, type ReactNode } from "react";
import { cn } from "@/lib/utils";
import { ShieldCheck, FileCheck2 } from "lucide-react";

export type InkStampCardProps = Readonly<
  {
    title?: string;
    reference?: string;
    auditedBy?: string;
    auditedOn?: string;
    stampLabel?: string;
    status?: string;
    metrics?: string;
    icon?: ReactNode;
  } & ComponentPropsWithoutRef<"div">
>;

export const InkStampCard = forwardRef<
  HTMLDivElement,
  InkStampCardProps
>(
  (
    {
      className,
      title = "Executive Dataset Audit",
      reference = "AUDIT-2026-X89",
      auditedBy = "Local Hygiene Engine",
      auditedOn = "Active Session",
      stampLabel = "CERTIFIED",
      status = "Clean & Boardroom Ready",
      metrics = "Zero Data Leakage • 100% Local",
      icon,
      ...props
    },
    ref,
  ) => {
    return (
      <div
        ref={ref}
        data-slot="ink-stamp-card"
        className={cn(
          "relative overflow-hidden rounded-2xl border border-slate-200 dark:border-white/10 bg-white dark:bg-[#0c0d12] p-5 font-sans shadow-sm dark:shadow-premium transition-all duration-200 hover:shadow-md",
          className,
        )}
        {...props}
      >
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="font-mono text-[10px] tracking-[0.2em] text-slate-400 dark:text-zinc-500 uppercase">
              {reference}
            </p>
            <h4 className="mt-1.5 text-base font-semibold text-slate-900 dark:text-white truncate">
              {title}
            </h4>
          </div>
          <div className="flex size-9 shrink-0 items-center justify-center rounded-xl bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
            {icon || <ShieldCheck className="w-4 h-4" />}
          </div>
        </div>

        <div className="mt-3.5 space-y-1.5 text-xs text-slate-600 dark:text-zinc-400">
          <p>
            <span className="text-slate-400 dark:text-zinc-500">Engine:</span> {auditedBy}
          </p>
          <p>
            <span className="text-slate-400 dark:text-zinc-500">Timestamp:</span> {auditedOn}
          </p>
          <p className="text-[11px] font-mono text-emerald-600 dark:text-emerald-400 font-medium">
            {metrics}
          </p>
        </div>

        <div className="mt-4 h-px bg-slate-100 dark:bg-white/5" />

        <div className="mt-3 flex items-center justify-between">
          <p className="text-xs font-medium text-slate-500 dark:text-zinc-400 flex items-center gap-1.5">
            <FileCheck2 className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
            {status}
          </p>
        </div>

        {/* Vintage Certification Stamp */}
        <div
          data-layer="ink-stamp"
          className="pointer-events-none absolute right-3 bottom-3 -rotate-12 rounded-lg border-2 border-emerald-600/70 dark:border-emerald-400/60 px-2.5 py-1 text-center bg-emerald-50/50 dark:bg-emerald-950/20 backdrop-blur-xs"
          aria-hidden
        >
          <p className="text-[10px] font-bold tracking-[0.22em] text-emerald-700 dark:text-emerald-400 uppercase">
            {stampLabel}
          </p>
        </div>
      </div>
    );
  },
);

InkStampCard.displayName = "InkStampCard";

