import {
  forwardRef,
  type ComponentPropsWithoutRef,
  type ReactNode,
} from "react";

import { cn } from "@/lib/utils";

export type SheenPillButtonProps = Readonly<
  {
    children: ReactNode;
    highlight?: number;
  } & ComponentPropsWithoutRef<"button">
>;

const SHEEN_SHELL =
  "group relative inline-flex shrink-0 cursor-pointer items-center justify-center gap-2.5 overflow-hidden rounded-full border border-zinc-950/20 dark:border-white/20 bg-zinc-950 dark:bg-white text-white dark:text-zinc-950 font-sans font-bold text-sm tracking-tight px-7 py-3 transition-[transform,background-color,box-shadow,filter] duration-200 ease-out hover:scale-[1.02] active:scale-95 disabled:pointer-events-none disabled:opacity-50 select-none shadow-[0_4px_16px_rgba(0,0,0,0.15)] dark:shadow-[0_4px_20px_rgba(255,255,255,0.22)]";

export const SheenPillButton = forwardRef<
  HTMLButtonElement,
  SheenPillButtonProps
>(
  (
    {
      children,
      className,
      highlight = 18,
      type = "button",
      disabled,
      ...props
    },
    ref,
  ) => {
    return (
      <button
        ref={ref}
        type={type}
        disabled={disabled}
        data-slot="sheen-pill-button"
        className={cn(SHEEN_SHELL, className)}
        {...props}
      >
        {/* Specular sheen beam layer */}
        <span
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 z-10 opacity-40 dark:opacity-70 blur-[6px] transition-[opacity,transform] duration-300 ease-out group-hover:scale-105 group-hover:opacity-80"
          style={{
            borderRadius: 9999,
            background: `linear-gradient(45deg, rgba(255,255,255,0.7) 0%, transparent ${highlight}%, transparent calc(100% - ${highlight}%), rgba(255,255,255,0.7) 100%)`,
          }}
        />

        {/* Dynamic gloss reflection */}
        <span
          aria-hidden="true"
          className="pointer-events-none absolute -inset-full z-15 bg-gradient-to-r from-transparent via-white/20 dark:via-zinc-950/10 to-transparent -translate-x-full transition-transform duration-700 ease-in-out group-hover:translate-x-full"
        />

        <span className="relative z-20 inline-flex items-center gap-2">
          {children}
        </span>
      </button>
    );
  },
);

SheenPillButton.displayName = "SheenPillButton";

