import {
  forwardRef,
  type ComponentPropsWithoutRef,
  type ReactNode,
} from "react";

import { cn } from "@/lib/utils";

export type DepthOutlineButtonSize = "sm" | "md" | "lg";

export type DepthOutlineButtonProps = Readonly<
  {
    children: ReactNode;
    size?: DepthOutlineButtonSize;
  } & ComponentPropsWithoutRef<"button">
>;

const SIZE: Record<DepthOutlineButtonSize, string> = {
  sm: "h-9 gap-1.5 rounded-lg px-3.5 text-xs tracking-wide",
  md: "h-10 gap-2 rounded-xl px-4 text-sm tracking-wide",
  lg: "h-12 gap-2 rounded-xl px-5 text-sm tracking-wide",
};

// Resting: dual-edge (hairline + float) with a soft top lip, tuned for both light and dark mode
const REST =
  "shadow-[0_0_0_1px_rgba(0,0,0,0.08),0_1px_1px_rgba(0,0,0,0.04),0_4px_10px_rgba(0,0,0,0.05),0_12px_28px_rgba(0,0,0,0.05),inset_0_1px_1px_rgba(255,255,255,0.8),inset_0_-1px_2px_rgba(0,0,0,0.04)] dark:shadow-[0_0_0_1px_rgba(255,255,255,0.12),0_1px_1px_rgba(0,0,0,0.4),0_4px_10px_rgba(0,0,0,0.3),inset_0_1px_1px_rgba(255,255,255,0.15)]";

const HOVER =
  "hover:bg-slate-50 hover:text-slate-950 dark:hover:bg-white/[0.08] dark:hover:text-white hover:shadow-[0_0_0_1px_rgba(0,0,0,0.12),0_2px_4px_rgba(0,0,0,0.05),0_8px_16px_rgba(0,0,0,0.07),0_20px_40px_rgba(0,0,0,0.06),inset_0_1px_1px_rgba(255,255,255,0.9),inset_0_-1px_2px_rgba(0,0,0,0.05)] dark:hover:shadow-[0_0_0_1px_rgba(255,255,255,0.2),0_2px_4px_rgba(0,0,0,0.4),0_8px_16px_rgba(0,0,0,0.3),inset_0_1px_1px_rgba(255,255,255,0.25)]";

const PRESSED =
  "active:bg-slate-100 active:shadow-[0_0_0_1px_rgba(0,0,0,0.1),0_1px_2px_rgba(0,0,0,0.04),inset_0_2px_4px_rgba(0,0,0,0.08),inset_0_1px_0_rgba(255,255,255,0.6)] dark:active:bg-white/[0.12]";

// DepthOutlineButton - adapted from opensourceui with full dark/light theme support
export const DepthOutlineButton = forwardRef<
  HTMLButtonElement,
  DepthOutlineButtonProps
>(
  (
    { className, children, size = "md", type = "button", disabled, ...props },
    ref,
  ) => {
    return (
      <button
        ref={ref}
        type={type}
        disabled={disabled}
        data-slot="depth-outline-button"
        data-size={size}
        className={cn(
          "inline-flex cursor-pointer items-center justify-center bg-white dark:bg-[#12141a] font-sans font-semibold text-slate-800 dark:text-zinc-200 outline-none select-none",
          "transition-[background-color,box-shadow,color,transform] duration-200 ease-[cubic-bezier(0.32,0.72,0,1)] motion-reduce:transition-none",
          "active:translate-y-px motion-reduce:active:translate-y-0",
          "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-500",
          "disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-40",
          REST,
          HOVER,
          PRESSED,
          SIZE[size],
          className,
        )}
        {...props}
      >
        {children}
      </button>
    );
  },
);

DepthOutlineButton.displayName = "DepthOutlineButton";

