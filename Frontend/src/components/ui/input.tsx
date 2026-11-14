import * as React from "react";

import { cn } from "@/lib/utils";

const Input = React.forwardRef<HTMLInputElement, React.ComponentProps<"input">>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={cn(
          "flex h-10 w-full rounded-xl border border-stone-300 bg-[#FAF8F5] px-3.5 py-2 text-sm text-stone-900 ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-stone-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#800000]/30 focus-visible:border-[#800000] dark:border-white/10 dark:bg-[#0e1017] dark:text-foreground dark:placeholder:text-zinc-500 dark:focus-visible:ring-white/20 dark:focus-visible:border-white/40 disabled:cursor-not-allowed disabled:opacity-50 transition-colors",
          className,
        )}
        ref={ref}
        {...props}
      />
    );
  },
);
Input.displayName = "Input";

export { Input };
