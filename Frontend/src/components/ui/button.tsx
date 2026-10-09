import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-medium ring-offset-background transition-all duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 select-none cursor-pointer [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default: "bg-white text-zinc-950 font-semibold hover:bg-zinc-100 shadow-[0_4px_14px_-2px_rgba(0,0,0,0.6),inset_0_1px_0_rgba(255,255,255,0.6)] border border-white/90 active:scale-[0.98]",
        destructive: "bg-destructive text-destructive-foreground hover:bg-destructive/90 shadow-sm active:scale-[0.98]",
        outline: "border border-border/80 bg-card/60 text-foreground hover:bg-secondary hover:text-foreground shadow-2xs active:scale-[0.98]",
        secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80 border border-border/60 shadow-2xs active:scale-[0.98]",
        ghost: "hover:bg-accent hover:text-accent-foreground text-muted-foreground active:scale-[0.98]",
        link: "text-foreground underline-offset-4 hover:underline",
        // Dedicated High-End Semantic Action Variants
        save: "bg-emerald-600 hover:bg-emerald-500 text-white shadow-sm shadow-emerald-900/40 border border-emerald-400/40 ring-1 ring-white/10 active:scale-[0.98] font-semibold",
        saveAll: "bg-emerald-600 hover:bg-emerald-500 text-white shadow-md shadow-emerald-900/50 border border-emerald-400/50 ring-1 ring-white/15 active:scale-[0.98] font-bold",
        delete: "border border-border/80 bg-background/50 text-muted-foreground hover:bg-destructive/10 hover:text-destructive hover:border-destructive/40 shadow-2xs active:scale-[0.98] font-medium transition-colors",
        deleteAll: "bg-rose-950/30 text-rose-300 border border-rose-500/30 hover:bg-rose-600 hover:text-white hover:border-rose-600 shadow-2xs active:scale-[0.98] font-bold transition-all duration-150",
        launch: "bg-white text-black hover:bg-slate-100 shadow-[0_4px_16px_-2px_rgba(255,255,255,0.2),inset_0_1px_0_rgba(255,255,255,0.8)] border border-white font-bold active:scale-[0.98]",
      },
      size: {
        default: "h-9 px-4 py-2 text-xs sm:text-sm",
        xs: "h-7 rounded-md px-2.5 text-[11px]",
        sm: "h-8 rounded-md px-3 text-xs",
        lg: "h-10 sm:h-11 rounded-lg px-6 text-sm",
        icon: "h-9 w-9 rounded-md",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return <Comp className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />;
  },
);
Button.displayName = "Button";

export { Button, buttonVariants };
