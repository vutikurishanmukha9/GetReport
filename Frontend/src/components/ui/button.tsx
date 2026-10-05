import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-xl text-sm font-medium ring-offset-background transition-all duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 select-none cursor-pointer [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary/90 shadow-sm active:scale-[0.98]",
        destructive: "bg-destructive text-destructive-foreground hover:bg-destructive/90 shadow-sm active:scale-[0.98]",
        outline: "border border-border/80 bg-white text-foreground hover:bg-slate-50 hover:border-slate-300 shadow-2xs active:scale-[0.98]",
        secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80 shadow-2xs active:scale-[0.98]",
        ghost: "hover:bg-accent hover:text-accent-foreground active:scale-[0.98]",
        link: "text-primary underline-offset-4 hover:underline",
        // Dedicated High-End Semantic Action Variants
        save: "bg-emerald-600 hover:bg-emerald-500 text-white shadow-sm shadow-emerald-600/25 border border-emerald-400/40 ring-1 ring-white/20 active:scale-[0.98] font-semibold",
        saveAll: "bg-gradient-to-r from-emerald-600 via-teal-600 to-emerald-700 hover:from-emerald-500 hover:to-teal-500 text-white shadow-md shadow-emerald-600/30 border border-emerald-400/50 ring-1 ring-white/25 active:scale-[0.98] font-bold",
        delete: "bg-white text-slate-700 border border-slate-200/90 hover:bg-rose-50 hover:text-rose-700 hover:border-rose-300 shadow-2xs hover:shadow-xs active:scale-[0.98] font-semibold transition-all duration-150",
        deleteAll: "bg-white text-rose-700 border border-rose-200 hover:bg-rose-600 hover:text-white hover:border-rose-600 shadow-2xs hover:shadow-sm hover:shadow-rose-600/20 active:scale-[0.98] font-bold transition-all duration-150",
        launch: "bg-gradient-to-r from-violet-600 via-purple-600 to-indigo-600 hover:from-violet-500 hover:via-purple-500 hover:to-indigo-500 text-white shadow-[0_4px_14px_-2px_rgba(124,58,237,0.38),inset_0_1px_1px_rgba(255,255,255,0.3)] border border-violet-400/40 ring-1 ring-white/20 active:scale-[0.98] font-bold",
      },
      size: {
        default: "h-10 px-4 py-2 text-xs sm:text-sm",
        xs: "h-7.5 rounded-lg px-2.5 text-[11px]",
        sm: "h-8.5 rounded-lg px-3 text-xs",
        lg: "h-11 sm:h-12 rounded-2xl px-6 sm:px-8 text-sm",
        icon: "h-9 w-9 rounded-xl",
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
