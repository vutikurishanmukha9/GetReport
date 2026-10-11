import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { motion, AnimatePresence, useReducedMotion, type HTMLMotionProps } from "framer-motion";
import { Loader2 } from "lucide-react";

import { cn } from "@/lib/utils";
import { SPRING_PRESS, EASE_OUT } from "@/lib/ease";
import { useHoverCapable } from "@/lib/hooks/use-hover-capable";

const buttonVariants = cva(
  "relative inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-xl text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 select-none cursor-pointer overflow-hidden isolate transform-gpu [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default:
          "bg-white text-zinc-950 font-semibold hover:bg-zinc-100 shadow-[0_4px_14px_-2px_rgba(0,0,0,0.5),inset_0_1px_0_rgba(255,255,255,0.8)] border border-white/90",
        primary:
          "bg-primary text-primary-foreground font-semibold hover:bg-primary/90 shadow-md shadow-primary/20 border border-primary/30",
        destructive:
          "bg-destructive text-destructive-foreground font-semibold hover:bg-destructive/90 shadow-sm border border-destructive/40",
        outline:
          "border border-white/10 bg-white/[0.04] text-foreground hover:bg-white/[0.09] hover:border-white/20 hover:text-foreground shadow-2xs backdrop-blur-sm",
        secondary:
          "bg-secondary text-secondary-foreground font-medium hover:bg-secondary/80 border border-border/70 shadow-2xs",
        ghost:
          "hover:bg-white/[0.06] hover:text-foreground text-muted-foreground",
        link:
          "text-foreground underline-offset-4 hover:underline p-0 h-auto font-normal",
        // Dedicated High-End Semantic Action Variants
        save:
          "bg-emerald-600 hover:bg-emerald-500 text-white shadow-sm shadow-emerald-950/50 border border-emerald-400/40 ring-1 ring-white/10 font-semibold",
        saveAll:
          "bg-emerald-600 hover:bg-emerald-500 text-white shadow-md shadow-emerald-950/60 border border-emerald-400/50 ring-1 ring-white/15 font-bold",
        delete:
          "border border-border/80 bg-background/50 text-muted-foreground hover:bg-destructive/10 hover:text-destructive hover:border-destructive/40 shadow-2xs font-medium transition-colors",
        deleteAll:
          "bg-rose-950/30 text-rose-300 border border-rose-500/30 hover:bg-rose-600 hover:text-white hover:border-rose-600 shadow-2xs font-bold transition-all duration-150",
        launch:
          "bg-white text-black hover:bg-zinc-100 shadow-[0_4px_20px_-2px_rgba(255,255,255,0.25),inset_0_1px_0_rgba(255,255,255,0.9)] border border-white font-bold",
        glow:
          "border border-blue-500/40 bg-blue-950/25 text-blue-300 hover:bg-blue-500/20 hover:border-blue-500/60 shadow-[0_0_16px_-4px_rgba(59,130,246,0.3)] font-semibold",
      },
      size: {
        default: "h-9 px-4 py-2 text-xs sm:text-sm",
        xs: "h-7 rounded-md px-2.5 text-[11px] gap-1.5",
        sm: "h-8 rounded-lg px-3 text-xs gap-1.5",
        lg: "h-11 rounded-xl px-6 text-sm font-semibold gap-2.5",
        xl: "h-12 rounded-2xl px-7 text-base font-bold gap-3",
        icon: "h-9 w-9 rounded-xl p-0",
        "icon-sm": "h-7 w-7 rounded-lg p-0",
        "icon-lg": "h-11 w-11 rounded-xl p-0",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
);

type Ripple = { id: number; x: number; y: number; size: number };

export interface ButtonProps
  extends Omit<
      React.ButtonHTMLAttributes<HTMLButtonElement>,
      "onAnimationStart" | "onDrag" | "onDragStart" | "onDragEnd"
    >,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
  pressScale?: number;
  ripple?: boolean;
  loading?: boolean;
  isLoading?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant,
      size,
      asChild = false,
      pressScale = 0.97,
      ripple = true,
      loading = false,
      isLoading = false,
      disabled,
      children,
      onPointerDown,
      ...props
    },
    ref,
  ) => {
    const isBusy = loading || isLoading;
    const isDisabled = disabled || isBusy;
    const reduceMotion = useReducedMotion();
    const canHover = useHoverCapable();
    const [ripples, setRipples] = React.useState<Ripple[]>([]);
    const nextId = React.useRef(0);

    // Ripple wave generator on click
    const handlePointerDown = React.useCallback(
      (event: React.PointerEvent<HTMLButtonElement>) => {
        if (ripple && !reduceMotion && !isDisabled && !asChild) {
          const rect = event.currentTarget.getBoundingClientRect();
          const rippleSize = Math.max(rect.width, rect.height) * 2;
          const id = nextId.current++;
          setRipples((prev) => [
            ...prev,
            {
              id,
              x: event.clientX - rect.left,
              y: event.clientY - rect.top,
              size: rippleSize,
            },
          ]);
        }
        onPointerDown?.(event);
      },
      [ripple, reduceMotion, isDisabled, asChild, onPointerDown],
    );

    // If asChild is true, delegate to Radix Slot (retains Link / component wrapping)
    if (asChild) {
      return (
        <Slot
          className={cn(buttonVariants({ variant, size, className }))}
          ref={ref}
          aria-busy={isBusy ? "true" : undefined}
          {...(props as React.HTMLAttributes<HTMLElement>)}
        >
          {children}
        </Slot>
      );
    }

    // Motion & haptic feedback calculations
    const shouldTapAnimate = !reduceMotion && !isDisabled;
    const shouldHoverAnimate = !reduceMotion && canHover && !isDisabled;

    // Filter motion-incompatible DOM attributes if any
    const motionProps = props as unknown as HTMLMotionProps<"button">;

    return (
      <motion.button
        ref={ref}
        whileTap={shouldTapAnimate ? { scale: pressScale } : undefined}
        whileHover={shouldHoverAnimate ? { scale: 1.01 } : undefined}
        transition={SPRING_PRESS}
        onPointerDown={handlePointerDown}
        disabled={isDisabled}
        aria-busy={isBusy ? "true" : undefined}
        className={cn(buttonVariants({ variant, size, className }))}
        {...motionProps}
      >
        {/* Subtle radial ripple wave effect */}
        <span className="pointer-events-none absolute inset-0 overflow-hidden rounded-[inherit] z-0">
          <AnimatePresence>
            {ripple && !reduceMotion && ripples.map((r) => (
              <motion.span
                key={r.id}
                className="absolute rounded-full bg-current opacity-25"
                style={{
                  left: r.x,
                  top: r.y,
                  width: r.size,
                  height: r.size,
                  x: "-50%",
                  y: "-50%",
                }}
                initial={{ scale: 0.05, opacity: 0.25 }}
                animate={{ scale: 1, opacity: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 1.0, ease: EASE_OUT }}
                onAnimationComplete={() =>
                  setRipples((prev) => prev.filter((x) => x.id !== r.id))
                }
              />
            ))}
          </AnimatePresence>
        </span>

        {/* Content with optional stateful spinner */}
        <span className="relative z-10 inline-flex items-center justify-center gap-2">
          {isBusy && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
          {children}
        </span>
      </motion.button>
    );
  },
);

Button.displayName = "Button";

export { Button, buttonVariants };
