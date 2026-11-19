import * as React from "react";
import * as ToastPrimitives from "@radix-ui/react-toast";
import { cva, type VariantProps } from "class-variance-authority";
import { X } from "lucide-react";

import { cn } from "@/lib/utils";
import { useTheme } from "@/hooks/useTheme";

const ToastProvider = ToastPrimitives.Provider;

const ToastViewport = React.forwardRef<
  React.ElementRef<typeof ToastPrimitives.Viewport>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.Viewport>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.Viewport
    ref={ref}
    className={cn(
      "fixed top-0 z-[9999] flex max-h-screen w-full flex-col-reverse p-4 sm:bottom-0 sm:right-0 sm:top-auto sm:flex-col md:max-w-[420px]",
      className,
    )}
    {...props}
  />
));
ToastViewport.displayName = ToastPrimitives.Viewport.displayName;

const toastVariants = cva(
  "toast-root group pointer-events-auto relative flex w-full items-start justify-between space-x-3 overflow-hidden rounded-2xl p-4 pr-10 transition-all data-[swipe=cancel]:translate-x-0 data-[swipe=end]:translate-x-[var(--radix-toast-swipe-end-x)] data-[swipe=move]:translate-x-[var(--radix-toast-swipe-move-x)] data-[swipe=move]:transition-none data-[state=open]:animate-in data-[state=closed]:animate-out data-[swipe=end]:animate-out data-[state=closed]:fade-out-80 data-[state=closed]:slide-out-to-right-full data-[state=open]:slide-in-from-top-full data-[state=open]:sm:slide-in-from-bottom-full border shadow-xl bg-white dark:bg-[#18181b] opacity-100",
  {
    variants: {
      variant: {
        default: "border-stone-200 dark:border-white/10 text-stone-900 dark:text-zinc-100 bg-white dark:bg-[#18181b]",
        destructive: "toast-destructive border-rose-200 dark:border-rose-500/40 text-stone-900 dark:text-zinc-100 bg-white dark:bg-[#1c1316]",
        success: "toast-success border-emerald-200 dark:border-emerald-500/40 text-stone-900 dark:text-zinc-100 bg-white dark:bg-[#0f1b15]",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
);

const Toast = React.forwardRef<
  React.ElementRef<typeof ToastPrimitives.Root>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.Root> & VariantProps<typeof toastVariants>
>(({ className, variant, style, children, ...props }, ref) => {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  const isDestructive = variant === "destructive";
  const isSuccess = variant === "success";

  const resolvedBg = isDark
    ? isDestructive
      ? "#1c1316"
      : isSuccess
      ? "#0f1b15"
      : "#18181b"
    : "#ffffff";

  return (
    <ToastPrimitives.Root
      ref={ref}
      data-getreport-toast="true"
      data-variant={variant || "default"}
      style={{
        backgroundColor: resolvedBg,
        opacity: 1,
        ...style,
      }}
      className={cn(toastVariants({ variant }), className)}
      {...props}
    >
      {/* 100% Solid opaque backing plate: guarantees zero see-through even against dashed borders or grid dots */}
      <div
        className="absolute inset-0 -z-10 rounded-2xl pointer-events-none"
        style={{ backgroundColor: resolvedBg, opacity: 1 }}
        aria-hidden="true"
      />
      {children}
    </ToastPrimitives.Root>
  );
});
Toast.displayName = ToastPrimitives.Root.displayName;

const ToastAction = React.forwardRef<
  React.ElementRef<typeof ToastPrimitives.Action>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.Action>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.Action
    ref={ref}
    className={cn(
      "inline-flex h-8 shrink-0 items-center justify-center rounded-lg border border-stone-200 bg-stone-50 px-3 text-xs font-medium text-stone-900 transition-colors hover:bg-stone-100 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100 dark:hover:bg-zinc-700 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
      className,
    )}
    {...props}
  />
));
ToastAction.displayName = ToastPrimitives.Action.displayName;

const ToastClose = React.forwardRef<
  React.ElementRef<typeof ToastPrimitives.Close>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.Close>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.Close
    ref={ref}
    className={cn(
      "absolute right-2.5 top-2.5 rounded-lg p-1.5 text-stone-400 transition-colors hover:bg-stone-100 hover:text-stone-700 dark:text-zinc-400 dark:hover:bg-zinc-800 dark:hover:text-zinc-100 group-[.toast-destructive]:text-rose-400 group-[.toast-destructive]:hover:text-rose-700 group-[.toast-destructive]:hover:bg-rose-100 dark:group-[.toast-destructive]:text-rose-400 dark:group-[.toast-destructive]:hover:text-rose-100 dark:group-[.toast-destructive]:hover:bg-rose-950/60 focus:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring",
      className,
    )}
    toast-close=""
    {...props}
  >
    <X className="h-4 w-4" />
  </ToastPrimitives.Close>
));
ToastClose.displayName = ToastPrimitives.Close.displayName;

const ToastTitle = React.forwardRef<
  React.ElementRef<typeof ToastPrimitives.Title>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.Title>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.Title
    ref={ref}
    className={cn(
      "text-[13px] font-semibold tracking-tight text-stone-900 dark:text-zinc-100 group-[.toast-destructive]:text-rose-950 dark:group-[.toast-destructive]:text-rose-100 group-[.toast-success]:text-emerald-950 dark:group-[.toast-success]:text-emerald-100",
      className,
    )}
    {...props}
  />
));
ToastTitle.displayName = ToastPrimitives.Title.displayName;

const ToastDescription = React.forwardRef<
  React.ElementRef<typeof ToastPrimitives.Description>,
  React.ComponentPropsWithoutRef<typeof ToastPrimitives.Description>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.Description
    ref={ref}
    className={cn(
      "text-xs leading-relaxed text-stone-600 dark:text-zinc-300 group-[.toast-destructive]:text-rose-900 dark:group-[.toast-destructive]:text-rose-200 group-[.toast-success]:text-emerald-900 dark:group-[.toast-success]:text-emerald-200",
      className,
    )}
    {...props}
  />
));
ToastDescription.displayName = ToastPrimitives.Description.displayName;

type ToastProps = React.ComponentPropsWithoutRef<typeof Toast>;

type ToastActionElement = React.ReactElement<typeof ToastAction>;

export {
  type ToastProps,
  type ToastActionElement,
  ToastProvider,
  ToastViewport,
  Toast,
  ToastTitle,
  ToastDescription,
  ToastClose,
  ToastAction,
};
