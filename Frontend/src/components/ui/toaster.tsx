import { useToast } from "@/hooks/use-toast";
import { Toast, ToastClose, ToastDescription, ToastProvider, ToastTitle, ToastViewport } from "@/components/ui/toast";
import { AlertCircle, CheckCircle2, Info, Loader2 } from "lucide-react";

export function Toaster() {
  const { toasts } = useToast();

  return (
    <ToastProvider>
      {toasts.map(function ({ id, title, description, action, variant, ...props }) {
        const titleStr = typeof title === "string" ? title.toLowerCase() : "";
        const descStr = typeof description === "string" ? description.toLowerCase() : "";

        const isDestructive =
          variant === "destructive" ||
          titleStr.includes("fail") ||
          titleStr.includes("error") ||
          descStr.includes("fail") ||
          descStr.includes("error");

        const isSuccess =
          !isDestructive &&
          (variant === "success" ||
            titleStr.includes("complete") ||
            titleStr.includes("ready") ||
            titleStr.includes("success") ||
            titleStr.includes("downloaded") ||
            titleStr.includes("copied"));

        const isLoading =
          !isDestructive &&
          !isSuccess &&
          (titleStr.includes("upload") ||
            titleStr.includes("loading") ||
            titleStr.includes("preparing") ||
            titleStr.includes("joining") ||
            titleStr.includes("analyzing") ||
            descStr.includes("sending") ||
            descStr.includes("merging") ||
            descStr.includes("packaging"));

        const resolvedVariant = isDestructive
          ? "destructive"
          : isSuccess
          ? "success"
          : (variant || "default");

        return (
          <Toast key={id} variant={resolvedVariant as any} {...props}>
            <div className="flex gap-3 items-start flex-1 min-w-0">
              <div className="shrink-0 mt-0.5">
                {isDestructive ? (
                  <div className="h-8 w-8 rounded-full bg-rose-50 text-rose-600 border border-rose-200 dark:bg-rose-500/20 dark:text-rose-400 dark:border-rose-500/30 flex items-center justify-center shadow-sm">
                    <AlertCircle className="h-4 w-4 shrink-0" strokeWidth={2.2} />
                  </div>
                ) : isSuccess ? (
                  <div className="h-8 w-8 rounded-full bg-emerald-50 text-emerald-600 border border-emerald-200 dark:bg-emerald-500/20 dark:text-emerald-400 dark:border-emerald-500/30 flex items-center justify-center shadow-sm">
                    <CheckCircle2 className="h-4 w-4 shrink-0" strokeWidth={2.2} />
                  </div>
                ) : isLoading ? (
                  <div className="h-8 w-8 rounded-full bg-stone-100 text-stone-700 border border-stone-200 dark:bg-zinc-800 dark:text-zinc-200 dark:border-zinc-700 flex items-center justify-center shadow-sm">
                    <Loader2 className="h-4 w-4 shrink-0 animate-spin text-stone-600 dark:text-zinc-300" strokeWidth={2.2} />
                  </div>
                ) : (
                  <div className="h-8 w-8 rounded-full bg-stone-100 text-stone-700 border border-stone-200 dark:bg-zinc-800 dark:text-zinc-200 dark:border-zinc-700 flex items-center justify-center shadow-sm">
                    <Info className="h-4 w-4 shrink-0 text-stone-600 dark:text-zinc-300" strokeWidth={2.2} />
                  </div>
                )}
              </div>
              <div className="grid gap-0.5 min-w-0 pt-0.5">
                {title && <ToastTitle>{title}</ToastTitle>}
                {description && <ToastDescription>{description}</ToastDescription>}
              </div>
            </div>
            {action}
            <ToastClose />
          </Toast>
        );
      })}
      <ToastViewport />
    </ToastProvider>
  );
}
