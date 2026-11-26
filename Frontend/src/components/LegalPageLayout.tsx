import type { ReactNode } from "react";
import { ShieldCheck } from "lucide-react";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";
import { cn } from "@/lib/utils";

interface LegalPageLayoutProps {
  badgeText: string;
  title: string;
  subtitle: string;
  contentClassName?: string;
  children: ReactNode;
}

const noopReset = () => {};

/** Shared page frame and header layout for legal documentation pages. */
export function LegalPageLayout({
  badgeText,
  title,
  subtitle,
  contentClassName,
  children,
}: LegalPageLayoutProps) {
  return (
    <div className="min-h-screen flex flex-col bg-[#F5F2EB] text-stone-900 dark:bg-[#07080a] dark:text-zinc-100 animate-in fade-in duration-500 relative">
      <div className="absolute inset-0 bg-[radial-gradient(rgba(0,0,0,0.05)_1px,transparent_1px)] dark:bg-[radial-gradient(rgba(255,255,255,0.06)_1px,transparent_1px)] [background-size:24px_24px] pointer-events-none opacity-40" />
      <div className="hero-grain-overlay" />

      <Header onReset={noopReset} showReset={false} />

      <main className="flex-1 pt-24 sm:pt-28 relative z-10">
        <div className="border-b border-stone-300 bg-[#FAF8F5]/90 dark:border-white/[0.08] dark:bg-[#07080a]/90 py-8 sm:py-14 relative z-10">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-5xl text-center space-y-3 sm:space-y-4">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-stone-300 bg-[#FAF8F5] text-stone-700 dark:border-white/10 dark:bg-white/[0.04] dark:text-zinc-300 text-xs font-mono shadow-xs">
              <ShieldCheck className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />
              <span>{badgeText}</span>
            </div>

            <h1 className="text-3xl sm:text-4xl md:text-5xl font-display font-black text-stone-900 dark:text-white tracking-[-0.03em] leading-[1.12]">
              {title}
            </h1>

            <p className="text-sm sm:text-base text-stone-600 dark:text-zinc-400 max-w-2xl mx-auto leading-relaxed font-sans">
              {subtitle}
            </p>
          </div>
        </div>

        <div className={cn("container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12", contentClassName)}>
          {children}
        </div>
      </main>

      <Footer />
    </div>
  );
}
