import { Link } from "react-router-dom";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";
import { ArrowRight, Compass, FileSpreadsheet } from "lucide-react";
import { Button } from "@/components/ui/button";

const NotFound = () => {
  return (
    <div className="min-h-screen flex flex-col bg-[#F5F2EB] text-stone-900 dark:bg-[#07080a] dark:text-zinc-100 relative">
      {/* Technical Background Grid & Film Grain */}
      <div className="absolute inset-0 bg-[radial-gradient(rgba(0,0,0,0.05)_1px,transparent_1px)] dark:bg-[radial-gradient(rgba(255,255,255,0.06)_1px,transparent_1px)] [background-size:24px_24px] pointer-events-none opacity-40" />
      <div className="hero-grain-overlay" />
      
      <Header onReset={() => {}} showReset={false} />

      <main className="flex-1 flex items-center justify-center px-4 sm:px-6 pt-28 pb-16 relative z-10">
        <div className="max-w-lg w-full text-center space-y-6">
          <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-stone-200/80 border border-stone-300 text-stone-700 dark:bg-white/[0.04] dark:border-white/10 dark:text-zinc-400 shadow-md dark:shadow-xl">
            <Compass className="h-8 w-8 text-stone-600 dark:text-zinc-300 animate-spin-slow" />
          </div>

          <div className="space-y-2">
            <span className="font-mono text-xs uppercase tracking-widest text-stone-500 dark:text-zinc-400 font-semibold">
              Error 404 • Destination Not Found
            </span>
            <h1 className="text-3xl sm:text-4xl font-display font-black tracking-tight text-stone-900 dark:text-white">
              Lost in the Pipeline
            </h1>
            <p className="text-sm text-stone-600 dark:text-zinc-400 max-w-sm mx-auto leading-relaxed font-sans">
              The page or audit session you requested does not exist, expired from memory, or has been relocated.
            </p>
          </div>

          <div className="pt-2 flex flex-col sm:flex-row items-center justify-center gap-3">
            <Link to="/workspace" className="w-full sm:w-auto">
              <Button size="default" className="w-full sm:w-auto rounded-xl bg-[#800000] text-white hover:bg-[#6b0000] dark:bg-white dark:text-black dark:hover:bg-zinc-200 font-semibold text-xs gap-2 h-10 px-5 shadow-sm">
                <FileSpreadsheet className="h-4 w-4" />
                <span>Launch Workspace</span>
              </Button>
            </Link>
            <Link to="/" className="w-full sm:w-auto">
              <Button variant="outline" size="default" className="w-full sm:w-auto rounded-xl border-stone-300 bg-stone-100 hover:bg-stone-200 text-stone-800 dark:border-white/10 dark:bg-white/[0.04] dark:hover:bg-white/[0.08] dark:text-zinc-300 text-xs gap-2 h-10 px-5">
                <span>Return to Home</span>
                <ArrowRight className="h-3.5 w-3.5 text-stone-600 dark:text-zinc-400" />
              </Button>
            </Link>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default NotFound;
