import { 
  FileSpreadsheet, 
  RotateCcw, 
  Menu, 
  ArrowRight,
  Layers,
  Cpu,
  CreditCard,
  BookOpen,
  BarChart3,
  Mail,
  ShieldCheck 
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useState, useEffect, useRef } from "react";
import { Link, NavLink, useLocation } from "react-router-dom";
import {
  Sheet,
  SheetContent,
  SheetTrigger,
  SheetTitle,
} from "@/components/ui/sheet";

interface HeaderProps {
  onReset: () => void;
  showReset: boolean;
}

export const Header = ({ onReset, showReset }: HeaderProps) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [isVisible, setIsVisible] = useState(true);
  const [isScrolled, setIsScrolled] = useState(false);
  const lastScrollY = useRef(0);
  const location = useLocation();

  useEffect(() => {
    const handleScroll = () => {
      const currentScrollY = window.scrollY;

      // Always visible when near the top
      if (currentScrollY <= 30) {
        setIsVisible(true);
        setIsScrolled(false);
        lastScrollY.current = currentScrollY;
        return;
      }

      setIsScrolled(true);
      const diff = currentScrollY - lastScrollY.current;

      // Ignore micro-jitter
      if (Math.abs(diff) < 8) return;

      if (diff > 0 && currentScrollY > 80) {
        // Scrolling down: hide smoothly out of the way
        setIsVisible(false);
      } else if (diff < 0) {
        // Scrolling up: reveal smoothly for instant navigation
        setIsVisible(true);
      }

      lastScrollY.current = currentScrollY;
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const navLinks = [
    { to: "/features", label: "Features" },
    { to: "/how-it-works", label: "How it works" },
    { to: "/pricing", label: "Pricing" },
    { to: "/documentation", label: "Docs" },
    { to: "/examples", label: "Examples" },
  ];

  const shouldBeVisible = isVisible || mobileMenuOpen;

  return (
    <header
      className={`fixed top-3.5 sm:top-5 left-0 right-0 z-50 px-3 sm:px-6 pointer-events-none transition-all duration-300 ease-out ${
        shouldBeVisible
          ? "translate-y-0 opacity-100"
          : "-translate-y-24 opacity-0"
      }`}
    >
      <div className="max-w-5xl mx-auto pointer-events-auto">
        {/* Clean Single-Border Pill Navigation without heavy halo ring */}
        <div
          className={`rounded-full bg-white/95 sm:bg-white/90 backdrop-blur-xl px-3.5 sm:px-4 py-2 flex items-center justify-between gap-2 sm:gap-4 transition-all duration-300 border border-white/60 ${
            isScrolled
              ? "shadow-[0_20px_48px_-12px_rgba(40,20,70,0.22)]"
              : "shadow-[0_16px_40px_-12px_rgba(40,20,70,0.16)]"
          }`}
        >
          {/* Left Pod: Brand Identity */}
          <div className="flex items-center gap-2.5">
            <Link 
              to="/" 
              onClick={onReset} 
              className="flex items-center gap-2.5 group shrink-0 rounded-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400"
            >
              <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-gradient-to-tr from-violet-600 via-purple-600 to-indigo-600 text-white shadow-md shadow-violet-600/25 transition-transform duration-200 group-hover:scale-105">
                <FileSpreadsheet className="h-4 w-4" />
              </div>
              <span className="text-base font-hero font-extrabold tracking-tight text-slate-900 group-hover:text-violet-700 transition-colors">
                GetReport
              </span>
            </Link>
          </div>

          {/* Center Pod: High-Contrast Segmented Navigator Track */}
          <nav className="hidden md:flex items-center p-1 rounded-full bg-slate-100/90 border border-slate-200/70 gap-0.5">
            {navLinks.map(({ to, label }) => (
              <NavLink
                key={to}
                to={to}
                className={({ isActive }) =>
                  `px-3.5 py-1.5 rounded-full text-[13px] font-hero font-medium transition-all duration-150 relative focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400 ${
                    isActive
                      ? "bg-white text-violet-950 font-bold shadow-xs border border-slate-200/80"
                      : "text-[#2b2b3d] hover:text-black hover:bg-slate-900/[0.06]"
                  }`
                }
              >
                {label}
              </NavLink>
            ))}
          </nav>

          {/* Right Pod: Action Console */}
          <div className="flex items-center gap-2">
            {showReset ? (
              <Button
                variant="outline"
                size="sm"
                onClick={onReset}
                className="h-9 px-3.5 rounded-xl shadow-2xs border-slate-200/90 bg-white text-slate-700 hover:bg-slate-50 hover:text-violet-700 hover:border-violet-300 transition-all duration-150 hover:-translate-y-0.5 active:scale-95 font-hero font-semibold text-xs gap-1.5 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400"
              >
                <RotateCcw className="h-3.5 w-3.5 text-violet-600" />
                <span>Start Over</span>
              </Button>
            ) : location.pathname !== "/workspace" ? (
              <Link to="/workspace" className="focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400 rounded-xl">
                <button 
                  type="button"
                  className="h-9 px-4 rounded-xl bg-gradient-to-r from-violet-600 via-purple-600 to-indigo-600 hover:from-violet-500 hover:via-purple-500 hover:to-indigo-500 text-white font-hero font-bold text-xs tracking-tight shadow-[0_3px_12px_-1px_rgba(124,58,237,0.4)] border border-violet-400/30 flex items-center gap-2 group transition-all duration-150 hover:-translate-y-0.5 active:scale-95 cursor-pointer shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400"
                >
                  <span>Start Free</span>
                  <ArrowRight className="h-3.5 w-3.5 text-white/90 transition-transform duration-200 group-hover:translate-x-0.5 shrink-0" />
                </button>
              </Link>
            ) : (
              <div className="hidden sm:inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-violet-50 border border-violet-200/80 text-violet-800 text-[11px] font-hero font-medium">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
                <span>Session Active</span>
              </div>
            )}

              {/* Mobile Drawer Trigger */}
              <div className="md:hidden">
                <Sheet open={mobileMenuOpen} onOpenChange={setMobileMenuOpen}>
                  <SheetTrigger asChild>
                    <button
                      className="h-9 w-9 rounded-xl bg-slate-100 hover:bg-slate-200/80 border border-slate-200/90 flex items-center justify-center text-slate-800 active:scale-95 transition-all shadow-2xs cursor-pointer"
                      aria-label="Open Navigation Menu"
                    >
                      <Menu className="h-4 w-4" />
                    </button>
                  </SheetTrigger>
                  <SheetContent
                    side="right"
                    className="w-[310px] sm:w-[350px] rounded-l-3xl border-l border-slate-200/80 bg-white/95 backdrop-blur-3xl p-5 sm:p-6 flex flex-col justify-between shadow-2xl"
                  >
                    <div>
                      {/* Drawer Brand Header */}
                      <div className="flex items-center gap-3 pb-5 border-b border-slate-100">
                        <div className="flex h-9 w-9 items-center justify-center rounded-2xl bg-gradient-to-tr from-violet-600 via-purple-600 to-indigo-600 text-white shadow-md shadow-violet-600/25">
                          <FileSpreadsheet className="h-4.5 w-4.5" />
                        </div>
                        <div>
                          <SheetTitle className="text-base font-display font-black text-slate-900 leading-none">
                            GetReport
                          </SheetTitle>
                          <span className="text-[10px] font-mono text-emerald-600 font-semibold flex items-center gap-1 mt-1">
                            <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
                            100% Private & Secure
                          </span>
                        </div>
                      </div>

                      {/* Tactile Native App Menu Cards */}
                      <nav className="flex flex-col gap-2 mt-5">
                        {[
                          { to: "/features", label: "Features", icon: Layers, desc: "Quality checks & auto-fixes" },
                          { to: "/how-it-works", label: "How It Works", icon: Cpu, desc: "Simple 4-step process" },
                          { to: "/pricing", label: "Pricing", icon: CreditCard, desc: "Free & flexible plans" },
                          { to: "/documentation", label: "Documentation", icon: BookOpen, desc: "Guides & help center" },
                          { to: "/examples", label: "Interactive Examples", icon: BarChart3, desc: "See sample reports" },
                          { to: "/contact", label: "Contact Team", icon: Mail, desc: "Enterprise support" },
                        ].map(({ to, label, icon: NavIcon, desc }) => {
                          const isActive = location.pathname === to;
                          return (
                            <Link
                              key={to}
                              to={to}
                              className={`p-3 rounded-2xl border transition-all duration-150 flex items-center justify-between active:scale-[0.98] ${
                                isActive
                                  ? "bg-violet-50/90 border-violet-300 text-violet-950 shadow-xs"
                                  : "bg-slate-50/80 hover:bg-slate-100/80 border-slate-200/60 text-slate-700"
                              }`}
                              onClick={() => setMobileMenuOpen(false)}
                            >
                              <div className="flex items-center gap-3">
                                <div className={`h-8 w-8 rounded-xl flex items-center justify-center shrink-0 ${
                                  isActive
                                    ? "bg-violet-600 text-white shadow-xs"
                                    : "bg-white text-slate-600 border border-slate-200 shadow-2xs"
                                }`}>
                                  <NavIcon className="h-4 w-4" />
                                </div>
                                <div className="text-left">
                                  <span className="text-xs font-display font-bold block leading-tight">
                                    {label}
                                  </span>
                                  <span className="text-[10px] text-slate-400 font-sans block leading-tight mt-0.5">
                                    {desc}
                                  </span>
                                </div>
                              </div>
                              <span className="text-slate-300 font-mono text-xs pr-1">→</span>
                            </Link>
                          );
                        })}
                      </nav>
                    </div>

                    {/* Bottom Action Pod */}
                    <div className="pt-4 border-t border-slate-100 space-y-2.5">
                      <Link
                        to="/workspace"
                        className="w-full block"
                        onClick={() => setMobileMenuOpen(false)}
                      >
                        <button className="w-full h-12 rounded-xl bg-gradient-to-r from-violet-600 via-purple-600 to-indigo-600 hover:from-violet-500 hover:via-purple-500 hover:to-indigo-500 text-white font-display font-bold text-sm shadow-[0_4px_16px_-2px_rgba(124,58,237,0.4),inset_0_1px_1px_rgba(255,255,255,0.3)] border border-violet-400/40 active:scale-[0.98] transition-all flex items-center justify-center gap-2.5 cursor-pointer">
                          <span>Launch Workspace</span>
                          <ArrowRight className="h-4 w-4" />
                        </button>
                      </Link>
                      <div className="flex items-center justify-center gap-1.5 text-[10px] font-mono text-slate-400">
                        <ShieldCheck className="h-3.5 w-3.5 text-emerald-500" />
                        <span>100% Private • Files never saved to disk</span>
                      </div>
                    </div>
                  </SheetContent>
                </Sheet>
              </div>
            </div>
          </div>
        </div>
      </header>
    );
  };
