import { FileSpreadsheet, ArrowUpRight, ShieldCheck, ArrowUp } from "lucide-react";
import { Link } from "react-router-dom";

const scrollToTop = () => {
  window.scrollTo({ top: 0, behavior: "smooth" });
};

export const Footer = () => {
  return (
    <footer className="border-t border-white/[0.08] bg-[#050608] pt-8 sm:pt-10 pb-6 sm:pb-8 mt-auto text-zinc-400">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8 pb-6 sm:pb-8 border-b border-white/[0.08]">
          
          {/* Brand & Value Proposition (Col 1-5) */}
          <div className="lg:col-span-5 space-y-3">
            <Link to="/" className="flex items-center gap-2 group w-fit">
              <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-white/[0.06] border border-white/10 text-white shadow-xs group-hover:scale-105 transition-transform duration-200">
                <FileSpreadsheet className="h-3.5 w-3.5 text-white" />
              </div>
              <span className="font-display font-bold text-base tracking-tight text-white group-hover:text-zinc-200 transition-colors">
                GetReport
              </span>
            </Link>
            
            <p className="text-xs text-zinc-400 leading-relaxed max-w-sm font-sans">
              Automated spreadsheet quality audit and executive reporting platform. Catch hidden errors, clean bad data instantly, and share boardroom-ready reports with 100% privacy.
            </p>

            <div className="flex items-center gap-2 text-[11px] font-mono text-zinc-400 pt-1">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
              <span>All Systems Operational</span>
            </div>
          </div>

          {/* Navigation Links Columns (Col 6-12) */}
          <div className="lg:col-span-7 grid grid-cols-2 sm:grid-cols-3 gap-6">
            
            {/* Column 1: Product */}
            <div className="space-y-2.5">
              <h4 className="text-[11px] font-sans font-bold text-white uppercase tracking-wider">
                Product
              </h4>
              <ul className="space-y-2 text-xs text-zinc-400 font-sans">
                <li>
                  <Link to="/features" className="hover:text-white transition-colors flex items-center gap-1 group">
                    <span>Features</span>
                    <ArrowUpRight className="h-3 w-3 opacity-0 -translate-x-1 group-hover:opacity-100 group-hover:translate-x-0 transition-all text-white" />
                  </Link>
                </li>
                <li>
                  <Link to="/how-it-works" className="hover:text-white transition-colors flex items-center gap-1 group">
                    <span>How It Works</span>
                    <ArrowUpRight className="h-3 w-3 opacity-0 -translate-x-1 group-hover:opacity-100 group-hover:translate-x-0 transition-all text-white" />
                  </Link>
                </li>
                <li>
                  <Link to="/pricing" className="hover:text-white transition-colors flex items-center gap-1 group">
                    <span>Pricing</span>
                    <ArrowUpRight className="h-3 w-3 opacity-0 -translate-x-1 group-hover:opacity-100 group-hover:translate-x-0 transition-all text-white" />
                  </Link>
                </li>
              </ul>
            </div>

            {/* Column 2: Resources */}
            <div className="space-y-2.5">
              <h4 className="text-[11px] font-sans font-bold text-white uppercase tracking-wider">
                Resources
              </h4>
              <ul className="space-y-2 text-xs text-zinc-400 font-sans">
                <li>
                  <Link to="/documentation" className="hover:text-white transition-colors flex items-center gap-1 group">
                    <span>Documentation</span>
                    <ArrowUpRight className="h-3 w-3 opacity-0 -translate-x-1 group-hover:opacity-100 group-hover:translate-x-0 transition-all text-white" />
                  </Link>
                </li>
                <li>
                  <Link to="/examples" className="hover:text-white transition-colors flex items-center gap-1 group">
                    <span>Interactive Examples</span>
                    <ArrowUpRight className="h-3 w-3 opacity-0 -translate-x-1 group-hover:opacity-100 group-hover:translate-x-0 transition-all text-white" />
                  </Link>
                </li>
              </ul>
            </div>

            {/* Column 3: Company */}
            <div className="space-y-2.5 col-span-2 sm:col-span-1">
              <h4 className="text-[11px] font-sans font-bold text-white uppercase tracking-wider">
                Company
              </h4>
              <ul className="space-y-2 text-xs text-zinc-400 font-sans">
                <li>
                  <Link to="/contact" className="hover:text-white transition-colors flex items-center gap-1 group">
                    <span>Contact Sales</span>
                    <ArrowUpRight className="h-3 w-3 opacity-0 -translate-x-1 group-hover:opacity-100 group-hover:translate-x-0 transition-all text-white" />
                  </Link>
                </li>
                <li>
                  <Link to="/privacy-policy" className="hover:text-white transition-colors flex items-center gap-1 group">
                    <span>Privacy Policy</span>
                    <ArrowUpRight className="h-3 w-3 opacity-0 -translate-x-1 group-hover:opacity-100 group-hover:translate-x-0 transition-all text-white" />
                  </Link>
                </li>
                <li>
                  <Link to="/terms-of-service" className="hover:text-white transition-colors flex items-center gap-1 group">
                    <span>Terms of Service</span>
                    <ArrowUpRight className="h-3 w-3 opacity-0 -translate-x-1 group-hover:opacity-100 group-hover:translate-x-0 transition-all text-white" />
                  </Link>
                </li>
              </ul>
            </div>

          </div>

        </div>

        {/* Bottom Bar: Copyright & Security Note */}
        <div className="pt-5 sm:pt-6 flex flex-col sm:flex-row sm:items-center justify-between gap-3 font-mono text-[11px] text-zinc-400">
          <div className="flex items-center gap-2">
            <ShieldCheck className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
            <span>© {new Date().getFullYear()} GetReport. 100% private. Your files are never saved on our servers.</span>
          </div>

          <div className="flex items-center gap-4">
            <span className="text-[10px] uppercase tracking-wider text-zinc-400">Safe & Encrypted</span>
            <button
              onClick={scrollToTop}
              className="p-1 px-2.5 rounded-lg border border-white/10 bg-white/[0.05] hover:bg-white/[0.1] text-zinc-200 transition-all duration-150 flex items-center gap-1.5 hover:-translate-y-0.5 cursor-pointer"
              title="Scroll to Top"
            >
              <ArrowUp className="h-3 w-3 text-zinc-300" />
              <span className="font-sans text-[10px] font-medium hidden sm:inline">Top</span>
            </button>
          </div>
        </div>

      </div>
    </footer>
  );
};

export default Footer;
