import { FileSpreadsheet, ArrowUpRight, ShieldCheck, ArrowUp } from "lucide-react";
import { Link } from "react-router-dom";

const scrollToTop = () => {
  window.scrollTo({ top: 0, behavior: "smooth" });
};

export const Footer = () => {
  return (
    <footer className="border-t border-border bg-card/80 backdrop-blur-md pt-6 sm:pt-8 pb-5 sm:pb-6 mt-auto text-foreground">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8 pb-5 sm:pb-6 border-b border-border/60">
          
          {/* Brand & Value Proposition (Col 1-5) */}
          <div className="lg:col-span-5 space-y-3">
            <Link to="/" className="flex items-center gap-2 group w-fit">
              <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-primary text-primary-foreground shadow-xs group-hover:scale-105 transition-transform duration-200">
                <FileSpreadsheet className="h-3.5 w-3.5" />
              </div>
              <span className="font-display font-bold text-base tracking-tight text-foreground group-hover:text-primary transition-colors">
                GetReport
              </span>
            </Link>
            
            <p className="text-xs text-muted-foreground leading-relaxed max-w-sm font-sans">
              Automated data intelligence, quality audit, and executive report platform. Transform messy CSV and Excel ledgers into auditable insights with in-memory security guarantees.
            </p>

            {/* System Operational Status Badge */}
            <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full border border-emerald-500/30 bg-emerald-500/10 text-emerald-700 text-[10px] font-mono font-medium">
              <ShieldCheck className="h-3 w-3 text-emerald-600 shrink-0" />
              <span>Polars Engine: 100% Operational</span>
            </div>
          </div>

          {/* Navigation Links Columns (Col 6-12) */}
          <div className="lg:col-span-7 grid grid-cols-2 sm:grid-cols-3 gap-6">
            
            {/* Column 1: Product */}
            <div className="space-y-2.5">
              <h4 className="text-[11px] font-display font-bold text-foreground uppercase tracking-wider text-primary/90">
                Product
              </h4>
              <ul className="space-y-1.5 text-xs text-muted-foreground font-sans">
                <li>
                  <Link to="/features" className="hover:text-primary transition-colors flex items-center gap-1 group">
                    <span>Features</span>
                    <ArrowUpRight className="h-3 w-3 opacity-0 -translate-x-1 group-hover:opacity-100 group-hover:translate-x-0 transition-all text-primary" />
                  </Link>
                </li>
                <li>
                  <Link to="/how-it-works" className="hover:text-primary transition-colors flex items-center gap-1 group">
                    <span>How It Works</span>
                    <ArrowUpRight className="h-3 w-3 opacity-0 -translate-x-1 group-hover:opacity-100 group-hover:translate-x-0 transition-all text-primary" />
                  </Link>
                </li>
                <li>
                  <Link to="/pricing" className="hover:text-primary transition-colors flex items-center gap-1 group">
                    <span>Pricing</span>
                    <ArrowUpRight className="h-3 w-3 opacity-0 -translate-x-1 group-hover:opacity-100 group-hover:translate-x-0 transition-all text-primary" />
                  </Link>
                </li>
              </ul>
            </div>

            {/* Column 2: Resources */}
            <div className="space-y-2.5">
              <h4 className="text-[11px] font-display font-bold text-foreground uppercase tracking-wider text-primary/90">
                Resources
              </h4>
              <ul className="space-y-1.5 text-xs text-muted-foreground font-sans">
                <li>
                  <Link to="/documentation" className="hover:text-primary transition-colors flex items-center gap-1 group">
                    <span>Documentation</span>
                    <ArrowUpRight className="h-3 w-3 opacity-0 -translate-x-1 group-hover:opacity-100 group-hover:translate-x-0 transition-all text-primary" />
                  </Link>
                </li>
                <li>
                  <Link to="/documentation#security" className="hover:text-primary transition-colors flex items-center gap-1 group">
                    <span>Architecture Specs</span>
                    <ArrowUpRight className="h-3 w-3 opacity-0 -translate-x-1 group-hover:opacity-100 group-hover:translate-x-0 transition-all text-primary" />
                  </Link>
                </li>
                <li>
                  <Link to="/examples" className="hover:text-primary transition-colors flex items-center gap-1 group">
                    <span>Interactive Examples</span>
                    <ArrowUpRight className="h-3 w-3 opacity-0 -translate-x-1 group-hover:opacity-100 group-hover:translate-x-0 transition-all text-primary" />
                  </Link>
                </li>
              </ul>
            </div>

            {/* Column 3: Company */}
            <div className="space-y-2.5 col-span-2 sm:col-span-1">
              <h4 className="text-[11px] font-display font-bold text-foreground uppercase tracking-wider text-primary/90">
                Company
              </h4>
              <ul className="space-y-1.5 text-xs text-muted-foreground font-sans">
                <li>
                  <Link to="/contact" className="hover:text-primary transition-colors flex items-center gap-1 group">
                    <span>Contact Sales</span>
                    <ArrowUpRight className="h-3 w-3 opacity-0 -translate-x-1 group-hover:opacity-100 group-hover:translate-x-0 transition-all text-primary" />
                  </Link>
                </li>
                <li>
                  <Link to="/privacy-policy" className="hover:text-primary transition-colors flex items-center gap-1 group">
                    <span>Privacy Policy</span>
                    <ArrowUpRight className="h-3 w-3 opacity-0 -translate-x-1 group-hover:opacity-100 group-hover:translate-x-0 transition-all text-primary" />
                  </Link>
                </li>
                <li>
                  <Link to="/terms-of-service" className="hover:text-primary transition-colors flex items-center gap-1 group">
                    <span>Terms of Service</span>
                    <ArrowUpRight className="h-3 w-3 opacity-0 -translate-x-1 group-hover:opacity-100 group-hover:translate-x-0 transition-all text-primary" />
                  </Link>
                </li>
              </ul>
            </div>

          </div>

        </div>

        {/* Bottom Bar: Copyright & Security Note */}
        <div className="pt-4 sm:pt-4.5 flex flex-col sm:flex-row sm:items-center justify-between gap-3 font-mono text-[11px] text-muted-foreground">
          <div className="flex items-center gap-2">
            <ShieldCheck className="h-3.5 w-3.5 text-emerald-600 shrink-0" />
            <span>© {new Date().getFullYear()} GetReport. Session-scoped processing for every audit.</span>
          </div>

          <div className="flex items-center gap-4">
            <span className="text-[10px] uppercase tracking-wider text-muted-foreground/70">TLS 1.3 Encrypted</span>
            <button
              onClick={scrollToTop}
              className="p-1 px-2 rounded-md border border-border bg-background hover:bg-muted/50 text-foreground transition-all duration-200 flex items-center gap-1 hover:-translate-y-0.5"
              title="Scroll to Top"
            >
              <ArrowUp className="h-3 w-3 text-primary" />
              <span className="font-sans text-[10px] font-medium hidden sm:inline">Top</span>
            </button>
          </div>
        </div>

      </div>
    </footer>
  );
};
