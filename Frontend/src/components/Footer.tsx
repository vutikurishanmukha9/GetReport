/* Hallmark · component: footer · genre: modern-minimal · theme: Quiet
 * macrostructure: Statement knobs: width=38ch, wordmark=bottom-left, rule=hairline
 * contrast: pass
 */

import { FileSpreadsheet } from "lucide-react";
import { Link } from "react-router-dom";

export const Footer = () => {
  return (
    <footer className="border-t border-border/70 bg-muted/20 py-12 sm:py-16 mt-auto">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-5xl">
        <div className="space-y-12 text-left">
          
          {/* Main bold Statement (Ft5) */}
          <div className="max-w-xl">
            <h3 className="text-xl sm:text-2xl font-display font-bold tracking-tight text-foreground leading-snug">
              Turn raw datasets into professional, publication-ready reports in seconds.
            </h3>
            <p className="text-xs sm:text-sm text-muted-foreground mt-3 leading-relaxed">
              GetReport processes your data securely. Uploaded files are evaluated for constraints locally and deleted automatically after processing.
            </p>
          </div>

          {/* Hairline separator */}
          <div className="h-px bg-border/60" />

          {/* Bottom metadata and clean inline link row */}
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-6 font-mono text-[11px] sm:text-xs">
            
            {/* Wordmark bottom left */}
            <div className="flex items-center gap-2 text-foreground font-semibold">
              <div className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-primary-foreground shadow-premium">
                <FileSpreadsheet className="h-3 w-3" />
              </div>
              <span className="font-display font-bold text-sm">GetReport</span>
            </div>

            {/* Flat links row aligned with bottom */}
            <div className="flex flex-wrap gap-x-6 gap-y-3 text-muted-foreground">
              <Link to="/features" className="hover:text-foreground hover:underline transition-colors duration-150">features</Link>
              <Link to="/how-it-works" className="hover:text-foreground hover:underline transition-colors duration-150">how it works</Link>
              <Link to="/pricing" className="hover:text-foreground hover:underline transition-colors duration-150">pricing</Link>
              <Link to="/documentation" className="hover:text-foreground hover:underline transition-colors duration-150">docs</Link>
              <Link to="/contact" className="hover:text-foreground hover:underline transition-colors duration-150">contact</Link>
              <Link to="/privacy-policy" className="hover:text-foreground hover:underline transition-colors duration-150">privacy</Link>
              <Link to="/terms-of-service" className="hover:text-foreground hover:underline transition-colors duration-150">terms</Link>
            </div>

            {/* Copyright */}
            <span className="text-muted-foreground self-start sm:self-center font-mono">
              © {new Date().getFullYear()} GetReport.
            </span>

          </div>

        </div>
      </div>
    </footer>
  );
};
