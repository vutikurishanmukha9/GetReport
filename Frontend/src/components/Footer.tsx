import { FileSpreadsheet } from "lucide-react";
import { Link } from "react-router-dom";

export const Footer = () => {
  return (
    <footer className="border-t border-border bg-card py-16 mt-auto">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-5xl">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 md:gap-12 pb-12">
          {/* Brand Col */}
          <div className="space-y-4 md:col-span-1">
            <div className="flex items-center gap-2 text-foreground font-semibold">
              <div className="flex h-7 w-7 items-center justify-center rounded-full bg-primary text-primary-foreground shadow-premium">
                <FileSpreadsheet className="h-3.5 w-3.5" />
              </div>
              <span className="font-display font-bold text-base tracking-tight">GetReport</span>
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Turn raw datasets into professional, publication-ready reports in seconds with in-memory security.
            </p>
          </div>

          {/* Links Cols */}
          <div>
            <h4 className="text-xs font-display font-bold text-foreground uppercase tracking-wider mb-4">Product</h4>
            <ul className="space-y-2.5 text-xs text-muted-foreground font-sans">
              <li><Link to="/features" className="hover:text-foreground hover:underline transition-colors">features</Link></li>
              <li><Link to="/how-it-works" className="hover:text-foreground hover:underline transition-colors">how it works</Link></li>
              <li><Link to="/pricing" className="hover:text-foreground hover:underline transition-colors">pricing</Link></li>
            </ul>
          </div>

          <div>
            <h4 className="text-xs font-display font-bold text-foreground uppercase tracking-wider mb-4">Resources</h4>
            <ul className="space-y-2.5 text-xs text-muted-foreground font-sans">
              <li><Link to="/documentation" className="hover:text-foreground hover:underline transition-colors">documentation</Link></li>
              <li><Link to="/api" className="hover:text-foreground hover:underline transition-colors">api reference</Link></li>
              <li><Link to="/examples" className="hover:text-foreground hover:underline transition-colors">examples</Link></li>
            </ul>
          </div>

          <div>
            <h4 className="text-xs font-display font-bold text-foreground uppercase tracking-wider mb-4">Company</h4>
            <ul className="space-y-2.5 text-xs text-muted-foreground font-sans">
              <li><Link to="/contact" className="hover:text-foreground hover:underline transition-colors">contact sales</Link></li>
              <li><Link to="/privacy-policy" className="hover:text-foreground hover:underline transition-colors">privacy policy</Link></li>
              <li><Link to="/terms-of-service" className="hover:text-foreground hover:underline transition-colors">terms of service</Link></li>
            </ul>
          </div>
        </div>

        {/* Hairline separator */}
        <div className="h-px bg-border/60" />

        {/* Bottom copyright */}
        <div className="pt-8 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 font-mono text-[10px] text-muted-foreground">
          <span>© {new Date().getFullYear()} GetReport Inc. Ephemeral in-memory calculation engine.</span>
          <span>All rights reserved.</span>
        </div>
      </div>
    </footer>
  );
};
