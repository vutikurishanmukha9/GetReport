import { FileText } from "lucide-react";
import { Link } from "react-router-dom";

export const Footer = () => {
  return (
    <footer className="border-t bg-muted/30">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-10 md:py-12">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8 lg:gap-12">
          {/* Brand */}
          <div className="sm:col-span-2 lg:col-span-1">
            <div className="flex items-center gap-2 mb-4">
              <Link to="/" className="flex items-center gap-2">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
                  <FileText className="h-4 w-4 text-primary-foreground" />
                </div>
                <span className="text-lg font-bold">GetReport</span>
              </Link>
            </div>
            <p className="text-sm text-muted-foreground max-w-xs">
              Transform your datasets into professional analytical reports instantly.
              No configuration needed.
            </p>
          </div>

          {/* Product */}
          <div>
            <h4 className="font-semibold mb-3 sm:mb-4">Product</h4>
            <ul className="space-y-2 sm:space-y-3 text-sm text-muted-foreground">
              <li><Link to="/features" className="hover:text-foreground transition-colors">Features</Link></li>
              <li><Link to="/how-it-works" className="hover:text-foreground transition-colors">How it Works</Link></li>
              <li><Link to="/pricing" className="hover:text-foreground transition-colors">Pricing</Link></li>
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h4 className="font-semibold mb-3 sm:mb-4">Resources</h4>
            <ul className="space-y-2 sm:space-y-3 text-sm text-muted-foreground">
              <li><Link to="/documentation" className="hover:text-foreground transition-colors">Documentation</Link></li>
              <li><Link to="/api" className="hover:text-foreground transition-colors">API</Link></li>
              <li><Link to="/examples" className="hover:text-foreground transition-colors">Examples</Link></li>
            </ul>
          </div>

          {/* Legal */}
          <div>
            <h4 className="font-semibold mb-3 sm:mb-4">Legal</h4>
            <ul className="space-y-2 sm:space-y-3 text-sm text-muted-foreground">
              <li><Link to="/privacy-policy" className="hover:text-foreground transition-colors">Privacy Policy</Link></li>
              <li><Link to="/terms-of-service" className="hover:text-foreground transition-colors">Terms of Service</Link></li>
              <li><Link to="/contact" className="hover:text-foreground transition-colors">Contact</Link></li>
            </ul>
          </div>
        </div>

        {/* Bottom */}
        <div className="mt-8 sm:mt-10 md:mt-12 pt-6 sm:pt-8 border-t">
          <p className="text-xs sm:text-sm text-center text-muted-foreground">
            © {new Date().getFullYear()} GetReport. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
};
