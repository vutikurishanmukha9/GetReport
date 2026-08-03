import { FileSpreadsheet, RotateCcw, Menu, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useState } from "react";
import { Link } from "react-router-dom";
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

  return (
    <header className="fixed top-4 left-0 right-0 z-50 w-full max-w-7xl mx-auto px-4 sm:px-6">
      <div className="border border-border/80 bg-background/90 backdrop-blur-xl shadow-premium rounded-full px-4 sm:px-6 transition-all duration-200">
        <div className="flex h-14 items-center justify-between">
          {/* Logo */}
          <div className="flex items-center gap-2">
            <Link to="/" onClick={onReset} className="flex items-center gap-2.5 group">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground shadow-md transition-all duration-200 group-hover:scale-105 group-hover:shadow-primary/20">
                <FileSpreadsheet className="h-4 w-4" />
              </div>
              <span className="text-base font-display font-bold tracking-tight text-foreground group-hover:text-primary transition-colors">
                GetReport
              </span>
            </Link>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center gap-6">
            <Link 
              to="/features" 
              className="text-xs font-display font-semibold uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors duration-150 relative after:absolute after:bottom-[-4px] after:left-0 after:h-[2px] after:w-0 hover:after:w-full after:bg-primary after:transition-all after:duration-200"
            >
              Features
            </Link>
            <Link 
              to="/how-it-works" 
              className="text-xs font-display font-semibold uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors duration-150 relative after:absolute after:bottom-[-4px] after:left-0 after:h-[2px] after:w-0 hover:after:w-full after:bg-primary after:transition-all after:duration-200"
            >
              How It Works
            </Link>
            <Link 
              to="/pricing" 
              className="text-xs font-display font-semibold uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors duration-150 relative after:absolute after:bottom-[-4px] after:left-0 after:h-[2px] after:w-0 hover:after:w-full after:bg-primary after:transition-all after:duration-200"
            >
              Pricing
            </Link>
            <Link 
              to="/documentation" 
              className="text-xs font-display font-semibold uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors duration-150 relative after:absolute after:bottom-[-4px] after:left-0 after:h-[2px] after:w-0 hover:after:w-full after:bg-primary after:transition-all after:duration-200"
            >
              Docs
            </Link>
            <Link 
              to="/examples" 
              className="text-xs font-display font-semibold uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors duration-150 relative after:absolute after:bottom-[-4px] after:left-0 after:h-[2px] after:w-0 hover:after:w-full after:bg-primary after:transition-all after:duration-200"
            >
              Examples
            </Link>

            {showReset ? (
              <>
                <div className="h-4 w-px bg-border/60 mx-1" />
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={onReset} 
                  className="gap-2 rounded-full shadow-2xs border-border/80 bg-white hover:bg-muted/40 transition-all duration-150 hover:-translate-y-0.5 active:scale-95 font-semibold text-xs px-3.5"
                >
                  <RotateCcw className="h-3.5 w-3.5 text-primary" />
                  <span>Start Over</span>
                </Button>
              </>
            ) : (
              <>
                <div className="h-4 w-px bg-border/60 mx-1" />
                <Link to="/workspace">
                  <Button 
                    size="sm" 
                    className="rounded-full shadow-premium bg-primary text-primary-foreground hover:bg-primary/90 transition-all duration-150 hover:-translate-y-0.5 active:scale-95 font-semibold text-xs px-4 cursor-pointer"
                  >
                    Launch Workspace
                  </Button>
                </Link>
              </>
            )}
          </nav>

          {/* Mobile Menu */}
          <div className="flex md:hidden items-center gap-2">
            {showReset && (
              <Button 
                variant="ghost" 
                size="icon" 
                onClick={onReset} 
                className="h-9 w-9 rounded-full transition-all duration-150 active:scale-95 text-primary"
              >
                <RotateCcw className="h-4 w-4" />
              </Button>
            )}
            <Sheet open={mobileMenuOpen} onOpenChange={setMobileMenuOpen}>
              <SheetTrigger asChild>
                <Button variant="ghost" size="icon" className="h-9 w-9 rounded-full active:scale-95">
                  <Menu className="h-5 w-5" />
                </Button>
              </SheetTrigger>
              <SheetContent side="right" className="w-[280px] sm:w-[320px] rounded-l-3xl border-l border-border/80 bg-background/95 backdrop-blur-xl">
                <SheetTitle className="sr-only">Navigation Menu</SheetTitle>
                <nav className="flex flex-col gap-4 mt-8">
                  <Link
                    to="/workspace"
                    className="text-base font-display font-bold text-primary hover:text-primary transition-colors py-2 border-b border-border/60"
                    onClick={() => setMobileMenuOpen(false)}
                  >
                    Launch Workspace
                  </Link>
                  <Link
                    to="/features"
                    className="text-sm font-display font-semibold uppercase tracking-wider hover:text-primary transition-colors py-2 border-b border-border/60"
                    onClick={() => setMobileMenuOpen(false)}
                  >
                    Features
                  </Link>
                  <Link
                    to="/how-it-works"
                    className="text-sm font-display font-semibold uppercase tracking-wider hover:text-primary transition-colors py-2 border-b border-border/60"
                    onClick={() => setMobileMenuOpen(false)}
                  >
                    How It Works
                  </Link>
                  <Link
                    to="/pricing"
                    className="text-sm font-display font-semibold uppercase tracking-wider hover:text-primary transition-colors py-2 border-b border-border/60"
                    onClick={() => setMobileMenuOpen(false)}
                  >
                    Pricing ($0 Free)
                  </Link>
                  <Link
                    to="/documentation"
                    className="text-sm font-display font-semibold uppercase tracking-wider hover:text-primary transition-colors py-2 border-b border-border/60"
                    onClick={() => setMobileMenuOpen(false)}
                  >
                    Documentation
                  </Link>
                  <Link
                    to="/examples"
                    className="text-sm font-display font-semibold uppercase tracking-wider hover:text-primary transition-colors py-2 border-b border-border/60"
                    onClick={() => setMobileMenuOpen(false)}
                  >
                    Examples
                  </Link>
                  <Link
                    to="/contact"
                    className="text-sm font-display font-semibold uppercase tracking-wider hover:text-primary transition-colors py-2 border-b border-border/60"
                    onClick={() => setMobileMenuOpen(false)}
                  >
                    Contact
                  </Link>
                </nav>
              </SheetContent>
            </Sheet>
          </div>
        </div>
      </div>
    </header>
  );
};
