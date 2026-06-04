/* Hallmark · component: nav · genre: modern-minimal · theme: Quiet
 * states: default · hover · focus · active · disabled
 * contrast: pass
 */

import { FileSpreadsheet, RotateCcw, Menu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useState } from "react";
import { Link } from "react-router-dom";
import {
  Sheet,
  SheetContent,
  SheetTrigger,
  SheetTitle,
} from "@/components/ui/sheet";
import { ThemeToggle } from "@/components/ThemeToggle";

interface HeaderProps {
  onReset: () => void;
  showReset: boolean;
}

export const Header = ({ onReset, showReset }: HeaderProps) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <header className="fixed top-4 left-0 right-0 z-50 w-full max-w-5xl mx-auto px-4 sm:px-6">
      <div className="border bg-background/80 backdrop-blur-lg shadow-sm rounded-full px-4 sm:px-6 transition-all duration-200">
        <div className="flex h-14 items-center justify-between">
          {/* Logo */}
          <div className="flex items-center gap-2">
            <Link to="/" onClick={onReset} className="flex items-center gap-2.5 group">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground group-hover:bg-primary/90 transition-colors">
                <FileSpreadsheet className="h-4 w-4" />
              </div>
              <span className="text-base sm:text-lg font-semibold tracking-tight">
                GetReport
              </span>
            </Link>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center gap-6">
            <Link 
              to="/features" 
              className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors duration-150"
            >
              Features
            </Link>
            <Link 
              to="/how-it-works" 
              className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors duration-150"
            >
              How it Works
            </Link>
            <div className="h-4 w-px bg-border mx-1" />
            <ThemeToggle />
            {showReset && (
              <Button 
                variant="outline" 
                size="sm" 
                onClick={onReset} 
                className="gap-2 rounded-full transition-all duration-150 active:scale-95"
              >
                <RotateCcw className="h-3.5 w-3.5" />
                Start Over
              </Button>
            )}
          </nav>

          {/* Mobile Menu */}
          <div className="flex md:hidden items-center gap-2">
            <ThemeToggle />
            {showReset && (
              <Button 
                variant="ghost" 
                size="icon" 
                onClick={onReset} 
                className="h-9 w-9 rounded-full transition-all duration-150"
              >
                <RotateCcw className="h-4 w-4" />
              </Button>
            )}
            <Sheet open={mobileMenuOpen} onOpenChange={setMobileMenuOpen}>
              <SheetTrigger asChild>
                <Button variant="ghost" size="icon" className="h-9 w-9 rounded-full">
                  <Menu className="h-5 w-5" />
                </Button>
              </SheetTrigger>
              <SheetContent side="right" className="w-[280px] sm:w-[320px] rounded-l-3xl border-l">
                <SheetTitle className="sr-only">Navigation Menu</SheetTitle>
                <nav className="flex flex-col gap-4 mt-8">
                  <Link
                    to="/features"
                    className="text-lg font-medium hover:text-primary transition-colors py-2 border-b"
                    onClick={() => setMobileMenuOpen(false)}
                  >
                    Features
                  </Link>
                  <Link
                    to="/how-it-works"
                    className="text-lg font-medium hover:text-primary transition-colors py-2 border-b"
                    onClick={() => setMobileMenuOpen(false)}
                  >
                    How it Works
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
