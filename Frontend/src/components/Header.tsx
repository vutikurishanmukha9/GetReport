import { FileText, RotateCcw, Menu } from "lucide-react";
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
import { motion } from "framer-motion";

interface HeaderProps {
  onReset: () => void;
  showReset: boolean;
}

export const Header = ({ onReset, showReset }: HeaderProps) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex h-14 sm:h-16 items-center justify-between">
          {/* Logo */}
          <div className="flex items-center gap-2 sm:gap-3">
            <Link to="/" className="flex items-center gap-2 sm:gap-3">
              <div className="flex h-8 w-8 sm:h-10 sm:w-10 items-center justify-center rounded-lg bg-primary">
                <FileText className="h-4 w-4 sm:h-5 sm:w-5 text-primary-foreground" />
              </div>
              <span className="text-lg sm:text-xl lg:text-2xl font-bold tracking-tight">
                GetReport
              </span>
            </Link>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center gap-4 lg:gap-6">
            <Link to="/features" className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors">
              Features
            </Link>
            <Link to="/how-it-works" className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors">
              How it Works
            </Link>
            <ThemeToggle />
            {showReset && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.2 }}
              >
                <Button variant="outline" size="sm" onClick={onReset} className="gap-2">
                  <RotateCcw className="h-4 w-4" />
                  Start Over
                </Button>
              </motion.div>
            )}
          </nav>

          {/* Mobile Menu */}
          <div className="flex md:hidden items-center gap-2">
            <ThemeToggle />
            {showReset && (
              <Button variant="ghost" size="icon" onClick={onReset} className="h-9 w-9">
                <RotateCcw className="h-4 w-4" />
              </Button>
            )}
            <Sheet open={mobileMenuOpen} onOpenChange={setMobileMenuOpen}>
              <SheetTrigger asChild>
                <Button variant="ghost" size="icon" className="h-9 w-9">
                  <Menu className="h-5 w-5" />
                </Button>
              </SheetTrigger>
              <SheetContent side="right" className="w-[280px] sm:w-[320px]">
                <SheetTitle className="sr-only">Navigation Menu</SheetTitle>
                <nav className="flex flex-col gap-4 mt-8">
                  <Link
                    to="/features"
                    className="text-lg font-medium hover:text-primary transition-colors"
                    onClick={() => setMobileMenuOpen(false)}
                  >
                    Features
                  </Link>
                  <Link
                    to="/how-it-works"
                    className="text-lg font-medium hover:text-primary transition-colors"
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
