import React from "react";
import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";

interface CtaBannerProps {
  title?: string;
  description?: string;
  buttonText?: string;
  buttonLink?: string;
  className?: string;
}

export const CtaBanner: React.FC<CtaBannerProps> = ({
  title = "Ready to audit a dataset?",
  description = "Launch the interactive workspace and see these algorithms in action.",
  buttonText = "Launch Workspace",
  buttonLink = "/workspace",
  className = "border-t border-border/60 bg-muted/20 py-10 sm:py-12",
}) => {
  return (
    <div className={className}>
      <div className="container mx-auto px-4 text-center space-y-4 max-w-3xl">
        <h2 className="text-xl sm:text-2xl font-display font-bold text-foreground tracking-[-0.015em]">
          {title}
        </h2>
        <p className="text-xs sm:text-sm text-muted-foreground font-sans max-w-lg mx-auto">
          {description}
        </p>
        <div className="pt-1 flex flex-wrap items-center justify-center gap-3">
          <Link to={buttonLink}>
            <Button
              size="lg"
              className="h-11 px-7 rounded-xl shadow-premium t-card-lift t-spring-press font-display font-semibold text-sm"
            >
              <span>{buttonText}</span>
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </Link>
        </div>
      </div>
    </div>
  );
};
