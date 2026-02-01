import { Sparkles, Zap, FileCheck } from "lucide-react";

export const HeroSection = () => {
  return (
    <section className="relative overflow-hidden">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-background to-accent/10 -z-10" />
      
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-12 sm:py-16 md:py-20 lg:py-24">
        <div className="max-w-4xl mx-auto text-center">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-3 py-1.5 sm:px-4 sm:py-2 rounded-full bg-primary/10 text-primary text-xs sm:text-sm font-medium mb-6 sm:mb-8">
            <Sparkles className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
            <span>AI-Powered Data Analysis</span>
          </div>

          {/* Headline */}
          <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight mb-4 sm:mb-6">
            Turn Your Data Into
            <span className="block text-primary mt-1 sm:mt-2">Professional Reports</span>
          </h1>

          {/* Subheadline */}
          <p className="text-base sm:text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto mb-8 sm:mb-10 md:mb-12 px-4">
            Upload any CSV or Excel file and get a complete analytical report with charts, 
            insights, and recommendations — in seconds, not hours.
          </p>

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 sm:gap-6 lg:gap-8 max-w-2xl mx-auto">
            <div className="flex flex-col items-center p-3 sm:p-4 rounded-xl bg-card border">
              <Zap className="h-5 w-5 sm:h-6 sm:w-6 text-primary mb-2" />
              <span className="text-xl sm:text-2xl font-bold">10s</span>
              <span className="text-xs sm:text-sm text-muted-foreground">Average Time</span>
            </div>
            <div className="flex flex-col items-center p-3 sm:p-4 rounded-xl bg-card border">
              <FileCheck className="h-5 w-5 sm:h-6 sm:w-6 text-primary mb-2" />
              <span className="text-xl sm:text-2xl font-bold">PDF</span>
              <span className="text-xs sm:text-sm text-muted-foreground">Ready Report</span>
            </div>
            <div className="flex flex-col items-center p-3 sm:p-4 rounded-xl bg-card border col-span-2 md:col-span-1">
              <Sparkles className="h-5 w-5 sm:h-6 sm:w-6 text-primary mb-2" />
              <span className="text-xl sm:text-2xl font-bold">Zero</span>
              <span className="text-xs sm:text-sm text-muted-foreground">Configuration</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
