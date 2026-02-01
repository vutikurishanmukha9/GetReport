import { 
  Wand2, 
  BarChart3, 
  FileText, 
  Sparkles, 
  Shield, 
  Clock 
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

const features = [
  {
    icon: Wand2,
    title: "Auto Data Cleaning",
    description: "Automatically handles missing values, duplicates, and data type corrections.",
  },
  {
    icon: BarChart3,
    title: "Smart Charts",
    description: "Generates the right visualizations based on your data structure automatically.",
  },
  {
    icon: Sparkles,
    title: "AI Insights",
    description: "Plain English explanations of patterns, trends, and key findings in your data.",
  },
  {
    icon: FileText,
    title: "PDF Export",
    description: "Download a professional, ready-to-share report with one click.",
  },
  {
    icon: Shield,
    title: "Secure Processing",
    description: "Your data is processed securely and never stored permanently.",
  },
  {
    icon: Clock,
    title: "Instant Results",
    description: "Get your complete analytical report in seconds, not hours.",
  },
];

export const FeaturesSection = () => {
  return (
    <section id="features" className="py-12 sm:py-16 md:py-20 lg:py-24 bg-muted/30">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-10 sm:mb-12 md:mb-16">
          <h2 className="text-2xl sm:text-3xl md:text-4xl font-bold mb-3 sm:mb-4">
            Everything Happens Automatically
          </h2>
          <p className="text-base sm:text-lg text-muted-foreground max-w-2xl mx-auto">
            No configuration needed. Just upload your data and get a complete report.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6 lg:gap-8">
          {features.map((feature, index) => (
            <Card 
              key={index} 
              className="group hover:shadow-lg transition-all duration-300 hover:-translate-y-1"
            >
              <CardContent className="p-5 sm:p-6">
                <div className="flex h-10 w-10 sm:h-12 sm:w-12 items-center justify-center rounded-lg bg-primary/10 text-primary mb-4 group-hover:bg-primary group-hover:text-primary-foreground transition-colors">
                  <feature.icon className="h-5 w-5 sm:h-6 sm:w-6" />
                </div>
                <h3 className="text-base sm:text-lg font-semibold mb-2">
                  {feature.title}
                </h3>
                <p className="text-sm sm:text-base text-muted-foreground">
                  {feature.description}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};
