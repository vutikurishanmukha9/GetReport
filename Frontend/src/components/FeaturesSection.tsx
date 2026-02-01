import { 
  Wand2, 
  BarChart3, 
  FileText, 
  Sparkles, 
  Shield, 
  Clock 
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { motion, type Variants } from "framer-motion";

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

const containerVariants: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
    },
  },
};

const cardVariants: Variants = {
  hidden: { opacity: 0, y: 30 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.5, ease: [0.25, 0.46, 0.45, 0.94] },
  },
};

export const FeaturesSection = () => {
  return (
    <section id="features" className="py-12 sm:py-16 md:py-20 lg:py-24 bg-muted/30">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div 
          className="text-center mb-10 sm:mb-12 md:mb-16"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.5 }}
        >
          <h2 className="text-2xl sm:text-3xl md:text-4xl font-bold mb-3 sm:mb-4">
            Everything Happens Automatically
          </h2>
          <p className="text-base sm:text-lg text-muted-foreground max-w-2xl mx-auto">
            No configuration needed. Just upload your data and get a complete report.
          </p>
        </motion.div>

        <motion.div 
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6 lg:gap-8"
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-50px" }}
        >
          {features.map((feature, index) => (
            <motion.div key={index} variants={cardVariants}>
              <Card className="group hover:shadow-lg transition-all duration-300 h-full">
                <CardContent className="p-5 sm:p-6">
                  <motion.div 
                    className="flex h-10 w-10 sm:h-12 sm:w-12 items-center justify-center rounded-lg bg-primary/10 text-primary mb-4 group-hover:bg-primary group-hover:text-primary-foreground transition-colors"
                    whileHover={{ scale: 1.1, rotate: 5 }}
                    transition={{ type: "spring", stiffness: 400 }}
                  >
                    <feature.icon className="h-5 w-5 sm:h-6 sm:w-6" />
                  </motion.div>
                  <h3 className="text-base sm:text-lg font-semibold mb-2">
                    {feature.title}
                  </h3>
                  <p className="text-sm sm:text-base text-muted-foreground">
                    {feature.description}
                  </p>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
};
