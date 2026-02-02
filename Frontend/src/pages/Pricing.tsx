import { Check, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Link } from "react-router-dom";

const Pricing = () => {
    return (
        <div className="min-h-screen bg-background animate-in fade-in duration-500">
            <div className="container mx-auto px-4 py-16 sm:py-24">
                <div className="text-center max-w-3xl mx-auto mb-16 space-y-4">
                    <h1 className="text-4xl sm:text-5xl font-bold tracking-tight">
                        Simple, Transparent Pricing
                    </h1>
                    <p className="text-xl text-muted-foreground">
                        Professional data analysis shouldn't break the bank.
                    </p>
                </div>

                <div className="max-w-md mx-auto">
                    <Card className="border-primary/20 shadow-lg relative overflow-hidden">
                        <div className="absolute top-0 right-0 p-3">
                            <span className="bg-primary/10 text-primary text-xs font-semibold px-3 py-1 rounded-full">
                                LIMITED TIME
                            </span>
                        </div>

                        <CardHeader className="text-center pb-8 pt-10">
                            <CardTitle className="text-2xl font-bold">Early Access</CardTitle>
                            <CardDescription>Everything you need to analyze data</CardDescription>
                            <div className="mt-4 flex items-baseline justify-center gap-1">
                                <span className="text-5xl font-extrabold">$0</span>
                                <span className="text-muted-foreground">/month</span>
                            </div>
                        </CardHeader>

                        <CardContent className="space-y-4">
                            <div className="bg-primary/5 p-4 rounded-lg text-center mb-6">
                                <p className="font-medium text-primary flex items-center justify-center gap-2">
                                    <Zap className="h-4 w-4 fill-primary" />
                                    Currently, we are offering all features for FREE.
                                </p>
                            </div>

                            <ul className="space-y-3 text-sm">
                                {[
                                    "Unlimited Data Uploads",
                                    "Advanced Statistical Analysis",
                                    "AI-Powered Insights",
                                    "PDF Report Generation",
                                    "Data Cleaning & Processing",
                                    "Visual Charts & Graphs",
                                    "Priority Support",
                                ].map((feature) => (
                                    <li key={feature} className="flex items-center gap-3 text-muted-foreground">
                                        <div className="h-5 w-5 rounded-full bg-green-100 dark:bg-green-900/40 flex items-center justify-center shrink-0">
                                            <Check className="h-3 w-3 text-green-600 dark:text-green-400" />
                                        </div>
                                        {feature}
                                    </li>
                                ))}
                            </ul>
                        </CardContent>

                        <CardFooter className="pt-8 pb-10">
                            <Link to="/" className="w-full">
                                <Button className="w-full h-12 text-lg font-medium shadow-md hover:shadow-lg transition-all">
                                    Get Started For Free
                                </Button>
                            </Link>
                        </CardFooter>
                    </Card>
                </div>

                <div className="mt-24 text-center">
                    <p className="text-muted-foreground text-sm">
                        No credit card required. No hidden fees. Just pure data power.
                    </p>
                </div>
            </div>
        </div>
    );
};

export default Pricing;
