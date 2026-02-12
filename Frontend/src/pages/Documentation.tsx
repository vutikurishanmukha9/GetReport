import { BookOpen, FileSpreadsheet, AlertTriangle, Gauge, Brain, CheckCircle2 } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";

const Documentation = () => {
    return (
        <div className="min-h-screen bg-background animate-in fade-in duration-500">
            {/* Header */}
            <div className="bg-muted/30 border-b">
                <div className="container mx-auto px-4 py-16 text-center">
                    <h1 className="text-4xl font-bold tracking-tight mb-4">Documentation</h1>
                    <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
                        Guides, references, and best practices for using GetReport.
                    </p>
                </div>
            </div>

            <div className="container mx-auto px-4 py-12 max-w-4xl">
                {/* Popular Topics Grid */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-16">
                    <Link to="#" className="p-6 border rounded-xl hover:bg-muted/50 transition-colors group">
                        <FileSpreadsheet className="h-8 w-8 text-green-600 mb-3 group-hover:scale-110 transition-transform" />
                        <h3 className="font-semibold mb-1">Data Prep</h3>
                        <p className="text-xs text-muted-foreground">Formatting your CSV/Excel for best results</p>
                    </Link>
                    <Link to="#" className="p-6 border rounded-xl hover:bg-muted/50 transition-colors group">
                        <Gauge className="h-8 w-8 text-blue-600 mb-3 group-hover:scale-110 transition-transform" />
                        <h3 className="font-semibold mb-1">Interpreting Scores</h3>
                        <p className="text-xs text-muted-foreground">Understanding A-F confidence grades</p>
                    </Link>
                    <Link to="#" className="p-6 border rounded-xl hover:bg-muted/50 transition-colors group">
                        <Brain className="h-8 w-8 text-purple-600 mb-3 group-hover:scale-110 transition-transform" />
                        <h3 className="font-semibold mb-1">AI Insights</h3>
                        <p className="text-xs text-muted-foreground">How our RAG architecture works</p>
                    </Link>
                </div>

                <div className="grid gap-12">
                    {/* Getting Started */}
                    <section>
                        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
                            <BookOpen className="h-6 w-6 text-primary" /> Getting Started
                        </h2>
                        <div className="prose prose-slate dark:prose-invert max-w-none">
                            <p>
                                GetReport transforms raw spreadsheets into professional, auditable reports.
                                Every analysis decision is explained. Every column is graded. Every action is transparent.
                            </p>
                            <h3 className="text-lg font-semibold mt-6 mb-3">Supported File Formats</h3>
                            <ul className="space-y-2 list-none pl-0">
                                <li className="flex items-center gap-2">
                                    <FileSpreadsheet className="h-4 w-4 text-green-600" />
                                    <span><strong>CSV (.csv):</strong> Comma-separated values. Auto-detects encoding and delimiters.</span>
                                </li>
                                <li className="flex items-center gap-2">
                                    <FileSpreadsheet className="h-4 w-4 text-green-600" />
                                    <span><strong>Excel (.xls, .xlsx):</strong> Microsoft Excel workbooks. Uses first sheet by default.</span>
                                </li>
                            </ul>
                        </div>
                    </section>

                    {/* Tier 1: Trust Foundation */}
                    <section>
                        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
                            <Gauge className="h-6 w-6 text-primary" /> Tier 1: Trust Foundation
                        </h2>
                        <div className="grid gap-4">
                            <div className="p-4 border rounded-lg">
                                <h4 className="font-semibold mb-2">Column Confidence Scores</h4>
                                <p className="text-muted-foreground text-sm">
                                    Each column is graded A-F across four dimensions: <strong>Completeness</strong> (% non-null),
                                    <strong> Consistency</strong> (format uniformity), <strong>Validity</strong> (expected ranges),
                                    and <strong>Stability</strong> (variance detection).
                                </p>
                            </div>
                            <div className="p-4 border rounded-lg">
                                <h4 className="font-semibold mb-2">Why I Did X Explanations</h4>
                                <p className="text-muted-foreground text-sm">
                                    Full transparency into analysis decisions. See why each analysis (correlation, outliers, time-series, etc.)
                                    was run or skipped, with evidence and reasoning.
                                </p>
                            </div>
                            <div className="p-4 border rounded-lg">
                                <h4 className="font-semibold mb-2">Semantic Domain Detection</h4>
                                <p className="text-muted-foreground text-sm">
                                    Automatic domain classification (Education, Sales, Healthcare, HR, Finance) with confidence percentage,
                                    matched keywords, and alternative domain candidates.
                                </p>
                            </div>
                        </div>
                    </section>

                    {/* Tier 2: Intelligence */}
                    <section>
                        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
                            <Brain className="h-6 w-6 text-primary" /> Tier 2: Advanced Intelligence
                        </h2>
                        <div className="grid gap-4">
                            <div className="p-4 border rounded-lg">
                                <h4 className="font-semibold mb-2">Feature Engineering Recommendations</h4>
                                <p className="text-muted-foreground text-sm">
                                    ML-ready suggestions: encoding (One-Hot, Label, Target), scaling (StandardScaler, MinMax, RobustScaler),
                                    and feature creation ideas.
                                </p>
                            </div>
                            <div className="p-4 border rounded-lg">
                                <h4 className="font-semibold mb-2">Smart Schema Inference</h4>
                                <p className="text-muted-foreground text-sm">
                                    Detects type mismatches, suggests corrections, identifies hidden dates in strings,
                                    and discovers implicit relationships between columns.
                                </p>
                            </div>
                            <div className="p-4 border rounded-lg">
                                <h4 className="font-semibold mb-2">Actionable Recommendations</h4>
                                <p className="text-muted-foreground text-sm">
                                    Domain-specific next steps with priority levels (Critical, High, Medium, Low).
                                    Categories: data_quality, analysis, reporting, ml_prep.
                                </p>
                            </div>
                        </div>
                    </section>

                    {/* FAQ / troubleshooting */}
                    <section>
                        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
                            <AlertTriangle className="h-6 w-6 text-amber-500" /> Troubleshooting & FAQ
                        </h2>
                        <Accordion type="single" collapsible className="w-full">
                            <AccordionItem value="item-1">
                                <AccordionTrigger>My file failed to upload. Why?</AccordionTrigger>
                                <AccordionContent>
                                    Common reasons include files larger than 50MB, corrupted formatting, or unsupported extensions.
                                    Ensure your file is a valid CSV or Excel file and try again.
                                </AccordionContent>
                            </AccordionItem>
                            <AccordionItem value="item-2">
                                <AccordionTrigger>What do the confidence grades mean?</AccordionTrigger>
                                <AccordionContent>
                                    Grades range from A (excellent, 90%+) to F (critical issues, below 50%).
                                    Each grade combines four scores: Completeness, Consistency, Validity, and Stability.
                                    High-confidence columns (A-B) are reliable for analysis. Low-confidence columns (D-F) need attention.
                                </AccordionContent>
                            </AccordionItem>
                            <AccordionItem value="item-3">
                                <AccordionTrigger>How does the AI Insights work?</AccordionTrigger>
                                <AccordionContent>
                                    We use statistical summaries (not raw confidential rows) with a Large Language Model to identify trends.
                                    Your raw PII data is never used for training or sent to external services.
                                </AccordionContent>
                            </AccordionItem>
                            <AccordionItem value="item-4">
                                <AccordionTrigger>Why was a specific analysis skipped?</AccordionTrigger>
                                <AccordionContent>
                                    Check the "Analysis Decisions" section in your PDF report. Each skipped analysis includes
                                    a reason (e.g., "Time-series skipped: no datetime columns detected" or "Correlation skipped: fewer than 2 numeric columns").
                                </AccordionContent>
                            </AccordionItem>
                        </Accordion>
                    </section>
                </div>

                <div className="mt-16 text-center border-t pt-8">
                    <p className="text-muted-foreground mb-4">Still have questions?</p>
                    <Link to="/contact">
                        <Button variant="secondary">Contact Support</Button>
                    </Link>
                </div>
            </div>
        </div>
    );
};

export default Documentation;
