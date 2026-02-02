import { BookOpen, FileSpreadsheet, AlertTriangle, FileText, CheckCircle2 } from "lucide-react";
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
                <div className="grid gap-12">
                    {/* Getting Started */}
                    <section>
                        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
                            <BookOpen className="h-6 w-6 text-primary" /> Getting Started
                        </h2>
                        <div className="prose prose-slate dark:prose-invert max-w-none">
                            <p>
                                GetReport is designed to be the fastest way to go from a raw spreadsheet to a professional board-ready report.
                                Our engine automatically handles data cleaning, statistical analysis, and visualization generation.
                            </p>
                            <h3 className="text-lg font-semibold mt-6 mb-3">Supported File Formats</h3>
                            <ul className="space-y-2 list-none pl-0">
                                <li className="flex items-center gap-2">
                                    <FileSpreadsheet className="h-4 w-4 text-green-600" />
                                    <span><strong>CSV (.csv):</strong> Comma-separated values. Ideal for large datasets.</span>
                                </li>
                                <li className="flex items-center gap-2">
                                    <FileSpreadsheet className="h-4 w-4 text-green-600" />
                                    <span><strong>Excel (.xls, .xlsx):</strong> Microsoft Excel workbooks. We support multiple sheets (defaults to first).</span>
                                </li>
                            </ul>
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
                                <AccordionTrigger>How does the AI Insights work?</AccordionTrigger>
                                <AccordionContent>
                                    We sample your data's statistical summary (not the raw confidential rows) and use a Large Language Model to identify trends and anomalies.
                                    Your raw PII data is never used for training.
                                </AccordionContent>
                            </AccordionItem>
                            <AccordionItem value="item-3">
                                <AccordionTrigger>Can I export the cleaned data?</AccordionTrigger>
                                <AccordionContent>
                                    Yes! After the analysis is complete, you can download the generated PDF report which contains the methodology.
                                    Raw cleaned data export is coming in the next release.
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
