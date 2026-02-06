import { Terminal, Copy, Gauge, Brain, FileText } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";

const ApiDocs = () => {
    return (
        <div className="min-h-screen bg-background animate-in fade-in duration-500">
            <div className="container mx-auto px-4 py-16 max-w-5xl">
                <div className="text-center mb-16">
                    <div className="h-16 w-16 rounded-2xl bg-primary/10 flex items-center justify-center mx-auto mb-6">
                        <Terminal className="h-8 w-8 text-primary" />
                    </div>
                    <h1 className="text-4xl font-bold tracking-tight mb-4">API Reference</h1>
                    <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
                        Integrate GetReport's analysis engine directly into your applications.
                    </p>
                </div>

                <div className="grid gap-12 lg:grid-cols-[1fr_300px]">
                    <div className="space-y-12">
                        {/* Authentication */}
                        <section id="auth">
                            <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
                                <ShieldIcon className="h-6 w-6 text-primary" /> Authentication
                            </h2>
                            <p className="text-muted-foreground mb-4">
                                Currently, the API is open for public use without authentication for standard workloads. Rate limits apply based on IP address.
                            </p>
                            <div className="bg-muted p-4 rounded-lg font-mono text-sm">
                                Base URL: <span className="text-primary">https://api.getreport.com/api</span>
                            </div>
                        </section>

                        {/* Endpoints */}
                        <section id="endpoints" className="space-y-8">
                            <h2 className="text-2xl font-bold mb-4">Endpoints</h2>

                            <div>
                                <h3 className="text-lg font-semibold mb-2 flex items-center gap-2">
                                    <span className="bg-blue-100 text-blue-700 px-2 py-1 rounded text-xs font-mono">POST</span>
                                    /upload
                                </h3>
                                <p className="text-muted-foreground mb-4">Upload a file (CSV/Excel) to start a processing job.</p>
                                <CodeBlock
                                    code={`curl -X POST https://api.getreport.com/api/upload \\
  -H "Content-Type: multipart/form-data" \\
  -F "file=@dataset.csv"`}
                                />
                            </div>

                            <div>
                                <h3 className="text-lg font-semibold mb-2 flex items-center gap-2">
                                    <span className="bg-green-100 text-green-700 px-2 py-1 rounded text-xs font-mono">GET</span>
                                    /status/{'{task_id}'}
                                </h3>
                                <p className="text-muted-foreground mb-4">Check the status of a job and retrieve results.</p>
                                <CodeBlock
                                    code={`curl https://api.getreport.com/api/status/123-abc-456`}
                                />
                            </div>

                            <div>
                                <h3 className="text-lg font-semibold mb-2 flex items-center gap-2">
                                    <span className="bg-blue-100 text-blue-700 px-2 py-1 rounded text-xs font-mono">POST</span>
                                    /jobs/{'{task_id}'}/analyze
                                </h3>
                                <p className="text-muted-foreground mb-4">Submit cleaning rules and start the full analysis.</p>
                                <CodeBlock
                                    code={`curl -X POST https://api.getreport.com/api/jobs/123-abc-456/analyze \\
  -H "Content-Type: application/json" \\
  -d '{"rules": {"age": {"action": "fill_mean"}}}'`}
                                />
                            </div>

                            <div>
                                <h3 className="text-lg font-semibold mb-2 flex items-center gap-2">
                                    <span className="bg-green-100 text-green-700 px-2 py-1 rounded text-xs font-mono">GET</span>
                                    /jobs/{'{task_id}'}/report
                                </h3>
                                <p className="text-muted-foreground mb-4">Download the generated PDF report.</p>
                                <CodeBlock
                                    code={`curl https://api.getreport.com/api/jobs/123-abc-456/report \\
  -o report.pdf`}
                                />
                            </div>
                        </section>

                        {/* Response Format */}
                        <section id="response" className="space-y-6">
                            <h2 className="text-2xl font-bold mb-4">Response Format</h2>
                            <p className="text-muted-foreground mb-4">
                                All API responses include confidence scores and analysis decisions for transparency.
                            </p>
                            <div className="grid gap-4 sm:grid-cols-3">
                                <div className="p-4 border rounded-lg">
                                    <Gauge className="h-5 w-5 text-primary mb-2" />
                                    <h4 className="font-semibold mb-1">Confidence Scores</h4>
                                    <p className="text-sm text-muted-foreground">A-F grades for each column</p>
                                </div>
                                <div className="p-4 border rounded-lg">
                                    <Brain className="h-5 w-5 text-primary mb-2" />
                                    <h4 className="font-semibold mb-1">Analysis Decisions</h4>
                                    <p className="text-sm text-muted-foreground">Why each analysis ran/skipped</p>
                                </div>
                                <div className="p-4 border rounded-lg">
                                    <FileText className="h-5 w-5 text-primary mb-2" />
                                    <h4 className="font-semibold mb-1">Semantic Domain</h4>
                                    <p className="text-sm text-muted-foreground">Detected industry + confidence</p>
                                </div>
                            </div>
                        </section>
                    </div>

                    {/* Sidebar */}
                    <div className="space-y-6">
                        <div className="p-6 rounded-xl border bg-card">
                            <h4 className="font-semibold mb-4">On this page</h4>
                            <ul className="space-y-2 text-sm text-muted-foreground">
                                <li><a href="#auth" className="hover:text-primary">Authentication</a></li>
                                <li><a href="#endpoints" className="hover:text-primary">Endpoints</a></li>
                                <li><a href="#response" className="hover:text-primary">Response Format</a></li>
                            </ul>
                        </div>
                        <div className="p-6 rounded-xl bg-primary/5 border border-primary/10">
                            <h4 className="font-semibold mb-2">Need Help?</h4>
                            <p className="text-sm text-muted-foreground mb-4">
                                Contact our developer support team for custom integrations.
                            </p>
                            <Link to="/contact">
                                <Button size="sm" variant="outline" className="w-full">Contact Support</Button>
                            </Link>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

// Helper Components
const CodeBlock = ({ code }: { code: string }) => (
    <div className="bg-slate-950 text-slate-50 p-4 rounded-lg font-mono text-sm overflow-x-auto relative group">
        <pre>{code}</pre>
        <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
            <Copy className="h-4 w-4 text-slate-400 cursor-pointer hover:text-white" />
        </div>
    </div>
);

const ShieldIcon = ({ className }: { className?: string }) => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10" />
    </svg>
);

export default ApiDocs;
