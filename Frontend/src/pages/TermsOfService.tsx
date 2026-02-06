import { Scale, FileText, AlertTriangle, CheckCircle2, XCircle } from "lucide-react";

const TermsOfService = () => {
    return (
        <div className="min-h-screen bg-background animate-in fade-in duration-500">
            <div className="container mx-auto px-4 py-16 max-w-4xl">
                <div className="mb-12 border-b pb-8">
                    <div className="flex items-center gap-4 mb-4">
                        <Scale className="h-8 w-8 text-primary" />
                        <h1 className="text-4xl font-bold">Terms of Service</h1>
                    </div>
                    <p className="text-muted-foreground">Last updated: February 2026</p>
                </div>

                {/* Quick Summary */}
                <div className="p-6 bg-muted/50 rounded-lg mb-12">
                    <h3 className="font-semibold mb-4">Quick Summary (Not Legal Advice)</h3>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
                        <div className="flex items-start gap-2">
                            <CheckCircle2 className="h-4 w-4 text-green-600 mt-0.5 shrink-0" />
                            <span>Use our service for legitimate data analysis</span>
                        </div>
                        <div className="flex items-start gap-2">
                            <CheckCircle2 className="h-4 w-4 text-green-600 mt-0.5 shrink-0" />
                            <span>Your data belongs to you</span>
                        </div>
                        <div className="flex items-start gap-2">
                            <CheckCircle2 className="h-4 w-4 text-green-600 mt-0.5 shrink-0" />
                            <span>Download and share your reports freely</span>
                        </div>
                        <div className="flex items-start gap-2">
                            <XCircle className="h-4 w-4 text-red-600 mt-0.5 shrink-0" />
                            <span>Don't upload illegal or harmful content</span>
                        </div>
                    </div>
                </div>

                <div className="prose prose-slate dark:prose-invert max-w-none space-y-8">
                    <section>
                        <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
                            <FileText className="h-5 w-5" /> 1. Acceptance of Terms
                        </h2>
                        <p className="text-muted-foreground leading-relaxed">
                            By accessing and using GetReport, you accept and agree to be bound by the terms and provision of this agreement. If you do not agree to these terms, please do not use our service.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold mb-4">2. Use License</h2>
                        <p className="text-muted-foreground leading-relaxed">
                            Permission is granted to use GetReport for personal and commercial data analysis purposes. Generated reports are yours to keep, share, and distribute. You retain all rights to your uploaded data and generated insights.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold mb-4">3. Acceptable Use</h2>
                        <p className="text-muted-foreground leading-relaxed mb-4">
                            You agree not to:
                        </p>
                        <ul className="space-y-2 text-muted-foreground">
                            <li className="flex items-start gap-2">
                                <AlertTriangle className="h-4 w-4 text-amber-500 mt-1 shrink-0" />
                                Upload data that violates any applicable laws or regulations
                            </li>
                            <li className="flex items-start gap-2">
                                <AlertTriangle className="h-4 w-4 text-amber-500 mt-1 shrink-0" />
                                Attempt to reverse-engineer or exploit our analysis algorithms
                            </li>
                            <li className="flex items-start gap-2">
                                <AlertTriangle className="h-4 w-4 text-amber-500 mt-1 shrink-0" />
                                Use automated tools to overwhelm our servers (rate limits apply)
                            </li>
                            <li className="flex items-start gap-2">
                                <AlertTriangle className="h-4 w-4 text-amber-500 mt-1 shrink-0" />
                                Misrepresent generated reports as coming from a different source
                            </li>
                        </ul>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold mb-4">4. Service Availability</h2>
                        <p className="text-muted-foreground leading-relaxed">
                            We strive to provide reliable service but do not guarantee 100% uptime. The service is provided "as is" and we reserve the right to modify, suspend, or discontinue features at any time. We will provide reasonable notice for significant changes.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold mb-4">5. Disclaimer</h2>
                        <p className="text-muted-foreground leading-relaxed">
                            GetReport provides statistical analysis and AI-generated insights for informational purposes. Generated reports should not be the sole basis for critical business, medical, legal, or financial decisions. Always verify important findings with domain experts. We are not liable for decisions made based on our analysis.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold mb-4">6. Intellectual Property</h2>
                        <p className="text-muted-foreground leading-relaxed">
                            The GetReport platform, including its algorithms, interface, and branding, is the intellectual property of GetReport. Your uploaded data and generated reports remain your property. Our analysis methodology and confidence scoring systems are proprietary.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold mb-4">7. Contact</h2>
                        <p className="text-muted-foreground leading-relaxed">
                            Questions about these terms? Contact us at legal@getreport.com.
                        </p>
                    </section>
                </div>
            </div>
        </div>
    );
};

export default TermsOfService;
