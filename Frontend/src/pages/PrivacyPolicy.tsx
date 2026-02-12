import { Shield, Database, Lock, Eye, Trash2, Server } from "lucide-react";

const PrivacyPolicy = () => {
    return (
        <div className="min-h-screen bg-background animate-in fade-in duration-500">
            <div className="container mx-auto px-4 py-16 max-w-4xl">
                <div className="mb-12 border-b pb-8">
                    <div className="flex items-center gap-4 mb-4">
                        <Shield className="h-8 w-8 text-primary" />
                        <h1 className="text-4xl font-bold">Privacy Policy</h1>
                    </div>
                    <p className="text-muted-foreground">Last updated: February 2026</p>
                </div>

                {/* Key Highlights */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-12">
                    <div className="p-4 border rounded-lg text-center">
                        <Trash2 className="h-6 w-6 text-primary mx-auto mb-2" />
                        <h4 className="font-semibold text-sm">Ephemeral Processing</h4>
                        <p className="text-xs text-muted-foreground">Data deleted after analysis</p>
                    </div>
                    <div className="p-4 border rounded-lg text-center">
                        <Lock className="h-6 w-6 text-primary mx-auto mb-2" />
                        <h4 className="font-semibold text-sm">No Data Selling</h4>
                        <p className="text-xs text-muted-foreground">Your data is never sold</p>
                    </div>
                    <div className="p-4 border rounded-lg text-center">
                        <Eye className="h-6 w-6 text-primary mx-auto mb-2" />
                        <h4 className="font-semibold text-sm">PII Masking</h4>
                        <p className="text-xs text-muted-foreground">Sensitive data masked before AI</p>
                    </div>
                </div>

                <div className="prose prose-slate dark:prose-invert max-w-none space-y-8">
                    <section>
                        <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
                            <Database className="h-5 w-5" /> 1. Data Collection
                        </h2>
                        <p className="text-muted-foreground leading-relaxed">
                            We take your data privacy seriously. When you upload a file to GetReport, it is processed ephemerally on our secure servers. We do not store your raw datasets permanently. Once the analysis is complete and your report is generated, the original data is discarded from our systems within 24 hours.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
                            <Eye className="h-5 w-5" /> 2. Usage of Information
                        </h2>
                        <p className="text-muted-foreground leading-relaxed">
                            The information you provide is used solely for the purpose of generating statistical reports and insights. We do not sell, trade, or otherwise transfer to outside parties your personally identifiable information or proprietary data. Statistical summaries (not raw data) may be used to improve our algorithms.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
                            <Lock className="h-5 w-5" /> 3. AI Processing
                        </h2>
                        <p className="text-muted-foreground leading-relaxed">
                            When AI-powered insights are generated, we only send statistical summaries and metadata to our AI providers never your raw data rows. Personally Identifiable Information (PII) is masked before any external processing. Column names and aggregate statistics are used to generate narrative insights.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
                            <Server className="h-5 w-5" /> 4. Security
                        </h2>
                        <p className="text-muted-foreground leading-relaxed">
                            We implement industry-standard security measures including HTTPS encryption for all data transmission, secure server infrastructure, and access controls. Your uploaded files are processed in isolated environments and are not accessible to other users or our staff.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold mb-4">5. Your Rights</h2>
                        <p className="text-muted-foreground leading-relaxed">
                            You have the right to: (a) know what data we have about you, (b) request deletion of your data, (c) opt out of any analytics, and (d) export your generated reports. Since we don't store your raw data after processing, there's nothing to delete our system is privacy-by-design.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold mb-4">6. Cookies & Tracking</h2>
                        <p className="text-muted-foreground leading-relaxed">
                            We use minimal cookies strictly for session management and security. We do not use third-party tracking cookies or advertising pixels. Your analysis session is private.
                        </p>
                    </section>

                    <section>
                        <h2 className="text-2xl font-semibold mb-4">7. Contact</h2>
                        <p className="text-muted-foreground leading-relaxed">
                            If you have any questions about this privacy policy, please contact us at privacy@getreport.com.
                        </p>
                    </section>
                </div>
            </div>
        </div>
    );
};

export default PrivacyPolicy;
