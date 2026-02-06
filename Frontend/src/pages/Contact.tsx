import { Mail, MessageSquare, Github, Twitter, Linkedin } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { useState } from "react";

const Contact = () => {
    const [submitted, setSubmitted] = useState(false);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        setSubmitted(true);
    };

    return (
        <div className="min-h-screen bg-background animate-in fade-in duration-500">
            <div className="container mx-auto px-4 py-16 max-w-5xl">
                <div className="text-center mb-12">
                    <h1 className="text-4xl font-bold mb-4">Get in Touch</h1>
                    <p className="text-xl text-muted-foreground">
                        Have questions about GetReport? We'd love to hear from you.
                    </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                    {/* Contact Info */}
                    <div className="space-y-6">
                        <Card>
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <Mail className="h-5 w-5 text-primary" />
                                    Email Us
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <p className="text-muted-foreground text-sm mb-1">General inquiries:</p>
                                <p className="font-medium">hello@getreport.com</p>
                                <p className="text-muted-foreground text-sm mt-3 mb-1">Support:</p>
                                <p className="font-medium">support@getreport.com</p>
                            </CardContent>
                        </Card>

                        <Card>
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <MessageSquare className="h-5 w-5 text-primary" />
                                    Response Time
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <p className="text-muted-foreground text-sm">
                                    We typically respond within 24 hours on business days. For urgent issues, mention "URGENT" in your subject line.
                                </p>
                            </CardContent>
                        </Card>

                        {/* Social Links */}
                        <Card>
                            <CardHeader>
                                <CardTitle>Connect With Us</CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="flex gap-4">
                                    <a href="https://github.com/vutikurishanmukha9/GetReport" target="_blank" rel="noopener noreferrer" className="h-10 w-10 rounded-full bg-muted flex items-center justify-center hover:bg-primary/10 transition-colors">
                                        <Github className="h-5 w-5" />
                                    </a>
                                    <a href="#" className="h-10 w-10 rounded-full bg-muted flex items-center justify-center hover:bg-primary/10 transition-colors">
                                        <Twitter className="h-5 w-5" />
                                    </a>
                                    <a href="#" className="h-10 w-10 rounded-full bg-muted flex items-center justify-center hover:bg-primary/10 transition-colors">
                                        <Linkedin className="h-5 w-5" />
                                    </a>
                                </div>
                            </CardContent>
                        </Card>
                    </div>

                    {/* Form */}
                    <div className="md:col-span-2">
                        <Card>
                            <CardHeader>
                                <CardTitle>Send us a Message</CardTitle>
                                <CardDescription>Fill out the form below and we'll get back to you as soon as possible.</CardDescription>
                            </CardHeader>
                            <CardContent>
                                {submitted ? (
                                    <div className="text-center py-12">
                                        <div className="h-16 w-16 rounded-full bg-green-100 dark:bg-green-900/40 flex items-center justify-center mx-auto mb-4">
                                            <Mail className="h-8 w-8 text-green-600 dark:text-green-400" />
                                        </div>
                                        <h3 className="text-xl font-semibold mb-2">Message Sent!</h3>
                                        <p className="text-muted-foreground">Thank you for reaching out. We'll respond within 24 hours.</p>
                                        <Button variant="outline" className="mt-6" onClick={() => setSubmitted(false)}>
                                            Send Another Message
                                        </Button>
                                    </div>
                                ) : (
                                    <form className="space-y-4" onSubmit={handleSubmit}>
                                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                            <div className="space-y-2">
                                                <label className="text-sm font-medium">First Name</label>
                                                <Input placeholder="John" required />
                                            </div>
                                            <div className="space-y-2">
                                                <label className="text-sm font-medium">Last Name</label>
                                                <Input placeholder="Doe" required />
                                            </div>
                                        </div>
                                        <div className="space-y-2">
                                            <label className="text-sm font-medium">Email</label>
                                            <Input type="email" placeholder="john@example.com" required />
                                        </div>
                                        <div className="space-y-2">
                                            <label className="text-sm font-medium">Subject</label>
                                            <Input placeholder="How can we help?" required />
                                        </div>
                                        <div className="space-y-2">
                                            <label className="text-sm font-medium">Message</label>
                                            <Textarea placeholder="Tell us more about your question or feedback..." className="min-h-[120px]" required />
                                        </div>
                                        <Button type="submit" className="w-full sm:w-auto">Send Message</Button>
                                    </form>
                                )}
                            </CardContent>
                        </Card>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Contact;
