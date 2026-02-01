import { Mail, MessageSquare } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";

const Contact = () => {
    return (
        <div className="container mx-auto px-4 py-16 max-w-5xl animate-in fade-in duration-500">
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
                            <p className="text-muted-foreground text-sm">
                                support@getreport.com
                            </p>
                        </CardContent>
                    </Card>
                    <Card>
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                                <MessageSquare className="h-5 w-5 text-primary" />
                                Live Chat
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <p className="text-muted-foreground text-sm">
                                Available Mon-Fri, 9am - 5pm EST
                            </p>
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
                            <form className="space-y-4" onSubmit={(e) => e.preventDefault()}>
                                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                    <div className="space-y-2">
                                        <label className="text-sm font-medium">First Name</label>
                                        <Input placeholder="John" />
                                    </div>
                                    <div className="space-y-2">
                                        <label className="text-sm font-medium">Last Name</label>
                                        <Input placeholder="Doe" />
                                    </div>
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm font-medium">Email</label>
                                    <Input type="email" placeholder="john@example.com" />
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm font-medium">Message</label>
                                    <Textarea placeholder="How can we help you?" className="min-h-[120px]" />
                                </div>
                                <Button className="w-full sm:w-auto">Send Message</Button>
                            </form>
                        </CardContent>
                    </Card>
                </div>
            </div>
        </div>
    );
};

export default Contact;
