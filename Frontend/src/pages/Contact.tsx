import { Mail, Github, Linkedin, Send, ShieldCheck, Clock } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";
import { useToast } from "@/hooks/use-toast";
import { useState } from "react";

export const Contact = () => {
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [email, setEmail] = useState("");
  const [subject, setSubject] = useState("");
  const [message, setMessage] = useState("");
  const { toast } = useToast();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !message) {
      toast({ title: "Incomplete Form", description: "Please enter your email and message.", variant: "destructive" });
      return;
    }

    const mailtoSubject = encodeURIComponent(subject || "GetReport Sales & Support Inquiry");
    const mailtoBody = encodeURIComponent(
      `Name: ${firstName} ${lastName}\nEmail: ${email}\n\nMessage:\n${message}`
    );
    window.location.href = `mailto:vutikurishanmukha@gmail.com?subject=${mailtoSubject}&body=${mailtoBody}`;
    
    toast({ title: "Message Triggered", description: "Opening your default email app..." });
    setFirstName("");
    setLastName("");
    setEmail("");
    setSubject("");
    setMessage("");
  };

  return (
    <div className="min-h-screen flex flex-col bg-background animate-in fade-in duration-500">
      <Header onReset={() => {}} showReset={false} />

      <main className="flex-1 pt-20">
        {/* Header */}
        <div className="border-b border-border/60 bg-gradient-to-b from-muted/30 to-background py-16 md:py-24">
          <div className="container mx-auto px-4 text-center space-y-4 max-w-4xl">
            <Badge variant="outline" className="font-mono text-xs uppercase tracking-wider text-primary border-primary/30 px-3 py-1">
              Sales & Enterprise Support
            </Badge>
            <h1 className="text-4xl sm:text-5xl md:text-6xl font-display font-extrabold text-foreground tracking-tight uppercase leading-[1.05]">
              Get in Touch with Our Team.
            </h1>
            <p className="text-base sm:text-lg text-muted-foreground max-w-2xl mx-auto leading-relaxed">
              Have questions about multi-dataset joins, custom threshold controls, or on-premise Docker deployments? We'd love to assist.
            </p>
          </div>
        </div>

        {/* Form & Support Cards Grid */}
        <div className="container mx-auto px-4 py-16 max-w-7xl">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            
            {/* Left Col: Info Cards (Col 1-4) */}
            <div className="lg:col-span-4 space-y-6">
              <Card className="border border-border bg-card shadow-premium p-6 rounded-2xl space-y-4">
                <div className="flex items-center gap-3">
                  <div className="p-2.5 rounded-xl bg-primary/10 text-primary">
                    <Mail className="h-5 w-5" />
                  </div>
                  <div>
                    <h3 className="font-display font-bold text-base text-foreground">Direct Email</h3>
                    <p className="text-xs text-muted-foreground">Engineering & Support</p>
                  </div>
                </div>
                <div className="pt-2 font-mono text-xs text-foreground bg-muted/60 p-3 rounded-xl border border-border/40 select-all">
                  vutikurishanmukha@gmail.com
                </div>
              </Card>

              <Card className="border border-border bg-card shadow-premium p-6 rounded-2xl space-y-4">
                <div className="flex items-center gap-3">
                  <div className="p-2.5 rounded-xl bg-primary/10 text-primary">
                    <Clock className="h-5 w-5" />
                  </div>
                  <div>
                    <h3 className="font-display font-bold text-base text-foreground">Response SLA</h3>
                    <p className="text-xs text-muted-foreground">Guaranteed turnaround</p>
                  </div>
                </div>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  We respond to all technical and sales inquiries within <strong className="text-foreground">24 business hours</strong>. For critical processing issues, include "URGENT" in subject line.
                </p>
              </Card>

              <Card className="border border-border bg-card shadow-premium p-6 rounded-2xl space-y-4">
                <div className="flex items-center gap-3">
                  <div className="p-2.5 rounded-xl bg-primary/10 text-primary">
                    <ShieldCheck className="h-5 w-5 text-emerald-600" />
                  </div>
                  <div>
                    <h3 className="font-display font-bold text-base text-foreground">Social & Code</h3>
                    <p className="text-xs text-muted-foreground">Open repositories</p>
                  </div>
                </div>
                <div className="flex gap-3 pt-2">
                  <a
                    href="https://github.com/vutikurishanmukha9/GetReport"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex-1 py-2.5 rounded-xl bg-muted/60 border border-border/40 flex items-center justify-center gap-2 text-xs font-mono text-foreground hover:bg-primary/10 hover:text-primary transition-colors"
                  >
                    <Github className="h-4 w-4" />
                    <span>GitHub</span>
                  </a>
                  <a
                    href="https://linkedin.com/in/vutikurishanmukha9"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex-1 py-2.5 rounded-xl bg-muted/60 border border-border/40 flex items-center justify-center gap-2 text-xs font-mono text-foreground hover:bg-primary/10 hover:text-primary transition-colors"
                  >
                    <Linkedin className="h-4 w-4" />
                    <span>LinkedIn</span>
                  </a>
                </div>
              </Card>
            </div>

            {/* Right Col: Contact Form (Col 5-12) */}
            <div className="lg:col-span-8">
              <Card className="border border-border bg-card shadow-premium p-8 rounded-2xl">
                <div className="mb-6 space-y-2">
                  <h2 className="text-2xl font-display font-bold text-foreground uppercase tracking-tight">Send Us a Message</h2>
                  <p className="text-xs sm:text-sm text-muted-foreground">Fill out the fields below to send an instant message directly to our core engineering team.</p>
                </div>

                <form onSubmit={handleSubmit} className="space-y-5">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div className="space-y-1.5">
                      <label htmlFor="contact-first-name" className="text-xs font-medium text-foreground">First Name</label>
                      <Input
                        id="contact-first-name"
                        value={firstName}
                        onChange={(e) => setFirstName(e.target.value)}
                        placeholder="Jane"
                        className="rounded-xl border-border bg-white text-xs"
                      />
                    </div>
                    <div className="space-y-1.5">
                      <label htmlFor="contact-last-name" className="text-xs font-medium text-foreground">Last Name</label>
                      <Input
                        id="contact-last-name"
                        value={lastName}
                        onChange={(e) => setLastName(e.target.value)}
                        placeholder="Doe"
                        className="rounded-xl border-border bg-white text-xs"
                      />
                    </div>
                  </div>

                  <div className="space-y-1.5">
                    <label htmlFor="contact-email" className="text-xs font-medium text-foreground">Email Address *</label>
                    <Input
                      id="contact-email"
                      required
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      placeholder="jane@company.com"
                      className="rounded-xl border-border bg-white text-xs"
                    />
                  </div>

                  <div className="space-y-1.5">
                    <label htmlFor="contact-subject" className="text-xs font-medium text-foreground">Subject</label>
                    <Input
                      id="contact-subject"
                      value={subject}
                      onChange={(e) => setSubject(e.target.value)}
                      placeholder="Enterprise Plan / Multi-Dataset Support"
                      className="rounded-xl border-border bg-white text-xs"
                    />
                  </div>

                  <div className="space-y-1.5">
                    <label htmlFor="contact-message" className="text-xs font-medium text-foreground">Message *</label>
                    <Textarea
                      id="contact-message"
                      required
                      rows={5}
                      value={message}
                      onChange={(e) => setMessage(e.target.value)}
                      placeholder="Tell us about your dataset workload, security requirements, or enterprise feature needs..."
                      className="rounded-xl border-border bg-white text-xs leading-relaxed"
                    />
                  </div>

                  <Button type="submit" size="lg" className="w-full rounded-xl h-12 font-display font-semibold text-sm shadow-premium">
                    <Send className="mr-2 h-4 w-4" />
                    <span>Send Message to Engineering</span>
                  </Button>
                </form>
              </Card>
            </div>

          </div>
        </div>

      </main>

      <Footer />
    </div>
  );
};

export default Contact;
