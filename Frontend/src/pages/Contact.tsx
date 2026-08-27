import { useState } from "react";
import { 
  Mail, Linkedin, Send, Clock, 
  Copy, Check, MessageSquare, ArrowRight
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";
import { useToast } from "@/hooks/use-toast";

const TOPICS = [
  "Technical Support",
  "Feature Request",
  "On-Premise Docker Deployment",
  "Security / Bug Bounty",
  "General Inquiry"
] as const;

export const Contact = () => {
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [email, setEmail] = useState("");
  const [selectedTopic, setSelectedTopic] = useState<string>("Technical Support");
  const [subject, setSubject] = useState("");
  const [message, setMessage] = useState("");
  const [copiedEmail, setCopiedEmail] = useState(false);
  const { toast } = useToast();

  const handleCopyEmail = () => {
    navigator.clipboard.writeText("vutikurishanmukha@gmail.com");
    setCopiedEmail(true);
    toast({ title: "Email Copied", description: "vutikurishanmukha@gmail.com copied to clipboard." });
    setTimeout(() => setCopiedEmail(false), 2000);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !message) {
      toast({ title: "Incomplete Form", description: "Please enter your email and message.", variant: "destructive" });
      return;
    }

    const mailtoSubject = encodeURIComponent(`[${selectedTopic}] ${subject || "GetReport Inquiry"}`);
    const mailtoBody = encodeURIComponent(
      `Name: ${firstName} ${lastName}\nEmail: ${email}\nTopic: ${selectedTopic}\n\nMessage:\n${message}`
    );
    window.location.href = `mailto:vutikurishanmukha@gmail.com?subject=${mailtoSubject}&body=${mailtoBody}`;
    
    toast({ title: "Message Triggered", description: "Opening your default email client..." });
    setFirstName("");
    setLastName("");
    setEmail("");
    setSubject("");
    setMessage("");
  };

  return (
    <div className="min-h-screen flex flex-col bg-background animate-in fade-in duration-500">
      <Header onReset={() => {}} showReset={false} />

      <main className="flex-1 pt-16 sm:pt-20">
        {/* Header */}
        <div className="border-b border-border/60 bg-gradient-to-b from-muted/20 via-background to-background py-8 sm:py-12">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-5xl text-center space-y-3 sm:space-y-4">
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-primary/10 text-primary text-xs font-semibold uppercase tracking-wider font-mono border border-primary/20 t-badge-shimmer">
              <MessageSquare className="h-3.5 w-3.5" />
              <span>Engineering & Community Support</span>
            </div>

            <h1 className="text-3xl sm:text-4xl md:text-5xl font-display font-extrabold text-foreground tracking-tight uppercase leading-[1.08]">
              Get in Touch with Our Team.
            </h1>
            
            <p className="text-sm sm:text-base text-muted-foreground max-w-2xl mx-auto leading-relaxed font-sans">
              Have questions regarding custom Polars transformation DAGs, on-premise Docker deployments, or security audits? We are here to help.
            </p>
          </div>
        </div>

        {/* Section: Main Contact Grid */}
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-10 max-w-7xl">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8 items-start">
            
            {/* Left Column: Direct Info Cards (Col 1-5) */}
            <div className="lg:col-span-5 space-y-4">
              
              {/* Direct Email Card with Copy Button */}
              <Card className="border border-border bg-card shadow-premium p-4 sm:p-5 rounded-2xl space-y-3 t-card-lift">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2.5">
                    <div className="p-2 rounded-xl bg-primary/10 text-primary">
                      <Mail className="h-4 w-4" />
                    </div>
                    <div>
                      <h3 className="font-display font-bold text-sm sm:text-base text-foreground">Direct Engineering Inbox</h3>
                      <p className="text-xs text-muted-foreground font-sans">Fast-response developer inbox</p>
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between gap-2 p-2.5 bg-muted/40 rounded-xl border border-border/60 font-mono text-xs text-foreground select-all">
                  <span className="truncate">vutikurishanmukha@gmail.com</span>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={handleCopyEmail}
                    className="h-7 px-2 font-mono text-[11px] gap-1 shrink-0"
                  >
                    {copiedEmail ? <Check className="h-3 w-3 text-emerald-600" /> : <Copy className="h-3 w-3" />}
                    <span>{copiedEmail ? "Copied" : "Copy"}</span>
                  </Button>
                </div>
              </Card>

              {/* Response SLA Card */}
              <Card className="border border-border bg-card shadow-premium p-4 sm:p-5 rounded-2xl space-y-2 t-card-lift">
                <div className="flex items-center gap-2.5">
                  <div className="p-2 rounded-xl bg-emerald-500/10 text-emerald-700">
                    <Clock className="h-4 w-4" />
                  </div>
                  <div>
                    <h3 className="font-display font-bold text-sm sm:text-base text-foreground">Response SLA</h3>
                    <p className="text-xs text-muted-foreground font-sans">Guaranteed turnaround timeline</p>
                  </div>
                </div>
                <p className="text-xs text-muted-foreground font-sans leading-relaxed">
                  We reply to all inquiries within <strong className="text-foreground">24 business hours</strong>. For urgent processing or security issues, include &quot;URGENT&quot; in the subject line.
                </p>
              </Card>

              {/* Open Source Community Card */}
              <Card className="border border-border bg-card shadow-premium p-4 sm:p-5 rounded-2xl space-y-3 t-card-lift">
                <div className="flex items-center gap-2.5">
                  <div className="p-2 rounded-xl bg-blue-500/10 text-blue-700">
                    <MessageSquare className="h-4 w-4" />
                  </div>
                  <div>
                    <h3 className="font-display font-bold text-sm sm:text-base text-foreground">Open Source & Community</h3>
                    <p className="text-xs text-muted-foreground font-sans">Contribute, file issues, or star</p>
                  </div>
                </div>
                <div className="flex flex-col gap-2 font-mono text-xs">
                  <a
                    href="https://github.com/vutikurishanmukha9/GetReport"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center justify-between p-2.5 rounded-xl bg-muted/20 border border-border/40 hover:bg-muted/40 transition-colors"
                  >
                    <span className="truncate pr-2">github.com/vutikurishanmukha9/GetReport</span>
                    <ArrowRight className="h-3.5 w-3.5 text-primary shrink-0" />
                  </a>
                  <a
                    href="https://www.linkedin.com/in/vutikuri-shanmukha-sai-19946824a"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center justify-between p-2.5 rounded-xl bg-muted/20 border border-border/40 hover:bg-muted/40 transition-colors"
                  >
                    <span>LinkedIn Profile</span>
                    <Linkedin className="h-3.5 w-3.5 text-blue-600" />
                  </a>
                </div>
              </Card>

            </div>

            {/* Right Column: Contact Form (Col 6-12) */}
            <div className="lg:col-span-7">
              <Card className="border border-border bg-card shadow-premium rounded-2xl sm:rounded-3xl p-5 sm:p-7 space-y-4 sm:space-y-5">
                <div className="space-y-1 border-b border-border/60 pb-3">
                  <h2 className="text-lg sm:text-xl font-display font-bold text-foreground uppercase tracking-tight">
                    Send a Message
                  </h2>
                  <p className="text-xs text-muted-foreground font-sans">
                    Fill out the form below and we will route your inquiry to the appropriate engineering team.
                  </p>
                </div>

                <form onSubmit={handleSubmit} className="space-y-4">
                  {/* Topic Pill Selector */}
                  <div className="space-y-1.5">
                    <span id="topic-label" className="text-xs font-mono font-bold uppercase tracking-wider text-muted-foreground block">
                      Inquiry Topic:
                    </span>
                    <div role="group" aria-labelledby="topic-label" className="flex flex-wrap gap-1.5">
                      {TOPICS.map((topic) => (
                        <button
                          key={topic}
                          type="button"
                          onClick={() => setSelectedTopic(topic)}
                          className={`px-3 py-1 rounded-xl text-xs font-mono transition-all cursor-pointer border ${
                            selectedTopic === topic
                              ? "bg-primary text-primary-foreground border-primary shadow-xs font-bold"
                              : "bg-muted/30 text-muted-foreground border-border hover:bg-muted hover:text-foreground"
                          }`}
                        >
                          {topic}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Name Inputs */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <div className="space-y-1">
                      <label htmlFor="firstName" className="text-xs font-mono font-medium text-foreground block">First Name</label>
                      <Input
                        id="firstName"
                        value={firstName}
                        onChange={(e) => setFirstName(e.target.value)}
                        placeholder="Ada"
                        className="rounded-xl border-border bg-muted/10 text-xs sm:text-sm h-9"
                      />
                    </div>
                    <div className="space-y-1">
                      <label htmlFor="lastName" className="text-xs font-mono font-medium text-foreground block">Last Name</label>
                      <Input
                        id="lastName"
                        value={lastName}
                        onChange={(e) => setLastName(e.target.value)}
                        placeholder="Lovelace"
                        className="rounded-xl border-border bg-muted/10 text-xs sm:text-sm h-9"
                      />
                    </div>
                  </div>

                  {/* Email & Subject */}
                  <div className="space-y-1">
                    <label htmlFor="email" className="text-xs font-mono font-medium text-foreground block">Your Email *</label>
                    <Input
                      id="email"
                      type="email"
                      required
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      placeholder="ada@example.com"
                      className="rounded-xl border-border bg-muted/10 text-xs sm:text-sm h-9"
                    />
                  </div>

                  <div className="space-y-1">
                    <label htmlFor="subject" className="text-xs font-mono font-medium text-foreground block">Subject</label>
                    <Input
                      id="subject"
                      value={subject}
                      onChange={(e) => setSubject(e.target.value)}
                      placeholder="e.g. Custom Polars Transformation Rule Question"
                      className="rounded-xl border-border bg-muted/10 text-xs sm:text-sm h-9"
                    />
                  </div>

                  {/* Message */}
                  <div className="space-y-1">
                    <label htmlFor="message" className="text-xs font-mono font-medium text-foreground block">Message *</label>
                    <Textarea
                      id="message"
                      required
                      rows={4}
                      value={message}
                      onChange={(e) => setMessage(e.target.value)}
                      placeholder="Describe your dataset requirements or technical inquiry in detail..."
                      className="rounded-xl border-border bg-muted/10 text-xs sm:text-sm resize-none"
                    />
                  </div>

                  {/* Submit Button */}
                  <Button
                    type="submit"
                    size="lg"
                    className="w-full h-11 rounded-xl shadow-premium t-card-lift t-spring-press font-display font-semibold text-sm gap-2"
                  >
                    <Send className="h-4 w-4" />
                    <span>Send Inquiry</span>
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
