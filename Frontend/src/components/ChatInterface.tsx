import { useState, useRef, useEffect } from "react";
import { Send, Bot, User, RefreshCw, ChevronDown, ChevronUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { api } from "@/services/api";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

interface Message {
    id: string;
    role: "user" | "assistant";
    content: string;
    sources?: string[]; // Optional sources for citations
    timestamp: Date;
}

interface ChatInterfaceProps {
    taskId: string;
}

export const ChatInterface = ({ taskId }: ChatInterfaceProps) => {
    const [messages, setMessages] = useState<Message[]>([
        {
            id: "welcome",
            role: "assistant",
            content: "Hello! I've analyzed your data. You can ask me questions like 'What is the trend in sales?' or 'Are there any anomalies?'",
            timestamp: new Date(),
        },
    ]);
    const [input, setInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const scrollRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to bottom
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollIntoView({ behavior: "smooth" });
        }
    }, [messages]);

    const handleSend = async () => {
        if (!input.trim() || isLoading) return;

        const userMsg: Message = {
            id: Date.now().toString(),
            role: "user",
            content: input.trim(),
            timestamp: new Date(),
        };

        setMessages((prev) => [...prev, userMsg]);
        setInput("");
        setIsLoading(true);

        try {
            const response = await api.chatWithJob(taskId, userMsg.content);

            const assistantMsg: Message = {
                id: (Date.now() + 1).toString(),
                role: "assistant",
                content: response.answer, // Assuming backend returns { answer: "...", sources: [...] }
                sources: response.sources,
                timestamp: new Date(),
            };

            setMessages((prev) => [...prev, assistantMsg]);
        } catch (error) {

            toast.error("Failed to get answer. Please try again.");
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    return (
        <Card className="w-full max-w-4xl mx-auto border-2 shadow-lg bg-card/50 backdrop-blur-sm">
            <CardHeader className="border-b bg-muted/20 pb-4">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-primary/10 rounded-lg">
                        <Bot className="w-6 h-6 text-primary" />
                    </div>
                    <div>
                        <CardTitle>Chat with your Data</CardTitle>
                        <CardDescription>AI-powered insights based on your report</CardDescription>
                    </div>
                </div>
            </CardHeader>

            <CardContent className="p-0">
                <ScrollArea className="h-[500px] p-4">
                    <div className="space-y-6">
                        {messages.map((msg) => (
                            <div
                                key={msg.id}
                                className={cn(
                                    "flex gap-3 max-w-[85%]",
                                    msg.role === "user" ? "ml-auto flex-row-reverse" : "mr-auto"
                                )}
                            >
                                {/* Avatar */}
                                <Avatar className={cn("h-8 w-8 mt-1", msg.role === "assistant" ? "bg-primary/10 text-primary" : "bg-muted text-muted-foreground")}>
                                    {msg.role === "assistant" ? <Bot className="h-5 w-5" /> : <User className="h-5 w-5" />}
                                </Avatar>

                                {/* Message Bubble */}
                                <div className="space-y-2">
                                    <div
                                        className={cn(
                                            "p-3 rounded-lg text-sm shadow-sm leading-relaxed",
                                            msg.role === "user"
                                                ? "bg-primary text-primary-foreground rounded-tr-none"
                                                : "bg-muted/50 border rounded-tl-none"
                                        )}
                                    >
                                        {msg.content}
                                    </div>

                                    {/* Sources (for Assistant only) */}
                                    {msg.role === "assistant" && msg.sources && msg.sources.length > 0 && (
                                        <SourcesExpander sources={msg.sources} />
                                    )}

                                    <div className="text-[10px] text-muted-foreground px-1">
                                        {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                    </div>
                                </div>
                            </div>
                        ))}

                        {isLoading && (
                            <div className="flex gap-3 mr-auto max-w-[85%]">
                                <Avatar className="h-8 w-8 mt-1 bg-primary/10 text-primary">
                                    <Bot className="h-5 w-5" />
                                </Avatar>
                                <div className="bg-muted/50 border p-3 rounded-lg rounded-tl-none flex items-center gap-2">
                                    <RefreshCw className="h-3 w-3 animate-spin text-muted-foreground" />
                                    <span className="text-xs text-muted-foreground">Thinking...</span>
                                </div>
                            </div>
                        )}
                        <div ref={scrollRef} />
                    </div>
                </ScrollArea>

                {/* Input Area */}
                <div className="p-4 border-t bg-muted/10">
                    <div className="flex gap-2">
                        <Input
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={handleKeyDown}
                            placeholder="Ask a question about your data..."
                            className="flex-1 bg-background"
                            disabled={isLoading}
                        />
                        <Button onClick={handleSend} disabled={isLoading || !input.trim()} size="icon">
                            <Send className="h-4 w-4" />
                        </Button>
                    </div>
                </div>
            </CardContent>
        </Card>
    );
};

// Helper component for Sources
const SourcesExpander = ({ sources }: { sources: string[] }) => {
    const [expanded, setExpanded] = useState(false);

    // Filter out internal RAG debug info if any, keep meaningful chunks
    // Assuming sources are text snippets.
    if (!sources.length) return null;

    return (
        <div className="text-xs">
            <button
                onClick={() => setExpanded(!expanded)}
                className="flex items-center gap-1 text-muted-foreground hover:text-foreground transition-colors"
            >
                {expanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                {expanded ? "Hide Sources" : `View ${sources.length} Context Sources`}
            </button>

            {expanded && (
                <div className="mt-2 space-y-2 pl-2 border-l-2 border-primary/20">
                    {sources.map((src, i) => (
                        <div key={i} className="bg-background p-2 rounded border text-[11px] text-muted-foreground truncate hover:whitespace-normal transition-all">
                            {src}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};
