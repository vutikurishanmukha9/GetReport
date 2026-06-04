import { useState, useRef, useEffect } from "react";
import { Send, Bot, User, RefreshCw, ChevronDown, ChevronUp, Sparkles, BookOpen, Quote } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { api } from "@/services/api";
import { useToast } from "@/hooks/use-toast";
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
  const { toast } = useToast();
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content: "Hello! I’ve completed the data audit and created the full PDF report. Ask me anything about trends, quality flags, or correlation patterns you see in the dataset [1].",
      sources: ["Data Quality Engine: Column scores calculated, VIF check completed, and outliers detected."],
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [activeSourceId, setActiveSourceId] = useState<string | null>(null);
  const [highlightedSourceIdx, setHighlightedSourceIdx] = useState<number | null>(null);
  
  const scrollRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom of chat
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isLoading]);

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
        content: response.answer, 
        sources: response.sources,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (error) {
      console.error("Chat error:", error);
      toast({
        title: "Communication Failure",
        description: "Failed to fetch response from analysis model. Please try again.",
        variant: "destructive",
      });
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

  // Parses text to extract [1], [2] etc. and renders them as clickable superscript badges
  const renderMessageContent = (msg: Message) => {
    const { content, sources, id: msgId } = msg;
    if (!sources || sources.length === 0) return <span>{content}</span>;

    const parts = content.split(/(\[\d+\])/g);
    return parts.map((part, index) => {
      const match = part.match(/^\[(\d+)\]$/);
      if (match) {
        const num = parseInt(match[1], 10);
        const sourceIdx = num - 1;
        const hasSource = sourceIdx >= 0 && sourceIdx < sources.length;
        
        return (
          <sup key={index} className="mx-0.5 select-none align-baseline">
            <button
              onClick={() => {
                if (hasSource) {
                  setActiveSourceId(msgId);
                  setHighlightedSourceIdx(sourceIdx);
                  toast({
                    title: `Footnote Reference [${num}]`,
                    description: sources[sourceIdx].substring(0, 80) + "…",
                  });
                }
              }}
              className="inline-flex items-center justify-center bg-primary/10 text-primary hover:bg-primary/25 border border-primary/20 rounded h-4 min-w-4 px-1 text-[9px] font-bold font-mono transition-all duration-150 active:scale-90"
              title={hasSource ? sources[sourceIdx] : `Footnote ${num}`}
            >
              {num}
            </button>
          </sup>
        );
      }
      return <span key={index}>{part}</span>;
    });
  };

  return (
    <Card className="w-full max-w-4xl mx-auto border border-border bg-card/45 backdrop-blur-sm shadow-premium rounded-2xl overflow-hidden">
      {/* Chat Title bar */}
      <CardHeader className="border-b border-border/55 bg-muted/10 pb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary/10 rounded-xl">
              <Bot className="w-5 h-5 text-primary" />
            </div>
            <div>
              <CardTitle className="text-base font-display font-bold text-foreground tracking-tight flex items-center gap-1.5">
                <span>AI Data Companion</span>
                <Badge variant="secondary" className="text-[9px] px-1.5 py-0 rounded bg-primary/10 text-primary font-mono font-bold uppercase tracking-wider">Active Context</Badge>
              </CardTitle>
              <CardDescription className="text-xs">Ask questions and extract insights directly from the compiled dataset</CardDescription>
            </div>
          </div>
          <Sparkles className="h-4.5 w-4.5 text-primary/40 animate-pulse hidden sm:block" />
        </div>
      </CardHeader>

      <CardContent className="p-0 flex flex-col">
        {/* Messages list */}
        <div 
          ref={chatContainerRef}
          className="h-[420px] overflow-y-auto p-4 space-y-6 scrollbar-thin"
        >
          {messages.map((msg) => {
            const isBot = msg.role === "assistant";
            return (
              <div
                key={msg.id}
                className={cn(
                  "flex gap-3 max-w-[85%] animate-in fade-in slide-in-from-bottom-2 duration-300",
                  isBot ? "mr-auto" : "ml-auto flex-row-reverse"
                )}
              >
                {/* Avatar Icon */}
                <Avatar className={cn(
                  "h-8 w-8 shrink-0 mt-0.5 border shadow-sm",
                  isBot 
                    ? "bg-background border-border text-primary" 
                    : "bg-primary text-primary-foreground border-transparent"
                )}>
                  <AvatarFallback className="flex items-center justify-center bg-transparent">
                    {isBot ? <Bot className="h-4 w-4" /> : <User className="h-4 w-4" />}
                  </AvatarFallback>
                </Avatar>

                {/* Bubble container */}
                <div className="space-y-2.5 max-w-full">
                  <div
                    className={cn(
                      "p-3.5 rounded-2xl text-xs sm:text-sm leading-relaxed shadow-sm",
                      isBot
                        ? "bg-muted dark:bg-card border border-border/80 text-foreground rounded-tl-sm font-sans"
                        : "bg-primary text-primary-foreground rounded-tr-sm font-sans"
                    )}
                  >
                    {renderMessageContent(msg)}
                  </div>

                  {/* Context Citations Drawer (Bot only) */}
                  {isBot && msg.sources && msg.sources.length > 0 && (
                    <SourcesExpander 
                      messageId={msg.id}
                      sources={msg.sources} 
                      activeSourceId={activeSourceId}
                      highlightedIdx={msg.id === activeSourceId ? highlightedSourceIdx : null}
                      onSelectSource={(idx) => {
                        setActiveSourceId(msg.id);
                        setHighlightedSourceIdx(idx);
                      }}
                    />
                  )}

                  {/* Timestamp */}
                  <div className={cn(
                    "text-[9px] font-mono text-muted-foreground px-1.5",
                    !isBot && "text-right"
                  )}>
                    {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                </div>
              </div>
            );
          })}

          {/* Loader typing block */}
          {isLoading && (
            <div className="flex gap-3 mr-auto max-w-[85%] animate-pulse">
              <Avatar className="h-8 w-8 mt-0.5 shrink-0 bg-background border border-border text-primary">
                <AvatarFallback className="flex items-center justify-center bg-transparent">
                  <Bot className="h-4 w-4" />
                </AvatarFallback>
              </Avatar>
              <div className="bg-muted dark:bg-card border border-border/80 p-3.5 rounded-2xl rounded-tl-sm flex items-center gap-2">
                <RefreshCw className="h-3.5 w-3.5 animate-spin text-primary" />
                <span className="text-xs font-mono text-muted-foreground">Synthesizing response…</span>
              </div>
            </div>
          )}
          <div ref={scrollRef} />
        </div>

        {/* Input Textbox bar */}
        <div className="p-4 border-t border-border/55 bg-muted/10">
          <div className="flex items-center gap-2 bg-background border border-border/80 focus-within:border-primary/40 focus-within:ring-1 focus-within:ring-primary/40 rounded-xl p-1.5 transition-all duration-200">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about trends, anomaly causes, schema corrections…"
              className="flex-1 bg-transparent border-0 focus-visible:ring-0 focus-visible:ring-offset-0 text-xs sm:text-sm placeholder:text-muted-foreground/60 h-9"
              disabled={isLoading}
            />
            <Button 
              onClick={handleSend} 
              disabled={isLoading || !input.trim()} 
              size="icon"
              className="h-8 w-8 rounded-lg shadow-premium shrink-0 transition-all duration-150 active:scale-95"
            >
              <Send className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

interface SourcesExpanderProps {
  messageId: string;
  sources: string[];
  activeSourceId: string | null;
  highlightedIdx: number | null;
  onSelectSource: (idx: number) => void;
}

const SourcesExpander = ({ 
  messageId, 
  sources, 
  activeSourceId, 
  highlightedIdx, 
  onSelectSource 
}: SourcesExpanderProps) => {
  const [expanded, setExpanded] = useState(false);

  if (!sources || sources.length === 0) return null;

  return (
    <div className="text-xs space-y-1">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-muted-foreground hover:text-foreground font-mono text-[10px] tracking-tight transition-colors duration-150 uppercase"
      >
        {expanded ? <ChevronUp className="h-3 w-3 text-primary" /> : <ChevronDown className="h-3 w-3 text-primary" />}
        <BookOpen className="h-3 w-3" />
        <span>{expanded ? "Hide Audit Citations" : `View Citations (${sources.length})`}</span>
      </button>

      {expanded && (
        <div className="mt-2 space-y-2 pl-3 border-l border-border/60 animate-in slide-in-from-top-1 duration-200">
          {sources.map((src, i) => {
            const isHighlighted = highlightedIdx === i;
            return (
              <div 
                key={i} 
                onClick={() => onSelectSource(i)}
                className={cn(
                  "p-2.5 rounded-lg border text-[11px] font-mono leading-relaxed transition-all duration-200 cursor-pointer select-none",
                  isHighlighted 
                    ? "bg-primary/5 border-primary/40 text-foreground ring-1 ring-primary/20 scale-[1.01]" 
                    : "bg-muted/80 dark:bg-card/85 border-border/60 text-muted-foreground hover:text-foreground hover:border-border"
                )}
              >
                <div className="flex items-center gap-1.5 font-bold text-[9px] uppercase tracking-wider text-primary mb-1">
                  <Quote className="h-2.5 w-2.5" />
                  <span>Citation Reference [{i + 1}]</span>
                </div>
                <div className="line-clamp-3 hover:line-clamp-none transition-all duration-300">
                  {src}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
