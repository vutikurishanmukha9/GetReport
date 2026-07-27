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
  suggested_followups?: string[];
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
      suggested_followups: [
        "What are the top quality issues in this dataset?",
        "Which variables share the strongest positive correlation?",
        "What data cleaning actions were applied?"
      ],
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

  const handleSendQuery = async (queryText: string) => {
    const trimmedInput = queryText.trim();
    if (!trimmedInput || isLoading) return;

    const query = trimmedInput.slice(0, 2000);

    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content: query,
      timestamp: new Date(),
    };

    const assistantMsgId = (Date.now() + 1).toString();
    const initialAssistantMsg: Message = {
      id: assistantMsgId,
      role: "assistant",
      content: "",
      sources: [],
      suggested_followups: [],
      timestamp: new Date(),
    };

    const formattedHistory = messages.map(m => ({ role: m.role, content: m.content }));

    setMessages((prev) => [...prev, userMsg, initialAssistantMsg]);
    setInput("");
    setIsLoading(true);

    api.streamChatWithJob(
      taskId,
      query,
      (token) => {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMsgId
              ? { ...msg, content: msg.content + token }
              : msg
          )
        );
      },
      (metadata) => {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMsgId
              ? {
                  ...msg,
                  sources: metadata.sources,
                  suggested_followups: metadata.suggested_followups,
                }
              : msg
          )
        );
      },
      () => {
        setIsLoading(false);
      },
      (error) => {
        console.error("Stream chat error:", error);
        toast({
          title: "Communication Failure",
          description: "Failed to stream response from analysis model. Please try again.",
          variant: "destructive",
        });
        setIsLoading(false);
      },
      formattedHistory
    );
  };

  const handleSend = () => handleSendQuery(input);

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
    <Card className="w-full max-w-4xl mx-auto border border-border bg-card shadow-premium rounded-2xl overflow-hidden">
      {/* Chat Title bar */}
      <CardHeader className="border-b border-border bg-muted/10 pb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary/10 rounded-xl">
              <Bot className="w-5 h-5 text-primary" />
            </div>
            <div>
              <CardTitle className="text-base font-display font-bold text-foreground tracking-tight flex items-center gap-1.5">
                <span>AI Data Companion</span>
                <Badge variant="secondary" className="text-[9px] px-2 py-0.5 rounded-full bg-primary/10 text-primary font-mono font-bold uppercase tracking-wider">Active Context</Badge>
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
          className="h-[400px] sm:h-[450px] overflow-y-auto p-3 sm:p-4 space-y-5 scrollbar-thin"
        >
          {messages.map((msg) => {
            const isBot = msg.role === "assistant";
            return (
              <div
                key={msg.id}
                className={cn(
                  "flex gap-2 sm:gap-3 max-w-[95%] sm:max-w-[85%] animate-in fade-in slide-in-from-bottom-2 duration-300",
                  isBot ? "mr-auto" : "ml-auto flex-row-reverse"
                )}
              >
                {/* Avatar Icon */}
                <Avatar className={cn(
                  "h-7 w-7 sm:h-8 sm:w-8 shrink-0 mt-0.5 border border-border bg-white text-primary shadow-sm"
                )}>
                  <AvatarFallback className="flex items-center justify-center bg-transparent">
                    {isBot ? <Bot className="h-3.5 w-3.5 sm:h-4 sm:w-4" /> : <User className="h-3.5 w-3.5 sm:h-4 sm:w-4" />}
                  </AvatarFallback>
                </Avatar>

                {/* Bubble container */}
                <div className="space-y-2 max-w-full">
                  <div
                    className={cn(
                      "p-3.5 sm:p-4 rounded-2xl text-xs sm:text-sm leading-relaxed shadow-premium",
                      isBot
                        ? "bg-white border border-border/80 text-foreground rounded-tl-xs font-sans"
                        : "bg-primary text-primary-foreground rounded-tr-xs font-sans shadow-md"
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

                  {/* Interactive Suggested Follow-Up Prompt Chips (Bot only) */}
                  {isBot && msg.suggested_followups && msg.suggested_followups.length > 0 && (
                    <div className="pt-1.5 space-y-1.5 animate-in fade-in slide-in-from-top-1 duration-200">
                      <div className="text-[10px] font-mono font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1">
                        <Sparkles className="h-3 w-3 text-primary" />
                        <span>Suggested Follow-Ups</span>
                      </div>
                      <div className="flex flex-wrap gap-1.5">
                        {msg.suggested_followups.map((qText, qIdx) => (
                          <button
                            key={qIdx}
                            onClick={() => handleSendQuery(qText)}
                            disabled={isLoading}
                            className="text-left text-xs font-sans bg-white hover:bg-primary/5 border border-border/80 hover:border-primary/40 text-foreground rounded-xl px-3.5 py-1.5 transition-all duration-150 shadow-2xs hover:shadow-xs active:scale-95 disabled:opacity-50 cursor-pointer"
                          >
                            {qText}
                          </button>
                        ))}
                      </div>
                    </div>
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
              <Avatar className="h-8 w-8 mt-0.5 shrink-0 bg-white border border-border/80 text-primary shadow-2xs">
                <AvatarFallback className="flex items-center justify-center bg-transparent">
                  <Bot className="h-4 w-4 text-primary" />
                </AvatarFallback>
              </Avatar>
              <div className="bg-white border border-border/80 p-3.5 rounded-2xl rounded-tl-xs flex items-center gap-2 shadow-2xs">
                <RefreshCw className="h-3.5 w-3.5 animate-spin text-primary" />
                <span className="text-xs font-mono text-muted-foreground">Synthesizing response…</span>
              </div>
            </div>
          )}
          <div ref={scrollRef} />
        </div>

        {/* Input Textbox bar */}
        <div className="p-4 border-t border-border/60 bg-muted/10">
          <div className="flex items-center gap-2 bg-white border border-border/80 focus-within:border-primary/40 focus-within:ring-2 focus-within:ring-primary/20 rounded-2xl p-1.5 transition-all duration-200 shadow-2xs">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about trends, anomaly causes, schema corrections…"
              className="flex-1 bg-transparent border-0 focus-visible:ring-0 focus-visible:ring-offset-0 text-xs sm:text-sm placeholder:text-muted-foreground/60 h-10 px-3 font-sans"
              disabled={isLoading}
              maxLength={2000}
            />
            <Button 
              onClick={handleSend} 
              disabled={isLoading || !input.trim()} 
              size="icon"
              className="h-9 w-9 rounded-xl shadow-premium bg-primary text-primary-foreground hover:bg-primary/90 shrink-0 transition-all duration-150 active:scale-90 cursor-pointer disabled:opacity-40"
            >
              <Send className="h-4 w-4" />
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
        <div className="mt-2 space-y-2 pl-3 border-l border-border animate-in slide-in-from-top-1 duration-200">
          {sources.map((src, i) => {
            const isHighlighted = highlightedIdx === i;
            return (
              <div 
                key={i} 
                onClick={() => onSelectSource(i)}
                className={cn(
                  "p-2.5 rounded-lg border text-[11px] font-mono leading-relaxed transition-all duration-200 cursor-pointer select-none",
                  isHighlighted 
                    ? "bg-primary/5 border-primary text-foreground ring-1 ring-primary/20 scale-[1.01]" 
                    : "bg-muted/30 border-border text-muted-foreground hover:text-foreground hover:border-border"
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
