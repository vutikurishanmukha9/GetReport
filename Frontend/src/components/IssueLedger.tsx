import React, { useState, useMemo, useCallback } from 'react';
import { 
    Check, 
    X, 
    AlertTriangle, 
    AlertCircle, 
    Info, 
    Lock, 
    CheckCheck, 
    XCircle, 
    ArrowRight, 
    Search, 
    Code2, 
    ShieldCheck, 
    Copy, 
    Check as CheckIcon,
    ChevronDown,
    ChevronUp,
    Sparkles
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { api } from '@/services/api';
import { cn } from '@/lib/utils';

// Types
export interface Issue {
    id: string;
    issue_type: string;
    severity: 'critical' | 'high' | 'medium' | 'low';
    column: string | null;
    affected_rows: number;
    affected_pct: number;
    description: string;
    suggested_fix: string;
    fix_code: string;
    status: 'pending' | 'approved' | 'rejected' | 'modified';
    user_note: string;
}

export interface IssueLedgerData {
    issues: Issue[];
    summary: {
        pending: number;
        approved: number;
        rejected: number;
        modified: number;
        total: number;
    };
    locked: boolean;
    locked_at: string | null;
}

export interface IssueLedgerProps {
    taskId: string;
    data: IssueLedgerData;
    onRefresh: () => void;
    onProceed: () => void;
}

// Severity badge styles with refined luminous dots
const severityConfig = {
    critical: {
        badge: 'bg-rose-500/10 text-rose-300 border-rose-500/30 shadow-[0_0_12px_-3px_rgba(244,63,94,0.3)]',
        dot: 'bg-rose-400 shadow-[0_0_8px_rgba(244,63,94,0.8)]',
        iconBg: 'bg-rose-500/15 border-rose-500/30 text-rose-400',
    },
    high: {
        badge: 'bg-orange-500/10 text-orange-300 border-orange-500/30 shadow-[0_0_12px_-3px_rgba(249,115,22,0.3)]',
        dot: 'bg-orange-400 shadow-[0_0_8px_rgba(249,115,22,0.8)]',
        iconBg: 'bg-orange-500/15 border-orange-500/30 text-orange-400',
    },
    medium: {
        badge: 'bg-amber-500/10 text-amber-300 border-amber-500/30 shadow-[0_0_12px_-3px_rgba(245,158,11,0.25)]',
        dot: 'bg-amber-400 shadow-[0_0_8px_rgba(245,158,11,0.8)]',
        iconBg: 'bg-amber-500/15 border-amber-500/30 text-amber-400',
    },
    low: {
        badge: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30 shadow-[0_0_12px_-3px_rgba(16,185,129,0.25)]',
        dot: 'bg-emerald-400 shadow-[0_0_8px_rgba(16,185,129,0.8)]',
        iconBg: 'bg-emerald-500/15 border-emerald-500/30 text-emerald-400',
    },
};

const issueTypeLabels: Record<string, string> = {
    missing_values: 'Missing Values',
    duplicates: 'Duplicates',
    type_mismatch: 'Type Mismatch',
    outliers: 'Outliers',
    multivariate_anomaly: 'Multivariate Anomaly',
    format_issue: 'Format Issue',
    high_cardinality: 'High Cardinality',
    empty_column: 'Empty Column',
    constant_column: 'Constant Column',
    encoding_issue: 'Encoding Issue',
};

function formatIssueType(type: string): string {
    if (type in issueTypeLabels) {
        return issueTypeLabels[type];
    }
    return type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

export const IssueLedger: React.FC<IssueLedgerProps> = ({
    taskId,
    data,
    onRefresh,
    onProceed
}) => {
    const [loading, setLoading] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [optimisticIssues, setOptimisticIssues] = useState<Issue[] | null>(null);
    const [filterStatus, setFilterStatus] = useState<'all' | 'pending' | 'approved' | 'rejected'>('all');
    const [searchQuery, setSearchQuery] = useState<string>('');
    const [expandedCodeId, setExpandedCodeId] = useState<string | null>(null);
    const [copiedId, setCopiedId] = useState<string | null>(null);

    // Sync optimistic state with incoming props
    const issuesList = optimisticIssues || data.issues;

    // Derived summary counts for instant UI reactivity
    const summary = useMemo(() => {
        const approved = issuesList.filter(i => i.status === 'approved').length;
        const rejected = issuesList.filter(i => i.status === 'rejected').length;
        const modified = issuesList.filter(i => i.status === 'modified').length;
        const pending = issuesList.filter(i => i.status === 'pending').length;
        return {
            approved,
            rejected,
            modified,
            pending,
            total: issuesList.length
        };
    }, [issuesList]);

    const reviewedCount = summary.approved + summary.rejected;
    const progressPct = summary.total > 0 ? Math.round((reviewedCount / summary.total) * 100) : 100;
    const hasPending = summary.pending > 0;
    const locked = data.locked;

    // Filter issues based on active tab and search query
    const filteredIssues = useMemo(() => {
        return issuesList.filter(issue => {
            // Status tab match
            if (filterStatus !== 'all' && issue.status !== filterStatus) {
                return false;
            }
            // Search query match
            if (searchQuery.trim()) {
                const q = searchQuery.toLowerCase().trim();
                const colMatch = (issue.column || 'all_rows').toLowerCase().includes(q);
                const typeMatch = formatIssueType(issue.issue_type).toLowerCase().includes(q);
                const descMatch = issue.description.toLowerCase().includes(q);
                const fixMatch = issue.suggested_fix.toLowerCase().includes(q);
                return colMatch || typeMatch || descMatch || fixMatch;
            }
            return true;
        });
    }, [issuesList, filterStatus, searchQuery]);

    // Handle single issue approval
    const handleApprove = async (issueId: string) => {
        setLoading(issueId);
        setError(null);

        const previous = issuesList;
        setOptimisticIssues(prev =>
            (prev || data.issues).map(i => (i.id === issueId ? { ...i, status: 'approved' as const } : i))
        );

        try {
            await api.approveIssue(taskId, issueId);
            onRefresh();
        } catch (e) {
            setOptimisticIssues(previous);
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            setLoading(null);
        }
    };

    // Handle single issue rejection
    const handleReject = async (issueId: string) => {
        setLoading(issueId);
        setError(null);

        const previous = issuesList;
        setOptimisticIssues(prev =>
            (prev || data.issues).map(i => (i.id === issueId ? { ...i, status: 'rejected' as const } : i))
        );

        try {
            await api.rejectIssue(taskId, issueId);
            onRefresh();
        } catch (e) {
            setOptimisticIssues(previous);
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            setLoading(null);
        }
    };

    // Handle Approve All
    const handleApproveAll = async () => {
        setLoading('all-approve');
        setError(null);

        const previous = issuesList;
        setOptimisticIssues(prev =>
            (prev || data.issues).map(i => (i.status === 'pending' ? { ...i, status: 'approved' as const } : i))
        );

        try {
            await api.approveAllIssues(taskId);
            onRefresh();
        } catch (e) {
            setOptimisticIssues(previous);
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            setLoading(null);
        }
    };

    // Handle Reject All
    const handleRejectAll = async () => {
        setLoading('all-reject');
        setError(null);

        const previous = issuesList;
        setOptimisticIssues(prev =>
            (prev || data.issues).map(i => (i.status === 'pending' ? { ...i, status: 'rejected' as const } : i))
        );

        try {
            await api.rejectAllIssues(taskId);
            onRefresh();
        } catch (e) {
            setOptimisticIssues(previous);
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            setLoading(null);
        }
    };

    // Handle Lock & Proceed
    const handleLockAndProceed = async () => {
        setLoading('lock');
        setError(null);
        try {
            await api.lockIssues(taskId);
            onProceed();
        } catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            setLoading(null);
        }
    };

    // Copy fix code snippet to clipboard
    const handleCopyCode = useCallback((id: string, code: string) => {
        navigator.clipboard.writeText(code);
        setCopiedId(id);
        setTimeout(() => setCopiedId(null), 2000);
    }, []);

    // Empty state: pristine clean dataset
    if (issuesList.length === 0) {
        return (
            <div className="relative rounded-[2rem] p-1.5 sm:p-2 bg-gradient-to-b from-white/[0.12] via-white/[0.04] to-white/[0.01] border border-white/10 shadow-[0_24px_70px_-15px_rgba(0,0,0,0.85)] backdrop-blur-2xl max-w-2xl mx-auto overflow-hidden animate-in fade-in duration-300">
                <div className="relative rounded-[calc(2rem-0.5rem)] bg-[#090a10]/95 border border-white/[0.06] p-8 text-center overflow-hidden">
                    {/* Ambient Glow */}
                    <div className="absolute -top-24 left-1/2 -translate-x-1/2 w-64 h-32 bg-emerald-500/20 blur-3xl pointer-events-none rounded-full" />
                    
                    <div className="relative z-10 flex flex-col items-center">
                        <div className="w-16 h-16 rounded-2xl bg-gradient-to-b from-emerald-500/25 to-emerald-500/10 border border-emerald-500/30 flex items-center justify-center mb-5 shadow-[0_0_30px_rgba(16,185,129,0.35),inset_0_1px_0_rgba(255,255,255,0.2)]">
                            <ShieldCheck className="w-8 h-8 text-emerald-300" />
                        </div>
                        <span className="px-3 py-1 rounded-full text-[11px] font-mono uppercase tracking-widest font-semibold bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 mb-3">
                            Dataset Health 100%
                        </span>
                        <h3 className="text-2xl font-display font-bold text-white tracking-tight">
                            Data Quality: Pristine
                        </h3>
                        <p className="text-zinc-400 text-sm mt-2 max-w-md mx-auto leading-relaxed">
                            Every column passed automated verification across completeness, type consistency, and validity. No data fixes are required.
                        </p>
                    </div>

                    <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 gap-3 relative z-10 text-left">
                        {[
                            { title: 'Completeness', desc: 'Zero missing, masked, or null values detected.' },
                            { title: 'Consistency', desc: 'Uniform data types and aligned formatting.' },
                            { title: 'Uniqueness', desc: 'No duplicate records or redundant keys.' },
                            { title: 'Validity', desc: 'No extreme distribution outliers or anomalies.' }
                        ].map((dim, idx) => (
                            <div key={idx} className="bg-white/[0.03] border border-white/5 rounded-xl p-3.5 flex items-start gap-3">
                                <div className="w-6 h-6 rounded-md bg-emerald-500/15 border border-emerald-500/25 flex items-center justify-center shrink-0 mt-0.5">
                                    <Check className="w-3.5 h-3.5 text-emerald-300" />
                                </div>
                                <div>
                                    <h4 className="text-xs font-semibold text-white tracking-tight">{dim.title}</h4>
                                    <p className="text-[11px] text-zinc-400 mt-0.5 leading-snug">{dim.desc}</p>
                                </div>
                            </div>
                        ))}
                    </div>

                    <div className="mt-8 flex justify-center relative z-10">
                        <button
                            type="button"
                            onClick={onProceed}
                            className="group relative inline-flex items-center gap-2.5 px-8 py-3.5 rounded-xl text-sm font-bold bg-gradient-to-r from-emerald-500 via-teal-500 to-emerald-400 hover:from-emerald-400 hover:via-teal-400 hover:to-emerald-300 text-zinc-950 shadow-[0_0_35px_-5px_rgba(16,185,129,0.5),inset_0_1px_0_rgba(255,255,255,0.7)] border border-emerald-300/60 transition-all duration-200 active:scale-[0.98] cursor-pointer"
                        >
                            <span>Proceed to Analysis</span>
                            <div className="w-6 h-6 rounded-lg bg-black/15 flex items-center justify-center group-hover:translate-x-0.5 transition-transform">
                                <ArrowRight className="w-3.5 h-3.5 text-zinc-950 stroke-[2.5]" />
                            </div>
                        </button>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="relative rounded-[2rem] p-1.5 sm:p-2 bg-gradient-to-b from-white/[0.12] via-white/[0.04] to-white/[0.01] border border-white/10 shadow-[0_24px_70px_-15px_rgba(0,0,0,0.85)] backdrop-blur-2xl overflow-hidden animate-in fade-in duration-300">
            {/* Top Ambient Glow Sheen */}
            <div className="absolute -top-32 left-1/2 -translate-x-1/2 w-[32rem] h-48 bg-gradient-to-r from-emerald-500/10 via-amber-500/10 to-teal-500/10 blur-3xl pointer-events-none rounded-full" />

            <div className="relative rounded-[calc(2rem-0.5rem)] bg-[#090a10]/95 border border-white/[0.06] overflow-hidden">
                {/* ─── 1. COMPONENT HEADER ────────────────────────────────────────── */}
                <div className="border-b border-white/[0.06] px-5 sm:px-7 py-5 bg-gradient-to-b from-white/[0.03] to-transparent">
                    <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
                        {/* Title & Metadata */}
                        <div className="flex items-start gap-4">
                            <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-amber-500/20 via-amber-500/10 to-transparent border border-amber-500/30 flex items-center justify-center shrink-0 shadow-[0_0_25px_-5px_rgba(245,158,11,0.3),inset_0_1px_0_rgba(255,255,255,0.2)]">
                                <AlertTriangle className="w-6 h-6 text-amber-400 stroke-[2.2]" />
                            </div>
                            <div>
                                <div className="flex flex-wrap items-center gap-2.5">
                                    <h3 className="text-xl sm:text-2xl font-display font-bold text-white tracking-tight">
                                        Issue Ledger
                                    </h3>
                                    <span className="px-2.5 py-0.5 rounded-full text-[10px] font-mono uppercase tracking-wider font-semibold bg-white/[0.05] border border-white/10 text-zinc-300">
                                        Data Hygiene Gate
                                    </span>
                                    {locked && (
                                        <span className="px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 flex items-center gap-1">
                                            <Lock className="w-3 h-3 text-emerald-400" /> Locked
                                        </span>
                                    )}
                                </div>
                                <p className="text-zinc-400 text-xs sm:text-sm mt-1 font-sans">
                                    Review data fixes, approve automated cleaning operations, and lock dataset before compiling analysis.
                                </p>
                            </div>
                        </div>

                        {/* Top Batch Actions */}
                        {!locked && (
                            <div className="flex flex-wrap items-center gap-2.5 self-start lg:self-center shrink-0">
                                {/* Approve All Button */}
                                <button
                                    type="button"
                                    onClick={handleApproveAll}
                                    disabled={loading !== null || !hasPending}
                                    className={cn(
                                        "group relative inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold tracking-tight transition-all duration-200 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 select-none",
                                        hasPending
                                            ? "bg-gradient-to-r from-emerald-500/20 via-emerald-500/25 to-teal-500/20 hover:from-emerald-500 hover:via-emerald-400 hover:to-teal-400 text-emerald-300 hover:text-zinc-950 border border-emerald-500/40 hover:border-emerald-300 shadow-[0_0_20px_-3px_rgba(16,185,129,0.35),inset_0_1px_0_rgba(255,255,255,0.2)] active:scale-[0.98]"
                                            : "bg-white/[0.02] text-zinc-600 border border-white/5 cursor-not-allowed opacity-50 shadow-none"
                                    )}
                                    title="Approve all remaining pending issues"
                                >
                                    <div className={cn(
                                        "w-5 h-5 rounded-md flex items-center justify-center transition-colors",
                                        hasPending 
                                            ? "bg-emerald-500/25 group-hover:bg-zinc-950/20 text-emerald-300 group-hover:text-zinc-950" 
                                            : "bg-white/5 text-zinc-600"
                                    )}>
                                        <CheckCheck className="w-3.5 h-3.5 stroke-[2.3]" />
                                    </div>
                                    <span>Approve All</span>
                                    {hasPending && (
                                        <span className="px-1.5 py-0.5 rounded-full text-[10px] font-mono font-bold bg-emerald-500/30 group-hover:bg-zinc-950/20 text-emerald-200 group-hover:text-zinc-950">
                                            +{summary.pending}
                                        </span>
                                    )}
                                </button>

                                {/* Reject All Button */}
                                <button
                                    type="button"
                                    onClick={handleRejectAll}
                                    disabled={loading !== null || !hasPending}
                                    className={cn(
                                        "group relative inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold tracking-tight transition-all duration-200 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-400 select-none",
                                        hasPending
                                            ? "bg-gradient-to-r from-rose-500/10 via-rose-500/15 to-rose-500/10 hover:from-rose-500/25 hover:via-rose-500/30 hover:to-rose-500/20 text-rose-300 hover:text-rose-100 border border-rose-500/30 hover:border-rose-500/50 shadow-[inset_0_1px_0_rgba(255,255,255,0.08)] active:scale-[0.98]"
                                            : "bg-white/[0.02] text-zinc-600 border border-white/5 cursor-not-allowed opacity-50 shadow-none"
                                    )}
                                    title="Reject all remaining pending issues"
                                >
                                    <div className={cn(
                                        "w-5 h-5 rounded-md flex items-center justify-center transition-colors",
                                        hasPending ? "bg-rose-500/20 text-rose-300" : "bg-white/5 text-zinc-600"
                                    )}>
                                        <XCircle className="w-3.5 h-3.5 stroke-[2.3]" />
                                    </div>
                                    <span>Reject All</span>
                                </button>
                            </div>
                        )}
                    </div>
                </div>

                {/* ─── 2. TOOLBAR: TABS, SEARCH, AND PROGRESS ─────────────────────────── */}
                <div className="bg-white/[0.02] border-b border-white/[0.06] px-5 sm:px-7 py-3.5">
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-3.5">
                        {/* Segmented Filter Tabs */}
                        <div className="inline-flex p-1 rounded-xl bg-zinc-950/80 border border-white/[0.07] shadow-inner text-xs font-medium self-start">
                            {[
                                { key: 'all', label: 'All Issues', count: summary.total },
                                { key: 'pending', label: 'Pending', count: summary.pending, alert: summary.pending > 0 },
                                { key: 'approved', label: 'Approved', count: summary.approved },
                                { key: 'rejected', label: 'Rejected', count: summary.rejected },
                            ].map(tab => {
                                const isActive = filterStatus === tab.key;
                                return (
                                    <button
                                        key={tab.key}
                                        type="button"
                                        onClick={() => setFilterStatus(tab.key as typeof filterStatus)}
                                        className={cn(
                                            "relative px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition-all duration-200 cursor-pointer select-none",
                                            isActive
                                                ? "bg-white/[0.12] text-white font-semibold shadow-[0_1px_3px_rgba(0,0,0,0.4),inset_0_1px_0_rgba(255,255,255,0.15)] border border-white/10"
                                                : "text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.04]"
                                        )}
                                    >
                                        {tab.alert && tab.key === 'pending' && (
                                            <span className="w-1.5 h-1.5 rounded-full bg-amber-400 shadow-[0_0_6px_rgba(251,191,36,0.9)] animate-pulse" />
                                        )}
                                        <span>{tab.label}</span>
                                        <span className={cn(
                                            "px-1.5 py-0.2 rounded-md font-mono text-[10px]",
                                            isActive 
                                                ? "bg-white/20 text-white font-bold" 
                                                : "bg-white/[0.04] text-zinc-400"
                                        )}>
                                            {tab.count}
                                        </span>
                                    </button>
                                );
                            })}
                        </div>

                        {/* Search & Review Progress Bar */}
                        <div className="flex flex-wrap items-center gap-4">
                            {/* Quick Search */}
                            <div className="relative min-w-[200px] sm:min-w-[240px]">
                                <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-zinc-400 pointer-events-none" />
                                <input
                                    type="text"
                                    value={searchQuery}
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                    placeholder="Filter by column or type..."
                                    className="w-full h-8 pl-8 pr-3 text-xs rounded-lg bg-zinc-950/70 border border-white/[0.08] text-white placeholder:text-zinc-500 focus:outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/30 transition-all font-sans"
                                />
                                {searchQuery && (
                                    <button
                                        type="button"
                                        onClick={() => setSearchQuery('')}
                                        className="absolute right-2.5 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-zinc-300"
                                    >
                                        <X className="w-3 h-3" />
                                    </button>
                                )}
                            </div>

                            {/* Progress Metric */}
                            <div className="hidden sm:flex items-center gap-2.5 text-xs">
                                <span className="text-zinc-400 font-mono text-[11px]">
                                    Reviewed <strong className="text-white">{reviewedCount}</strong>/{summary.total}
                                </span>
                                <div className="w-24 h-1.5 rounded-full bg-zinc-900 border border-white/5 overflow-hidden">
                                    <motion.div
                                        className={cn(
                                            "h-full rounded-full transition-all duration-300",
                                            progressPct === 100 
                                                ? "bg-gradient-to-r from-emerald-500 to-teal-400 shadow-[0_0_8px_rgba(16,185,129,0.8)]" 
                                                : "bg-gradient-to-r from-amber-500 to-emerald-500"
                                        )}
                                        style={{ width: `${progressPct}%` }}
                                    />
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* ─── 3. ERROR BANNER ────────────────────────────────────────────── */}
                {error && (
                    <div className="bg-rose-500/15 border-b border-rose-500/30 px-6 py-3 text-xs font-semibold text-rose-300 flex items-center gap-2">
                        <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
                        <span>{error}</span>
                    </div>
                )}

                {/* ─── 4. ISSUES LIST (RESPONSIVE NON-SCROLLING GRID) ─────────────── */}
                <div className="max-h-[500px] overflow-y-auto custom-scrollbar divide-y divide-white/[0.04]">
                    {/* Header Columns for Desktop */}
                    <div className="hidden md:grid grid-cols-12 items-center gap-4 px-6 py-3 bg-white/[0.015] border-b border-white/[0.06] text-[11px] font-sans font-bold text-zinc-400 uppercase tracking-wider sticky top-0 backdrop-blur-md z-10">
                        <div className="col-span-5">Issue & Affected Column</div>
                        <div className="col-span-2">Severity & Impact</div>
                        <div className="col-span-3">Suggested Remediation</div>
                        <div className="col-span-2 text-right">Action / Status</div>
                    </div>

                    {filteredIssues.length === 0 ? (
                        <div className="py-12 text-center text-zinc-500 text-xs font-sans">
                            No issues matching current filter.
                        </div>
                    ) : (
                        filteredIssues.map((issue) => {
                            const sev = severityConfig[issue.severity];
                            const isApproved = issue.status === 'approved';
                            const isRejected = issue.status === 'rejected';
                            const isPending = issue.status === 'pending';
                            const isBusy = loading === issue.id;
                            const isExpanded = expandedCodeId === issue.id;

                            return (
                                <div
                                    key={issue.id}
                                    className={cn(
                                        "px-5 sm:px-6 py-4 hover:bg-white/[0.02] transition-colors duration-150 group",
                                        isRejected && "opacity-55 bg-white/[0.01]"
                                    )}
                                >
                                    {/* Desktop Grid Layout */}
                                    <div className="grid grid-cols-1 md:grid-cols-12 items-start md:items-center gap-3.5 md:gap-4">
                                        {/* Column 1: Issue & Column (5 cols) */}
                                        <div className="md:col-span-5 flex items-start gap-3">
                                            {/* Severity Icon in Squircle */}
                                            <div className={cn(
                                                "w-8 h-8 rounded-xl border flex items-center justify-center shrink-0 mt-0.5 transition-transform group-hover:scale-105",
                                                sev.iconBg
                                            )}>
                                                {issue.severity === 'critical' && <AlertCircle className="w-4 h-4" />}
                                                {issue.severity === 'high' && <AlertTriangle className="w-4 h-4" />}
                                                {issue.severity === 'medium' && <Info className="w-4 h-4" />}
                                                {issue.severity === 'low' && <Check className="w-4 h-4" />}
                                            </div>

                                            <div className="min-w-0 flex-1">
                                                <div className="flex flex-wrap items-center gap-2">
                                                    <span className="font-display font-bold text-sm text-white tracking-tight">
                                                        {formatIssueType(issue.issue_type)}
                                                    </span>
                                                    {/* Target Column Pill */}
                                                    <code className="text-[11px] font-mono bg-white/[0.05] text-emerald-300/90 border border-white/10 px-2 py-0.5 rounded-md shadow-2xs font-semibold">
                                                        {issue.column || 'all_rows'}
                                                    </code>
                                                </div>

                                                <p className="text-xs text-zinc-400 mt-1 leading-relaxed font-sans line-clamp-2">
                                                    {issue.description}
                                                </p>

                                                {/* Code Snippet Expander Toggle */}
                                                {issue.fix_code && (
                                                    <button
                                                        type="button"
                                                        onClick={() => setExpandedCodeId(isExpanded ? null : issue.id)}
                                                        className="mt-1.5 inline-flex items-center gap-1 text-[11px] font-mono text-zinc-400 hover:text-emerald-400 transition-colors cursor-pointer"
                                                    >
                                                        <Code2 className="w-3 h-3" />
                                                        <span>{isExpanded ? 'Hide Code' : 'View Polars Code'}</span>
                                                        {isExpanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                                                    </button>
                                                )}
                                            </div>
                                        </div>

                                        {/* Column 2: Severity & Impact (2 cols) */}
                                        <div className="md:col-span-2 flex flex-row md:flex-col items-center md:items-start justify-between md:justify-center gap-2">
                                            {/* Severity Badge */}
                                            <span className={cn(
                                                "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-mono font-bold uppercase tracking-wider border",
                                                sev.badge
                                            )}>
                                                <span className={cn("w-1.5 h-1.5 rounded-full", sev.dot)} />
                                                {issue.severity}
                                            </span>

                                            {/* Impact rows */}
                                            <div className="text-xs text-zinc-400 font-sans">
                                                <strong className="text-white font-semibold">{issue.affected_rows.toLocaleString()}</strong> rows
                                                <span className="text-zinc-500 font-mono text-[11px] ml-1">({issue.affected_pct}%)</span>
                                            </div>
                                        </div>

                                        {/* Column 3: Suggested Fix (3 cols) */}
                                        <div className="md:col-span-3 text-xs text-zinc-300 font-sans leading-relaxed">
                                            <span className="text-zinc-400 block text-[11px] uppercase font-mono font-semibold tracking-wider md:hidden mb-1">
                                                Suggested Fix:
                                            </span>
                                            {issue.suggested_fix}
                                        </div>

                                        {/* Column 4: Actions & Status (2 cols, right aligned) */}
                                        <div className="md:col-span-2 flex items-center justify-end gap-2 shrink-0">
                                            {/* ─── PENDING STATE BUTTONS ─── */}
                                            {!locked && isPending && (
                                                <div className="flex items-center gap-2">
                                                    {/* Tactile Approve Button */}
                                                    <button
                                                        type="button"
                                                        onClick={() => handleApprove(issue.id)}
                                                        disabled={loading !== null}
                                                        className={cn(
                                                            "group/btn inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold tracking-tight transition-all duration-150 cursor-pointer select-none",
                                                            "bg-emerald-500/15 hover:bg-emerald-500 text-emerald-300 hover:text-zinc-950 border border-emerald-500/35 hover:border-emerald-400 shadow-[0_2px_12px_-2px_rgba(16,185,129,0.3),inset_0_1px_0_rgba(255,255,255,0.2)] active:scale-95",
                                                            isBusy && "opacity-50 pointer-events-none"
                                                        )}
                                                        title="Approve this remediation fix"
                                                    >
                                                        <div className="w-4 h-4 rounded-md bg-emerald-500/25 group-hover/btn:bg-zinc-950/20 flex items-center justify-center transition-colors">
                                                            <Check className="w-3 h-3 stroke-[2.5]" />
                                                        </div>
                                                        <span>Approve</span>
                                                    </button>

                                                    {/* Tactile Reject Button */}
                                                    <button
                                                        type="button"
                                                        onClick={() => handleReject(issue.id)}
                                                        disabled={loading !== null}
                                                        className={cn(
                                                            "group/btn inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl text-xs font-medium tracking-tight transition-all duration-150 cursor-pointer select-none",
                                                            "bg-zinc-900/80 hover:bg-rose-500/20 text-zinc-400 hover:text-rose-200 border border-white/10 hover:border-rose-500/35 shadow-2xs active:scale-95",
                                                            isBusy && "opacity-50 pointer-events-none"
                                                        )}
                                                        title="Reject this remediation fix"
                                                    >
                                                        <div className="w-4 h-4 rounded-md bg-white/5 group-hover/btn:bg-rose-500/20 flex items-center justify-center transition-colors">
                                                            <X className="w-3 h-3 text-zinc-400 group-hover/btn:text-rose-300 stroke-[2.5]" />
                                                        </div>
                                                        <span>Reject</span>
                                                    </button>
                                                </div>
                                            )}

                                            {/* ─── APPROVED STATE BADGE ─── */}
                                            {isApproved && (
                                                <div className="flex items-center gap-2">
                                                    <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-xl bg-emerald-500/15 border border-emerald-500/35 text-emerald-300 text-xs font-semibold shadow-[0_0_12px_-3px_rgba(16,185,129,0.3)]">
                                                        <Check className="w-3.5 h-3.5 stroke-[2.5]" />
                                                        <span>Approved</span>
                                                    </span>
                                                    {!locked && (
                                                        <button
                                                            type="button"
                                                            onClick={() => handleReject(issue.id)}
                                                            disabled={loading !== null}
                                                            className="text-[11px] text-zinc-500 hover:text-zinc-300 underline transition-colors cursor-pointer"
                                                            title="Change decision to Reject"
                                                        >
                                                            Change
                                                        </button>
                                                    )}
                                                </div>
                                            )}

                                            {/* ─── REJECTED STATE BADGE ─── */}
                                            {isRejected && (
                                                <div className="flex items-center gap-2">
                                                    <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-xl bg-zinc-900/80 border border-white/10 text-zinc-400 text-xs font-medium">
                                                        <X className="w-3.5 h-3.5 stroke-[2.5]" />
                                                        <span>Rejected</span>
                                                    </span>
                                                    {!locked && (
                                                        <button
                                                            type="button"
                                                            onClick={() => handleApprove(issue.id)}
                                                            disabled={loading !== null}
                                                            className="text-[11px] text-zinc-500 hover:text-emerald-400 underline transition-colors cursor-pointer"
                                                            title="Change decision to Approve"
                                                        >
                                                            Approve
                                                        </button>
                                                    )}
                                                </div>
                                            )}

                                            {/* Locked Non-Pending Status */}
                                            {locked && isPending && (
                                                <span className="text-xs text-zinc-500 font-mono">Pending (Locked)</span>
                                            )}
                                        </div>
                                    </div>

                                    {/* Expandable Polars Fix Code Drawer */}
                                    <AnimatePresence>
                                        {isExpanded && issue.fix_code && (
                                            <motion.div
                                                initial={{ opacity: 0, height: 0 }}
                                                animate={{ opacity: 1, height: 'auto' }}
                                                exit={{ opacity: 0, height: 0 }}
                                                transition={{ duration: 0.2 }}
                                                className="overflow-hidden mt-3"
                                            >
                                                <div className="p-3 rounded-xl bg-zinc-950 border border-white/[0.08] shadow-inner flex items-center justify-between gap-3">
                                                    <div className="flex items-center gap-2.5 overflow-x-auto custom-scrollbar flex-1 font-mono text-xs text-emerald-300">
                                                        <span className="text-zinc-500 select-none"># Polars Transformation</span>
                                                        <code className="text-zinc-200">{issue.fix_code}</code>
                                                    </div>
                                                    <button
                                                        type="button"
                                                        onClick={() => handleCopyCode(issue.id, issue.fix_code)}
                                                        className="px-2.5 py-1 rounded-lg text-[11px] font-sans font-medium bg-white/[0.05] hover:bg-white/[0.1] text-zinc-300 border border-white/10 flex items-center gap-1.5 transition-colors shrink-0 cursor-pointer"
                                                    >
                                                        {copiedId === issue.id ? (
                                                            <>
                                                                <CheckIcon className="w-3 h-3 text-emerald-400" />
                                                                <span className="text-emerald-300">Copied</span>
                                                            </>
                                                        ) : (
                                                            <>
                                                                <Copy className="w-3 h-3" />
                                                                <span>Copy</span>
                                                            </>
                                                        )}
                                                    </button>
                                                </div>
                                            </motion.div>
                                        )}
                                    </AnimatePresence>
                                </div>
                            );
                        })
                    )}
                </div>

                {/* ─── 5. FOOTER & PRIMARY PROCEED ACTION ───────────────────────────── */}
                <div className="bg-white/[0.02] border-t border-white/[0.06] px-5 sm:px-7 py-4">
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                        {/* Status Alert Indicator */}
                        <div className="flex items-center gap-3">
                            {hasPending ? (
                                <div className="inline-flex items-center gap-2.5 px-3.5 py-2 rounded-xl bg-amber-500/10 border border-amber-500/25 text-amber-300 text-xs sm:text-sm font-medium shadow-[0_0_15px_-4px_rgba(245,158,11,0.2)]">
                                    <div className="w-2 h-2 rounded-full bg-amber-400 shadow-[0_0_8px_rgba(251,191,36,0.9)] animate-pulse" />
                                    <span>
                                        <strong className="text-white font-semibold">{summary.pending}</strong> issue(s) pending review before locking dataset.
                                    </span>
                                </div>
                            ) : (
                                <div className="inline-flex items-center gap-2.5 px-3.5 py-2 rounded-xl bg-emerald-500/10 border border-emerald-500/25 text-emerald-300 text-xs sm:text-sm font-medium shadow-[0_0_15px_-4px_rgba(16,185,129,0.2)]">
                                    <Sparkles className="w-4 h-4 text-emerald-400 animate-pulse" />
                                    <span>All issues reviewed! Dataset is ready to be locked and analyzed.</span>
                                </div>
                            )}
                        </div>

                        {/* Primary Lock & Proceed CTA */}
                        <div>
                            {hasPending || locked ? (
                                <div
                                    className="inline-flex items-center gap-2.5 px-6 py-2.5 rounded-xl text-xs sm:text-sm font-semibold bg-zinc-900/80 text-zinc-500 border border-white/10 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)] cursor-not-allowed select-none"
                                    title="All pending issues must be approved or rejected before proceeding"
                                >
                                    <div className="w-5 h-5 rounded-md bg-white/5 flex items-center justify-center text-zinc-500">
                                        <Lock className="w-3.5 h-3.5" />
                                    </div>
                                    <span>Lock & Proceed</span>
                                    <span className="font-mono text-[11px] text-zinc-600">
                                        ({summary.pending} pending)
                                    </span>
                                </div>
                            ) : (
                                <button
                                    type="button"
                                    onClick={handleLockAndProceed}
                                    disabled={loading !== null}
                                    className="group relative inline-flex items-center gap-3 px-7 py-3 rounded-xl text-sm font-bold bg-gradient-to-r from-emerald-500 via-teal-500 to-emerald-400 hover:from-emerald-400 hover:via-teal-400 hover:to-emerald-300 text-zinc-950 shadow-[0_0_35px_-5px_rgba(16,185,129,0.5),inset_0_1px_0_rgba(255,255,255,0.7)] border border-emerald-300/60 transition-all duration-200 active:scale-[0.98] cursor-pointer"
                                >
                                    <span>Lock & Proceed to Analysis</span>
                                    <div className="w-6 h-6 rounded-lg bg-black/15 flex items-center justify-center group-hover:translate-x-0.5 transition-transform">
                                        <ArrowRight className="w-3.5 h-3.5 text-zinc-950 stroke-[2.5]" />
                                    </div>
                                </button>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default IssueLedger;
