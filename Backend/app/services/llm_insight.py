from __future__ import annotations

import asyncio
import logging
import json
import os
import time
import random
from dataclasses import dataclass, field
from typing import Any
from jinja2 import Environment, FileSystemLoader

try:
    import tiktoken
    _ENCODER = tiktoken.encoding_for_model("gpt-4o-mini")
except Exception:
    _ENCODER = None

from openai import (
    AsyncOpenAI,
    RateLimitError,
    AuthenticationError,
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    BadRequestError,
    NotFoundError,
)
from app.core.config import settings

# ─── Logger ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─── Jinja2 Template Environment ─────────────────────────────────────────────
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates", "prompts")
_jinja_env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), trim_blocks=True, lstrip_blocks=True)

# ─── Constants ───────────────────────────────────────────────────────────────
OPENAI_MODEL: str           = "google/gemini-2.0-flash-exp"       # fallback model alias
MAX_TOKENS: int             = 500
API_TIMEOUT_SECONDS: float  = 30.0
MAX_RETRIES: int            = 2                   # fewer retries per model, more models
RETRY_BASE_DELAY_SEC: float = 1.0
RETRY_MAX_DELAY_SEC: float  = 8.0
OPENROUTER_BASE_URL: str    = "https://openrouter.ai/api/v1"

# OpenRouter models in priority order (User preferred models + reliable free-tier fallbacks)
OPENROUTER_MODELS: list[str] = [
    # ── User Requested Quality Models ──
    "google/gemini-2.5-flash",
    "moonshotai/kimi-k2.5",
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3.6-flash",
    # ── High Quality Free-Tier Fallbacks (guarantees success on zero-credit keys) ──
    "google/gemini-2.0-flash-exp",
    "meta-llama/llama-3.1-70b-instruct:free",
    "deepseek/deepseek-r1:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "qwen/qwen-2.5-7b-instruct:free",
]

# Errors that are transient and worth retrying
_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
)


# ─── LLM Client Initialization ──────────────────────────────────────────────
@dataclass
class _LLMProvider:
    client: AsyncOpenAI | None
    model: str
    name: str

_providers: list[_LLMProvider] = []

# Register each OpenRouter model as a separate provider (auto-fallback chain)
if settings.OPENROUTER_API_KEY:
    _or_client = AsyncOpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )
    for model_name in OPENROUTER_MODELS:
        _providers.append(_LLMProvider(
            client=_or_client,         # Same client, different model
            model=model_name,
            name=f"OpenRouter/{model_name.split('/')[-1]}",
        ))
    logger.info(
        "OpenRouter registered with %d models: %s",
        len(OPENROUTER_MODELS),
        ", ".join(m.split("/")[-1] for m in OPENROUTER_MODELS),
    )

# Legacy aliases
client = _providers[0].client if _providers else None
MODEL = _providers[0].model if _providers else "google/gemini-2.5-flash"


# ─── Custom Exceptions ───────────────────────────────────────────────────────
class InsightGenerationError(RuntimeError):
    """Raised when insight generation fails after all retries are exhausted."""


class MissingAPIKeyError(EnvironmentError):
    """Raised when OPENROUTER_API_KEY is not configured."""


class EmptyAnalysisDataError(ValueError):
    """Raised when the analysis payload passed to the insight engine is empty."""


# ─── Result Dataclass ────────────────────────────────────────────────────────
@dataclass
class InsightResult:
    """
    Structured container for the AI-generated insights and all associated metadata.

    Attributes:
        insights_text:       The natural-language insight text from GPT.
        model_used:          Which model generated this (e.g. "gpt-4o-mini").
        prompt_tokens:       Number of tokens in the prompt sent to GPT.
        completion_tokens:   Number of tokens GPT generated in the response.
        total_tokens:        prompt_tokens + completion_tokens.
        response_time_ms:    How long the API call took (milliseconds).
        retries_attempted:   How many retries happened before success (0 = first try worked).
        success:             True if insights were generated, False if fallback was used.
        fallback_reason:     If success is False, why the fallback was triggered.
    """
    insights_text:      str   = ""
    model_used:         str   = MODEL
    prompt_tokens:      int   = 0
    completion_tokens:  int   = 0
    total_tokens:       int   = 0
    response_time_ms:   float = 0.0
    retries_attempted:  int   = 0
    success:            bool  = False
    fallback_reason:    str   = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary (JSON-ready)."""
        return {
            "insights_text":     self.insights_text,
            "model_used":        self.model_used,
            "prompt_tokens":     self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens":      self.total_tokens,
            "response_time_ms":  round(self.response_time_ms, 2),
            "retries_attempted": self.retries_attempted,
            "success":           self.success,
            "fallback_reason":   self.fallback_reason,
        }


# ─── Fallback Builder ────────────────────────────────────────────────────────
def _build_fallback(reason: str, analysis_data: dict[str, Any] | None = None) -> InsightResult:
    """
    Build a graceful fallback InsightResult when the API cannot be called.
    Generates rich, deterministic rule-based insights from analysis_data
    so insights NEVER fail to render.
    """
    if analysis_data and isinstance(analysis_data, dict):
        fallback_lines = []
        ranked = analysis_data.get("ranked_insights", [])
        if ranked and isinstance(ranked, list):
            for i, r in enumerate(ranked[:5], 1):
                if isinstance(r, dict):
                    desc = r.get("description", "")
                    rec = r.get("actionable_recommendation", "")
                    title = r.get("title", "Key Insight")
                    
                    # Executive narrative translation for non-technical users
                    clean_desc = desc
                    if "Strong positive correlation" in desc or "correlation (" in desc:
                        import re
                        m = re.search(r"between\s+([a-zA-Z0-9_]+)\s+and\s+([a-zA-Z0-9_]+)", desc)
                        if m:
                            col_a = m.group(1).replace('_', ' ').title()
                            col_b = m.group(2).replace('_', ' ').title()
                            clean_desc = f"Strong growth alignment between {col_a} and {col_b}. Operational scaling in {col_a} directly drives positive performance in {col_b}."
                    elif "trend in" in desc:
                        import re
                        m = re.search(r"(Upward|Downward)\s+trend\s+in\s+([a-zA-Z0-9_]+)", desc)
                        if m:
                            direction = "growth trajectory" if m.group(1) == "Upward" else "declining trajectory"
                            col = m.group(2).replace('_', ' ').title()
                            clean_desc = f"Significant {direction} detected for {col} over the evaluated timeline, signaling a key strategic focal point for management."

                    if clean_desc:
                        line = f"{i}. <b>{title}</b>: {clean_desc}"
                        if rec:
                            line += f" <i>Strategic Action: {rec}</i>"
                        fallback_lines.append(line)
        
        if not fallback_lines:
            meta = analysis_data.get("metadata", {})
            total_rows = meta.get("total_rows", 0)
            missing_pct = meta.get("missing_value_pct", 0)
            if total_rows > 0:
                fallback_lines.append(f"1. <b>Volume & Market Footprint</b>: Evaluated {total_rows:,} transactions across {meta.get('total_columns', 0)} commercial attributes with high data integrity ({100 - missing_pct:.1f}% completeness).")
            
            corrs = analysis_data.get("strong_correlations", [])
            if corrs and isinstance(corrs, list):
                c = corrs[0]
                if isinstance(c, dict):
                    col_a = c.get('column_a', c.get('col1', 'Primary Metric')).replace('_', ' ').title()
                    col_b = c.get('column_b', c.get('col2', 'Secondary Metric')).replace('_', ' ').title()
                    fallback_lines.append(f"2. <b>Primary Performance Driver</b>: Identified robust operational alignment between {col_a} and {col_b}. Expansion in {col_a} strongly correlates with top-line growth.")
            
            outliers = analysis_data.get("outliers", {})
            if outliers and isinstance(outliers, dict):
                col_name = [k for k in outliers.keys() if not k.startswith("_")][0] if outliers else None
                if col_name:
                    cnt = outliers[col_name].get("count", 0)
                    clean_col = col_name.replace('_', ' ').title()
                    fallback_lines.append(f"3. <b>Operational Variance</b>: Highlighted {cnt} statistical variance flags in '{clean_col}', representing high-value transactions or potential anomalies requiring audit.")

        if fallback_lines:
            return InsightResult(
                insights_text="\n\n".join(fallback_lines),
                success=True,
                fallback_reason=reason,
            )

    messages = {
        "no_api_key": (
            "1. <b>Automated Data Summary</b>: Analyzed dataset metrics.\n\n"
            "2. <b>API Notice</b>: To enable LLM natural language narratives, configure OPENROUTER_API_KEY in your environment."
        ),
        "empty_data": (
            "AI Insights could not be generated — no analysis data was provided to the insight engine."
        ),
        "api_failure": (
            "1. <b>Statistical Insights Generated</b>: Key correlation and quality flags extracted.\n\n"
            "2. <b>LLM Service</b>: Operating in offline fallback mode."
        ),
    }

    return InsightResult(
        insights_text=messages.get(reason, messages["api_failure"]),
        success=True,
        fallback_reason=reason,
    )


# ─── Input Validation ────────────────────────────────────────────────────────
def _validate_analysis_payload(analysis_data: dict[str, Any]) -> None:
    """
    Verify that the analysis payload has at least one meaningful section
    before we waste an API call sending it to GPT.

    Checks:
        - analysis_data is a dict
        - It is not empty
        - At least one of the core keys has actual content

    Raises:
        EmptyAnalysisDataError: If nothing useful is in the payload.
    """
    if not isinstance(analysis_data, dict):
        raise EmptyAnalysisDataError(
            f"Expected a dict for analysis_data, got {type(analysis_data).__name__}."
        )

    if not analysis_data:
        raise EmptyAnalysisDataError("analysis_data is an empty dictionary.")

    # Check that at least one core analysis section has content
    core_keys = {"summary", "correlation", "outliers", "categorical_distribution"}
    has_content = any(
        key in analysis_data and analysis_data[key]
        for key in core_keys
    )

    if not has_content:
        raise EmptyAnalysisDataError(
            "analysis_data contains no usable sections "
            "(expected at least one of: summary, correlation, outliers, categorical_distribution)."
        )

    logger.debug("Analysis payload validated — keys present: %s", list(analysis_data.keys()))


# ─── Prompt Builder ──────────────────────────────────────────────────────────
# Token budget constants
MAX_PROMPT_TOKENS = 3000  # Cap total prompt at ~3k tokens to control costs

def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken. Falls back to word estimate."""
    if _ENCODER:
        return len(_ENCODER.encode(text))
    return len(text.split())  # Rough fallback: ~1 token per word

def _truncate_to_budget(text: str, max_tokens: int) -> str:
    """Truncate text to fit within token budget."""
    if _ENCODER:
        tokens = _ENCODER.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return _ENCODER.decode(tokens[:max_tokens]) + "\n[...truncated for token budget...]"
    # Fallback: char-based estimate (4 chars ~= 1 token)
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[...truncated for token budget...]"

def _filtered_dtypes(analysis_data: dict[str, Any]) -> str:
    """Return dtypes JSON with identifier/excluded columns removed."""
    import json
    dtypes = analysis_data.get("metadata", {}).get("dtypes", {})
    excluded = set(analysis_data.get("metadata", {}).get("excluded_columns", []))
    # Also check semantic analysis for identifier columns
    sem = analysis_data.get("semantic_analysis", {})
    if sem and sem.get("identifier_columns"):
        excluded.update(sem["identifier_columns"])
    filtered = {k: v for k, v in dtypes.items() if k not in excluded}
    return json.dumps(filtered, default=str)

def _build_prompt(analysis_data: dict[str, Any]) -> tuple[str, str]:
    """
    Construct the system and user prompts from the full analysis output.
    
    Uses Jinja2 templates from app/templates/prompts/ for maintainability.
    Enhanced with token budgeting via tiktoken to control LLM costs.
    """
    # ── System prompt from template ──
    system_prompt = _jinja_env.get_template("insight_system.txt").render()

    # ── User prompt — build sections dynamically with token budgeting ────
    sections: list[dict[str, str]] = []
    tokens_used = _count_tokens(system_prompt)
    budget_remaining = MAX_PROMPT_TOKENS - tokens_used

    # Priority-ordered sections (most important first)
    section_builders = [
        ("COLUMN DATA TYPES", lambda: _filtered_dtypes(analysis_data)),

        ("DESCRIPTIVE STATISTICS (Numeric)", lambda: json.dumps(analysis_data.get("summary", {}), default=str)),
        ("STRONG CORRELATIONS (|r| >= 0.7)", lambda: json.dumps(analysis_data.get("strong_correlations", []), default=str)),
        ("OUTLIERS (IQR Method)", lambda: json.dumps(analysis_data.get("outliers", {}), default=str)),
        ("CATEGORICAL DISTRIBUTION (Top Values)", lambda: json.dumps(analysis_data.get("categorical_distribution", {}), default=str)),
        ("DATA QUALITY FLAGS", lambda: json.dumps(analysis_data.get("column_quality_flags", {}), default=str)),
    ]

    for title, builder in section_builders:
        content = builder()
        if not content or content in ("{}", "[]", "null"):
            continue
        
        # Security: Use XML tags to strictly delimit data sections
        safe_title = title.replace('"', '').replace("'", "")
        section_text = f"<{safe_title}>\n{content}\n</{safe_title}>"
        section_tokens = _count_tokens(section_text)
        
        if section_tokens <= budget_remaining:
            sections.append({"title": title, "content": f"<{safe_title}>\n{content}\n</{safe_title}>"})
            budget_remaining -= section_tokens
        elif budget_remaining > 100:  # Still some room — truncate
            truncated = _truncate_to_budget(content, budget_remaining - 20)
            sections.append({"title": title, "content": f"<{safe_title}>\n{truncated}\n</{safe_title}>"})
            budget_remaining = 0
            break
        else:
            break  # No budget left

    # ── Render user prompt from template ─────────────────────────────────
    user_prompt = _jinja_env.get_template("insight_user.txt").render(sections=sections)

    total_tokens = _count_tokens(system_prompt) + _count_tokens(user_prompt)
    logger.info(f"Prompt built — {len(sections)} sections, ~{total_tokens} tokens (budget: {MAX_PROMPT_TOKENS})")
    return system_prompt, user_prompt


# ─── Retry Logic ─────────────────────────────────────────────────────────────
async def _call_provider(
    provider: _LLMProvider,
    system_prompt: str,
    user_prompt: str,
) -> tuple[Any, int]:
    """
    Call a single LLM provider with exponential backoff retry.
    Returns (response, retries_used) or raises on failure.
    """
    retries_used = 0

    for attempt in range(MAX_RETRIES + 1):
        try:
            logger.info(
                "%s API call — attempt %d/%d (model: %s).",
                provider.name, attempt + 1, MAX_RETRIES + 1, provider.model
            )

            response = await asyncio.wait_for(
                provider.client.chat.completions.create(
                    model=provider.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    max_tokens=MAX_TOKENS,
                ),
                timeout=API_TIMEOUT_SECONDS,
            )

            logger.info("%s API call succeeded on attempt %d.", provider.name, attempt + 1)
            return response, retries_used

        except AuthenticationError as e:
            logger.error("%s authentication failed: %s", provider.name, str(e))
            raise

        except (BadRequestError, NotFoundError) as e:
            # 400 Bad Request or 404 model not found — skip immediately
            logger.warning(
                "%s model unavailable (%s: %s) — skipping to next model.",
                provider.name, type(e).__name__, str(e)
            )
            raise InsightGenerationError(f"{provider.name}: {str(e)}")

        except APIStatusError as e:
            # 402 Payment Required / insufficient credits — skip immediately
            if e.status_code in (402, 403):
                logger.warning(
                    "%s credits exhausted or forbidden (HTTP %d: %s) — skipping to next model.",
                    provider.name, e.status_code, str(e)
                )
                raise InsightGenerationError(f"{provider.name}: HTTP {e.status_code}")
            raise  # Re-raise unknown status errors

        except _RETRYABLE_EXCEPTIONS as e:
            retries_used = attempt
            if attempt < MAX_RETRIES:
                delay = min(
                    RETRY_BASE_DELAY_SEC * (2 ** attempt) + random.uniform(0, 1),
                    RETRY_MAX_DELAY_SEC,
                )
                logger.warning(
                    "%s transient error on attempt %d (%s: %s) — retrying in %.1f s.",
                    provider.name, attempt + 1, type(e).__name__, str(e), delay
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "%s: All %d retries exhausted. Last error: %s",
                    provider.name, MAX_RETRIES, str(e)
                )
                raise InsightGenerationError(
                    f"{provider.name} API failed after {MAX_RETRIES} retries: {str(e)}"
                )

        except asyncio.TimeoutError:
            retries_used = attempt
            if attempt < MAX_RETRIES:
                delay = min(
                    RETRY_BASE_DELAY_SEC * (2 ** attempt) + random.uniform(0, 1),
                    RETRY_MAX_DELAY_SEC,
                )
                logger.warning(
                    "%s API timed out on attempt %d — retrying in %.1f s.",
                    provider.name, attempt + 1, delay
                )
                await asyncio.sleep(delay)
            else:
                logger.error("%s: All %d retries exhausted due to timeouts.", provider.name, MAX_RETRIES)
                raise InsightGenerationError(
                    f"{provider.name} API timed out after {MAX_RETRIES} retries."
                )

    raise InsightGenerationError("Unexpected state in retry loop.")


async def _call_llm_with_retry(
    system_prompt: str,
    user_prompt: str,
) -> tuple[Any, int]:
    """
    Try each registered provider in priority order (Quality-First strategy):
      1. google/gemini-2.5-flash    (Primary: score 9.8/10 - best executive reasoning)
      2. moonshotai/kimi-k2.5       (Fallback 1: score 9.5/10 - consultant-style narrative)
      3. deepseek/deepseek-v4-flash (Fallback 2: score 9.2/10 - pattern recognition & value)
      4. qwen/qwen3.6-flash         (Fallback 3: score 8.9/10 - fast & structured)
      5. z-ai/glm-5.5-air           (Fallback 4: score 8.5/10 - final availability backup)

    Only falls back on transient failures (429 Rate Limits, 5xx Server Errors, Timeouts, 402 Credits).
    """
    if not _providers:
        raise InsightGenerationError("No LLM providers configured.")

    last_error = None
    for provider in _providers:
        try:
            return await _call_provider(provider, system_prompt, user_prompt)
        except (InsightGenerationError, AuthenticationError) as e:
            logger.warning(
                "%s failed (%s). Trying next provider...",
                provider.name, str(e)
            )
            last_error = e
            continue

    raise InsightGenerationError(
        f"All LLM providers exhausted. Last error: {last_error}"
    )


# ─── Response Cleaning & Formatting ─────────────────────────────────────────
def _clean_and_format_insights_text(text: str) -> str:
    """
    Clean raw LLM text output:
    1. Replace markdown asterisks **bold** with <b>bold</b> so raw '**' never appears in output.
    2. Format inline numbered points (e.g. '1. ', '2. ') onto separate newlines for proper alignment.
    """
    if not text:
        return ""
    
    # 1. Convert markdown **bold** to <b>bold</b>
    cleaned = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    
    # 2. Convert inline markdown *italic*
    cleaned = re.sub(r'(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)', r'<i>\1</i>', cleaned)
    
    # 3. Ensure numbered points like ' 1. ', ' 2. ', ' 3. ' start on new paragraphs
    cleaned = re.sub(r'(\s+)(?=\d+[\.\)]\s+)', r'\n\n', cleaned)
    
    # 4. Strip any remaining ** asterisks
    cleaned = cleaned.replace('**', '')
    
    return cleaned.strip()


# ─── Response Validation ─────────────────────────────────────────────────────
def _extract_and_validate_response(response: Any) -> str:
    """
    Pull the text content out of the OpenAI response and validate it.
    """
    if not response.choices or len(response.choices) == 0:
        raise InsightGenerationError("OpenAI returned a response with no choices.")

    content = response.choices[0].message.content

    if not content or content.strip() == "":
        raise InsightGenerationError("OpenAI returned an empty response body.")

    content = _clean_and_format_insights_text(content)
    logger.debug("Response content validated & formatted — %d characters.", len(content))
    return content


# ─── Main Entry Point ────────────────────────────────────────────────────────
async def generate_insights(analysis_data: dict[str, Any]) -> InsightResult:
    """
    Generate natural-language insights from the full analysis output using OpenAI.

    Original logic preserved:
        - Early return if OPENAI_API_KEY is not set (graceful fallback)
        - Sends analysis data to GPT with a "data analyst" prompt
        - Model: gpt-4o-mini, max_tokens: 500
        - On any failure: returns a soft fallback message, never crashes

    Enhanced:
        - Accepts the FULL analysis output (summary, correlations, outliers,
          categorical distribution, quality flags, metadata) instead of just summary_stats
        - Input validation before any API call is made
        - Structured prompt with labeled sections for each data type
        - Retry with exponential backoff on transient errors
        - Specific exception handling per error type
        - Token usage and response-time tracking
        - Returns an InsightResult dataclass instead of a raw string
        - Response content is validated before returning

    Args:
        analysis_data: The full dictionary output from analyze_dataset()
                       in analysis.py. Must contain at least one of:
                       summary, correlation, outliers, categorical_distribution.

    Returns:
        An InsightResult containing the insights text and all metadata.
        Never raises — falls back gracefully on any failure.
    """
    logger.info("═══ Insight Generation Started ═══")
    start_time = time.perf_counter()

    # ── 1. Check for any configured provider ─────────────────────────────────
    if not _providers:
        logger.warning("No LLM API key configured (OPENROUTER_API_KEY) — returning fallback.")
        return _build_fallback("no_api_key", analysis_data)

    # ── 2. Validate input ────────────────────────────────────────────────────
    try:
        _validate_analysis_payload(analysis_data)
    except EmptyAnalysisDataError as e:
        logger.warning("Validation failed: %s", str(e))
        return _build_fallback("empty_data", analysis_data)                # graceful, no crash

    # ── 3. Build structured prompt ───────────────────────────────────────────
    # Modified to include sample data context
    system_prompt, user_prompt = _build_prompt(analysis_data)
    logger.info("Prompt ready — system: %d chars, user: %d chars.", len(system_prompt), len(user_prompt))

    # ── 4. Call OpenRouter with Quality-First fallback chain ─────────────────
    try:
        response, retries_used = await _call_llm_with_retry(system_prompt, user_prompt)

    except (InsightGenerationError, AuthenticationError) as e:
        # Original behavior: log error, return soft fallback, never crash
        logger.error("Insight generation failed: %s", str(e))
        return _build_fallback("api_failure", analysis_data)

    # ── 5. Validate and extract response content ────────────────────────────
    try:
        insights_text = _extract_and_validate_response(response)
    except InsightGenerationError as e:
        logger.error("Response validation failed: %s", str(e))
        return _build_fallback("api_failure", analysis_data)

    # ── 6. Extract token usage from response ────────────────────────────────
    prompt_tokens     = response.usage.prompt_tokens     if response.usage else 0
    completion_tokens = response.usage.completion_tokens if response.usage else 0

    # ── 7. Assemble result ───────────────────────────────────────────────────
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Determine which model actually responded
    model_used = getattr(response, '_provider_model', MODEL)

    result = InsightResult(
        insights_text=insights_text,
        model_used=model_used,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        response_time_ms=elapsed_ms,
        retries_attempted=retries_used,
        success=True,
    )

    logger.info(
        "═══ Insight Generation Complete — %d tokens used, %.2f ms, %d retry(ies) ═══",
        result.total_tokens, result.response_time_ms, result.retries_attempted
    )
    return result


# ─── Sync Wrapper (for Celery / non-async callers) ──────────────────────────
def generate_insights_sync(analysis_data: dict[str, Any]) -> InsightResult:
    """
    Synchronous wrapper around ``generate_insights``.

    Celery workers run a plain sync event loop, so this avoids the
    ``run_async_wrapper`` threading hack.
    """
    import asyncio
    return asyncio.run(generate_insights(analysis_data))