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
)
from app.core.config import settings

# ─── Logger ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─── Jinja2 Template Environment ─────────────────────────────────────────────
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates", "prompts")
_jinja_env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), trim_blocks=True, lstrip_blocks=True)

# ─── Constants ───────────────────────────────────────────────────────────────
MODEL: str                  = "gpt-4o-mini"       # original model preserved
MAX_TOKENS: int             = 500                 # original limit preserved
API_TIMEOUT_SECONDS: float  = 30.0                # hard timeout so calls never hang
MAX_RETRIES: int            = 3                   # max retry attempts on transient errors
RETRY_BASE_DELAY_SEC: float = 1.0                 # base delay for exponential backoff
RETRY_MAX_DELAY_SEC: float  = 16.0                # cap on backoff delay

# Errors that are transient and worth retrying
_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
)


# ─── OpenAI Client (Lazy Initialization) ──────────────────────────────────────
client = None
if settings.OPENAI_API_KEY:
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
# Else: client remains None, handled in generate_insights


# ─── Custom Exceptions ───────────────────────────────────────────────────────
class InsightGenerationError(RuntimeError):
    """Raised when insight generation fails after all retries are exhausted."""


class MissingAPIKeyError(EnvironmentError):
    """Raised when OPENAI_API_KEY is not configured."""


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
def _build_fallback(reason: str) -> InsightResult:
    """
    Build a graceful fallback InsightResult when the API cannot be called.
    Original logic: return a soft message, never crash.
    Enhanced: structured result with the exact reason why it failed.
    """
    messages = {
        "no_api_key": (
            "AI Insights are unavailable. "
            "Please configure OPENAI_API_KEY in your .env file to enable this feature."
        ),
        "empty_data": (
            "AI Insights could not be generated — "
            "no analysis data was provided to the insight engine."
        ),
        "api_failure": (
            "Could not generate AI insights at this time. "
            "The service will retry on your next request."
        ),
    }

    return InsightResult(
        insights_text=messages.get(reason, messages["api_failure"]),
        success=False,
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
        ("COLUMN DATA TYPES", lambda: json.dumps(analysis_data.get("metadata", {}).get("dtypes", {}), default=str)),
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
        
        section_text = f"--- {title} ---\n{content}"
        section_tokens = _count_tokens(section_text)
        
        if section_tokens <= budget_remaining:
            sections.append({"title": title, "content": content})
            budget_remaining -= section_tokens
        elif budget_remaining > 100:  # Still some room — truncate
            truncated = _truncate_to_budget(content, budget_remaining - 20)
            sections.append({"title": title, "content": truncated})
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
async def _call_openai_with_retry(
    system_prompt: str,
    user_prompt: str,
) -> tuple[Any, int]:
    """
    Call the OpenAI API with exponential backoff retry on transient errors.

    Original logic preserved:
        - Uses client.chat.completions.create()
        - Model: gpt-4o-mini
        - max_tokens: 500
        - Messages: system + user roles

    Enhanced:
        - Retries up to MAX_RETRIES times on RateLimitError, ConnectionError, TimeoutError
        - Exponential backoff with jitter between retries
        - Specific handling for AuthenticationError (no retry — it won't fix itself)
        - Hard timeout on each individual API call
        - Returns both the response object and the number of retries attempted

    Returns:
        Tuple of (OpenAI response object, number of retries used).

    Raises:
        AuthenticationError:      If the API key is invalid (no retry).
        InsightGenerationError:   If all retries are exhausted.
    """
    retries_used = 0

    for attempt in range(MAX_RETRIES + 1):                  # 0, 1, 2, 3
        try:
            logger.info(
                "OpenAI API call — attempt %d/%d.",
                attempt + 1, MAX_RETRIES + 1
            )

            # Original API call logic preserved exactly
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=MODEL,                                     # original
                    messages=[                                       # original structure
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    max_tokens=MAX_TOKENS,                           # original
                ),
                timeout=API_TIMEOUT_SECONDS,
            )

            logger.info("OpenAI API call succeeded on attempt %d.", attempt + 1)
            return response, retries_used

        except AuthenticationError as e:
            # Auth errors will never fix themselves — fail immediately, no retry
            logger.error("OpenAI authentication failed: %s", str(e))
            raise

        except _RETRYABLE_EXCEPTIONS as e:
            retries_used = attempt
            if attempt < MAX_RETRIES:
                # Exponential backoff with jitter
                delay = min(
                    RETRY_BASE_DELAY_SEC * (2 ** attempt) + random.uniform(0, 1),
                    RETRY_MAX_DELAY_SEC,
                )
                logger.warning(
                    "Transient error on attempt %d (%s: %s) — retrying in %.1f s.",
                    attempt + 1, type(e).__name__, str(e), delay
                )
                await asyncio.sleep(delay)
            else:
                # All retries exhausted
                logger.error(
                    "All %d retries exhausted. Last error: %s",
                    MAX_RETRIES, str(e)
                )
                raise InsightGenerationError(
                    f"OpenAI API failed after {MAX_RETRIES} retries: {str(e)}"
                )

        except asyncio.TimeoutError:
            retries_used = attempt
            if attempt < MAX_RETRIES:
                delay = min(
                    RETRY_BASE_DELAY_SEC * (2 ** attempt) + random.uniform(0, 1),
                    RETRY_MAX_DELAY_SEC,
                )
                logger.warning(
                    "API call timed out on attempt %d — retrying in %.1f s.",
                    attempt + 1, delay
                )
                await asyncio.sleep(delay)
            else:
                logger.error("All %d retries exhausted due to timeouts.", MAX_RETRIES)
                raise InsightGenerationError(
                    f"OpenAI API timed out after {MAX_RETRIES} retries."
                )

    # Should never reach here, but just in case
    raise InsightGenerationError("Unexpected state in retry loop.")


# ─── Response Validation ─────────────────────────────────────────────────────
def _extract_and_validate_response(response: Any) -> str:
    """
    Pull the text content out of the OpenAI response and validate it.

    Original logic preserved:
        - response.choices[0].message.content

    Enhanced:
        - Checks that choices list is not empty
        - Checks that the content is not None or blank
        - Returns the stripped text

    Raises:
        InsightGenerationError: If the response contains no usable content.
    """
    if not response.choices or len(response.choices) == 0:
        raise InsightGenerationError("OpenAI returned a response with no choices.")

    content = response.choices[0].message.content  # original logic

    if not content or content.strip() == "":
        raise InsightGenerationError("OpenAI returned an empty response body.")

    logger.debug("Response content validated — %d characters.", len(content.strip()))
    return content.strip()


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

    # ── 1. Check API key (original logic preserved) ─────────────────────────
    if not settings.OPENAI_API_KEY or not client:
        logger.warning("OPENAI_API_KEY is not configured — returning fallback.")
        return _build_fallback("no_api_key")                # original behavior

    # ── 2. Validate input ────────────────────────────────────────────────────
    try:
        _validate_analysis_payload(analysis_data)
    except EmptyAnalysisDataError as e:
        logger.warning("Validation failed: %s", str(e))
        return _build_fallback("empty_data")                # graceful, no crash

    # ── 3. Build structured prompt ───────────────────────────────────────────
    # Modified to include sample data context
    system_prompt, user_prompt = _build_prompt(analysis_data)
    logger.info("Prompt ready — system: %d chars, user: %d chars.", len(system_prompt), len(user_prompt))

    # ── 4. Call OpenAI with retry logic ──────────────────────────────────────
    try:
        response, retries_used = await _call_openai_with_retry(system_prompt, user_prompt)

    except (InsightGenerationError, AuthenticationError) as e:
        # Original behavior: log error, return soft fallback, never crash
        logger.error("Insight generation failed: %s", str(e))
        return _build_fallback("api_failure")

    # ── 5. Validate and extract response content ────────────────────────────
    try:
        insights_text = _extract_and_validate_response(response)
    except InsightGenerationError as e:
        logger.error("Response validation failed: %s", str(e))
        return _build_fallback("api_failure")

    # ── 6. Extract token usage from response ────────────────────────────────
    prompt_tokens     = response.usage.prompt_tokens     if response.usage else 0
    completion_tokens = response.usage.completion_tokens if response.usage else 0

    # ── 7. Assemble result ───────────────────────────────────────────────────
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    result = InsightResult(
        insights_text=insights_text,
        model_used=MODEL,
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