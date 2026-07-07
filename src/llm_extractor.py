from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

import json
import logging
import os
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_random_exponential

from .preprocessing import chunk_text
from .prompts import (
    build_extraction_prompt,
    build_merge_clauses_prompt,
    build_summary_prompt,
)

logger = logging.getLogger(__name__)

# Groq hosts several open-weight models; llama-3.3-70b-versatile is a solid
# default (strong instruction-following, good at structured JSON output).
# Override via GROQ_MODEL in .env, e.g. to llama-3.1-8b-instant for speed.
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_CHUNK_CHARS = 12_000  # ~3k tokens, keeps prompt+completion well within limits

CLAUSE_KEYS = ["termination_clause", "confidentiality_clause", "liability_clause"]

_EMPTY_CLAUSE = {"found": False, "excerpts": [], "notes": ""}


def _use_mock() -> bool:
    return os.environ.get("USE_MOCK_LLM", "false").lower() in {"1", "true", "yes"}


def _get_client():
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Either set it in your .env file "
            "(get a free key at https://console.groq.com/keys), or set "
            "USE_MOCK_LLM=true to run without calling the API."
        )
    return Groq(api_key=api_key)


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(5))
def _call_groq_api(client, system: str, user_prompt: str, max_tokens: int) -> str:
    """The actual network call -- wrapped in retry for transient errors only
    (rate limits, timeouts, 5xx). Config errors (e.g. missing API key) are
    raised before this function is ever called, so they fail fast instead
    of retrying 5 times for nothing."""
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""


def _call_llm(system: str, user_prompt: str, max_tokens: int = 1500) -> str:
    if _use_mock():
        return _mock_response(user_prompt)

    client = _get_client()  # raises RuntimeError immediately if misconfigured
    return _call_groq_api(client, system, user_prompt, max_tokens)


def _mock_response(user_prompt: str) -> str:
    """Deterministic stand-in for the LLM, used for offline pipeline testing."""
    if "single valid JSON object" in user_prompt and "excerpts" in user_prompt:
        return json.dumps(
            {
                "termination_clause": {
                    "found": True,
                    "excerpts": ["[MOCK] Either party may terminate this Agreement upon 30 days notice."],
                    "notes": "[MOCK] Standard 30-day termination for convenience.",
                },
                "confidentiality_clause": {
                    "found": True,
                    "excerpts": ["[MOCK] Each party shall keep Confidential Information secret."],
                    "notes": "[MOCK] Mutual confidentiality obligation.",
                },
                "liability_clause": {
                    "found": False,
                    "excerpts": [],
                    "notes": "",
                },
            }
        )
    return (
        "[MOCK SUMMARY] This is a placeholder summary generated in mock mode "
        "(USE_MOCK_LLM=true), standing in for a 100-150 word LLM-written "
        "summary of the contract's purpose, obligations, and risks. Set "
        "USE_MOCK_LLM=false and provide a GROQ_API_KEY to generate a "
        "real summary from the contract text."
    )


def _parse_json_response(raw: str) -> dict:
    """Parse a JSON object out of the model's response, tolerating stray
    markdown fences some models add despite instructions not to."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # last resort: grab the outermost {...} span
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1:
            return json.loads(cleaned[start : end + 1])
        raise


def _normalize_clause_result(result: dict) -> dict:
    normalized = {}
    for key in CLAUSE_KEYS:
        val = result.get(key, _EMPTY_CLAUSE)
        normalized[key] = {
            "found": bool(val.get("found", False)),
            "excerpts": list(val.get("excerpts", [])),
            "notes": val.get("notes", ""),
        }
    return normalized


SYSTEM_PROMPT = (
    "You are a precise, conservative contract-review assistant. You never "
    "invent contract language that isn't present in the provided text, and "
    "you always follow the requested output format exactly."
)


def extract_clauses(contract_text: str) -> dict:
    """
    Extract termination / confidentiality / liability clauses from a
    (possibly long) contract. Long contracts are chunked and the per-chunk
    results merged into one final answer.
    """
    chunks = chunk_text(contract_text, max_chars=MAX_CHUNK_CHARS)

    if len(chunks) == 1:
        prompt = build_extraction_prompt(chunks[0])
        raw = _call_llm(SYSTEM_PROMPT, prompt)
        try:
            return _normalize_clause_result(_parse_json_response(raw))
        except (json.JSONDecodeError, ValueError):
            logger.error("Failed to parse extraction response: %s", raw[:500])
            return {key: dict(_EMPTY_CLAUSE) for key in CLAUSE_KEYS}

    # Long contract: extract per chunk, then merge.
    partial_results = []
    for chunk in chunks:
        prompt = build_extraction_prompt(chunk)
        raw = _call_llm(SYSTEM_PROMPT, prompt)
        try:
            partial_results.append(_normalize_clause_result(_parse_json_response(raw)))
        except (json.JSONDecodeError, ValueError):
            logger.warning("Skipping unparsable chunk result: %s", raw[:300])

    if not partial_results:
        return {key: dict(_EMPTY_CLAUSE) for key in CLAUSE_KEYS}
    if len(partial_results) == 1:
        return partial_results[0]

    merge_prompt = build_merge_clauses_prompt(json.dumps(partial_results, indent=2))
    raw = _call_llm(SYSTEM_PROMPT, merge_prompt)
    try:
        return _normalize_clause_result(_parse_json_response(raw))
    except (json.JSONDecodeError, ValueError):
        logger.error("Failed to parse merge response, falling back to first chunk result.")
        return partial_results[0]


def summarize_contract(contract_text: str, max_chars_for_summary: int = 20_000) -> str:
    """
    Generate a 100-150 word summary of the contract. For very long
    contracts, truncate to the first `max_chars_for_summary` characters
    (title/recitals/main body carry most of the purpose+obligations signal;
    boilerplate schedules at the end rarely change the summary).
    """
    text_for_summary = contract_text[:max_chars_for_summary]
    prompt = build_summary_prompt(text_for_summary)
    return _call_llm(SYSTEM_PROMPT, prompt, max_tokens=400).strip()
