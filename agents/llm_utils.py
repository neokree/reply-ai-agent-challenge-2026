# agents/llm_utils.py
import re
import json
import logging
from typing import Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from openai import RateLimitError, APIConnectionError

logger = logging.getLogger(__name__)


def strip_thinking_tags(response: str) -> str:
    """
    Remove <think>...</think> blocks from Qwen responses.
    The actual answer comes AFTER the closing </think> tag.
    """
    pattern = r'<think>.*?</think>\s*'
    cleaned = re.sub(pattern, '', response, flags=re.DOTALL)
    return cleaned.strip()


def parse_json_response(response: str) -> dict[str, Any]:
    """
    Parse JSON from LLM response, handling thinking tags and extraction.

    Returns:
        Parsed JSON dict, or {"signals": []} as fallback
    """
    # First strip thinking tags
    cleaned = strip_thinking_tags(response)

    # Try direct JSON parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from text
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback
    logger.warning(f"Could not parse JSON from response: {cleaned[:200]}...")
    return {"signals": []}


def parse_transaction_ids(response: str, valid_ids: set[str]) -> list[str]:
    """
    Parse transaction IDs from coordinator response.

    Raises:
        ValueError: If result is empty or contains all IDs
    """
    # Strip thinking tags
    cleaned = strip_thinking_tags(response)

    # Parse lines
    lines = cleaned.strip().split('\n')
    flagged = []

    for line in lines:
        tx_id = line.strip()
        if tx_id in valid_ids:
            flagged.append(tx_id)

    # Validation
    if len(flagged) == 0:
        raise ValueError("Coordinator returned empty list")

    if len(flagged) == len(valid_ids):
        raise ValueError("Coordinator flagged all transactions")

    return flagged


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError))
)
def call_llm_with_retry(model, messages, config):
    """
    Call LLM with retry logic for transient errors.
    """
    return model.invoke(messages, config=config)
