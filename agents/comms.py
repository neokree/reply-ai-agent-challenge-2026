# agents/comms.py
import logging
from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse.langchain import CallbackHandler

from .llm_utils import parse_json_response, call_llm_with_retry

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a fraud analyst specialized in detecting social engineering attacks.

Analyze the provided communications (SMS, emails, audio transcripts) looking for:
- Phishing attempts (credential requests, suspicious links)
- Psychological manipulation (artificial urgency, threats)
- Impersonation (someone pretending to be a bank or service)
- Unusual requests (urgent transfers to "safe" accounts)
- Temporal correlation: suspicious communication received shortly before unusual transactions

User IBAN mapping is provided to help you identify which user received each communication.

Respond ONLY in JSON format:
{"signals": [{"user_iban": "...", "user_name": "...", "source": "sms|email|audio", "severity": "high|medium|low", "timestamp": "...", "reason": "..."}]}

If no suspicious signals are found, respond: {"signals": []}"""


def build_comms_prompt(
    sms: list[dict],
    mails: list[dict],
    audio_transcripts: dict[str, dict],
    user_mapping: dict[str, str]
) -> str:
    """Build the user prompt with all communication data."""
    parts = []

    # User mapping
    if user_mapping:
        parts.append("## User Mapping (Name -> IBAN)")
        for name, iban in user_mapping.items():
            parts.append(f"- {name}: {iban}")
        parts.append("")

    # SMS
    parts.append("## SMS Messages")
    if sms:
        for item in sms:
            parts.append(item.get("sms", str(item)))
        parts.append("")
    else:
        parts.append("No SMS messages.")
        parts.append("")

    # Emails
    parts.append("## Emails")
    if mails:
        for item in mails:
            parts.append(item.get("mail", str(item)))
        parts.append("")
    else:
        parts.append("No emails.")
        parts.append("")

    # Audio transcripts
    parts.append("## Audio Transcripts")
    if audio_transcripts:
        for filename, data in audio_transcripts.items():
            user = data.get("user", "Unknown")
            transcript = data.get("transcript", "")
            ts = data.get("timestamp", "")
            parts.append(f"[{ts}] {user}: {transcript}")
        parts.append("")
    else:
        parts.append("No audio transcripts available.")
        parts.append("")

    return "\n".join(parts)


class CommsAgent:
    """Agent for analyzing communications for social engineering signals."""

    def __init__(self, model, session_id: str):
        self.model = model
        self.session_id = session_id

    def analyze(
        self,
        sms: list[dict],
        mails: list[dict],
        audio_transcripts: dict[str, dict],
        user_mapping: dict[str, str]
    ) -> dict[str, Any]:
        """
        Analyze communications for fraud signals.

        Returns:
            Dict with "signals" list
        """
        try:
            handler = CallbackHandler()
        except Exception:
            handler = None

        user_prompt = build_comms_prompt(sms, mails, audio_transcripts, user_mapping)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]

        config = {"metadata": {"langfuse_session_id": self.session_id}}
        if handler:
            config["callbacks"] = [handler]

        try:
            response = call_llm_with_retry(
                self.model,
                messages,
                config=config
            )

            return parse_json_response(response.content)

        except Exception as e:
            logger.error(f"Comms agent failed: {e}")
            return {"signals": []}
