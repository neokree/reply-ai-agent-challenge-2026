# preprocessing/audio.py
import os
import logging
from pathlib import Path
from typing import Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

def extract_user_from_filename(filename: str) -> str:
    """
    Extract user name from audio filename.
    Format: YYYYMMDD_HHMMSS-name_surname.mp3

    Returns:
        User name with spaces (e.g., "guido döhn")
    """
    # Remove extension
    name = Path(filename).stem

    # Extract part after the dash
    if "-" in name:
        name = name.split("-", 1)[1]

    # Replace underscores with spaces
    return name.replace("_", " ")


class AudioTranscriber:
    """Transcribe audio files using OpenAI Whisper API."""

    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY")

        self.client = OpenAI(api_key=api_key)

    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file."""
        with open(audio_path, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        return transcript

    def transcribe_directory(self, audio_dir: str) -> dict[str, dict]:
        """
        Transcribe all audio files in a directory.

        Returns:
            Dict mapping filename to {user: str, transcript: str}
        """
        results = {}
        audio_path = Path(audio_dir)

        if not audio_path.exists():
            logger.warning(f"Audio directory not found: {audio_dir}")
            return results

        for audio_file in audio_path.glob("*.mp3"):
            try:
                user = extract_user_from_filename(audio_file.name)
                transcript = self.transcribe(str(audio_file))
                results[audio_file.name] = {
                    "user": user,
                    "transcript": transcript,
                    "timestamp": audio_file.name[:15],  # YYYYMMDD_HHMMSS
                }
                logger.info(f"Transcribed: {audio_file.name}")
            except Exception as e:
                logger.error(f"Failed to transcribe {audio_file.name}: {e}")

        return results
