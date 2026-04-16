# tests/test_audio.py
import pytest
from preprocessing.audio import extract_user_from_filename, AudioTranscriber

def test_extract_user_from_filename():
    filename = "20870117_010505-guido_döhn.mp3"
    user = extract_user_from_filename(filename)
    assert user == "guido döhn"

def test_extract_user_handles_underscores():
    filename = "20870206_221040-juliette_brunet.mp3"
    user = extract_user_from_filename(filename)
    assert user == "juliette brunet"

def test_audio_transcriber_init():
    # Just test initialization doesn't crash when API key missing
    # AudioTranscriber raises ValueError if no key - that's acceptable behavior
    import os
    if not os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        with pytest.raises(ValueError):
            AudioTranscriber()
    else:
        transcriber = AudioTranscriber()
        assert transcriber is not None

# Integration test - skip if no API key
@pytest.mark.skip(reason="Requires OpenAI API key and real audio file")
def test_audio_transcriber_transcribe():
    transcriber = AudioTranscriber()
    result = transcriber.transcribe("path/to/test.mp3")
    assert isinstance(result, str)
