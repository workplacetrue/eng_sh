from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from english_life.config import Settings
from english_life.voice.local import LocalSpeech


async def test_say_uses_stdin_without_shell(monkeypatch):
    process = SimpleNamespace(communicate=AsyncMock(), returncode=0)
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr("english_life.voice.local.platform.system", lambda: "Darwin")
    monkeypatch.setattr("english_life.voice.local.shutil.which", lambda _: "/usr/bin/say")
    monkeypatch.setattr("english_life.voice.local.asyncio.create_subprocess_exec", spawn)
    speech = LocalSpeech(Settings(_env_file=None))
    text = "Hello; $(do-not-execute)"
    await speech.speak(text)
    assert spawn.call_args.args == ("say",)
    process.communicate.assert_awaited_once_with(text.encode())


async def test_stt_missing_model_preserves_text_mode(tmp_path):
    speech = LocalSpeech(Settings(_env_file=None, whisper_model=str(tmp_path / "missing.bin")))
    with pytest.raises(RuntimeError, match="whisper-cli"):
        await speech.record_and_transcribe(10)
    with pytest.raises(ValueError):
        await speech.record_and_transcribe(90)


async def test_tts_unavailable_is_explicit(monkeypatch):
    monkeypatch.setattr("english_life.voice.local.platform.system", lambda: "Linux")
    with pytest.raises(RuntimeError, match="macOS"):
        await LocalSpeech(Settings(_env_file=None)).speak("Hello")
