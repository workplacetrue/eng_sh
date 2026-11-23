from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from english_life.config import Settings
from english_life.learning.service import LISTENING_SCRIPT
from english_life.voice.local import LocalSpeech


@pytest.fixture
def mac_say(monkeypatch):
    monkeypatch.setattr("english_life.voice.local.platform.system", lambda: "Darwin")
    monkeypatch.setattr("english_life.voice.local.shutil.which", lambda _: "/usr/bin/say")
    spawn = AsyncMock()
    monkeypatch.setattr("english_life.voice.local.asyncio.create_subprocess_exec", spawn)
    return spawn


async def test_say_selects_russian_and_english_voices_and_passes_stdin(mac_say):
    voices = b"Alex    en_US    # Hello\nSamantha    en_US    # Hello\nMilena    ru_RU    # Text\n"
    listing = SimpleNamespace(communicate=AsyncMock(return_value=(voices, None)), returncode=0)
    russian, english, listening = [
        SimpleNamespace(communicate=AsyncMock(), returncode=0) for _ in range(3)
    ]
    mac_say.side_effect = [listing, russian, english, listening]
    speech = LocalSpeech(Settings(_env_file=None))
    await speech.speak("Закажите напиток.\n\nHello; $(do-not-execute)")
    await speech.speak(LISTENING_SCRIPT)
    assert [call.args for call in mac_say.await_args_list] == [
        ("say", "-v", "?"),
        ("say", "-v", "Milena"),
        ("say", "-v", "Samantha"),
        ("say", "-v", "Samantha"),
    ]
    russian.communicate.assert_awaited_once_with("Закажите напиток.".encode())
    english.communicate.assert_awaited_once_with(b"Hello; $(do-not-execute)")
    listening.communicate.assert_awaited_once_with(LISTENING_SCRIPT.encode())


async def test_say_reports_missing_russian_voice(mac_say):
    mac_say.return_value = SimpleNamespace(
        communicate=AsyncMock(return_value=(b"Samantha    en_US    # Hello\n", None)), returncode=0
    )
    with pytest.raises(RuntimeError, match="русский голос"):
        await LocalSpeech(Settings(_env_file=None)).speak("Закажите напиток.\n\nHello!")
    assert mac_say.await_count == 1


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
