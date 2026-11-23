"""Optional offline audio, loaded only on demand; never alongside another neural TTS model."""

import asyncio
import platform
import re
import shutil
import tempfile
from pathlib import Path
from typing import Protocol

LETTERS = re.compile(r"[A-Za-z]|[А-Яа-яЁё]")
SAY_VOICE = re.compile(r"^(.+?)\s+(en|ru)[_-][A-Z]{2}\s+#")
PREFERRED_VOICES = {"en": "Samantha", "ru": "Milena"}


def speech_segments(text):
    language = None
    start = 0
    for match in LETTERS.finditer(text):
        current = "en" if match[0].isascii() else "ru"
        if language and current != language:
            yield language, text[start : match.start()].strip()
            start = match.start()
        language = current
    if language:
        yield language, text[start:].strip()


class SpeechAdapter(Protocol):
    async def speak(self, text: str) -> None: ...
    async def record_and_transcribe(self, seconds: int) -> str: ...


class LocalSpeech:
    def __init__(self, settings):
        self.settings = settings
        self.voices = None

    def check_tts(self):
        if platform.system() != "Darwin" or not shutil.which("say"):
            raise RuntimeError("Локальная озвучка требует macOS say; текстовый режим доступен.")

    async def available_voices(self):
        process = await asyncio.create_subprocess_exec(
            "say",
            "-v",
            "?",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        output, _ = await process.communicate()
        if process.returncode:
            raise RuntimeError("Не удалось получить список голосов macOS say.")
        voices = {}
        for line in output.decode(errors="replace").splitlines():
            match = SAY_VOICE.match(line)
            if match:
                voice, language = match.groups()
                if language not in voices or voice == PREFERRED_VOICES[language]:
                    voices[language] = voice
        return voices

    async def speak(self, text: str):
        self.check_tts()
        if self.voices is None:
            self.voices = await self.available_voices()
        segments = list(speech_segments(text))
        for language, _ in segments:
            if language not in self.voices:
                name = "русский" if language == "ru" else "английский"
                raise RuntimeError(f"Установите {name} голос в настройках речи macOS.")
        for language, segment in segments:
            # stdin avoids treating model-produced text as shell or command arguments.
            process = await asyncio.create_subprocess_exec(
                "say",
                "-v",
                self.voices[language],
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await process.communicate(segment.encode())
            if process.returncode:
                raise RuntimeError(
                    "macOS say не смогла произнести текст. Проверьте системные голоса."
                )

    async def record_and_transcribe(self, seconds: int) -> str:
        if not 1 <= seconds <= 30:
            raise ValueError("Запись: от 1 до 30 секунд.")
        binary = shutil.which(self.settings.whisper_binary)
        model = Path(self.settings.whisper_model).expanduser()
        if not binary or not model.is_file():
            raise RuntimeError(
                "Для записи нужны whisper-cli и локальная модель WHISPER_MODEL. См. README."
            )
        try:
            import sounddevice as sd
            import soundfile as sf
        except ImportError as error:
            raise RuntimeError("Установите голосовые зависимости: uv sync --extra voice") from error

        def record(path):
            print(f"Говорите по-английски ({seconds} секунд)…")
            samples = sd.rec(seconds * 16000, samplerate=16000, channels=1, dtype="float32")
            sd.wait()
            sf.write(path, samples, 16000, subtype="PCM_16")

        # Capture and inference are sequential, with immediate temporary-file cleanup.
        with tempfile.TemporaryDirectory(prefix="english-life-audio-") as directory:
            wav = Path(directory) / "input.wav"
            output = Path(directory) / "transcript"
            await asyncio.to_thread(record, wav)
            process = await asyncio.create_subprocess_exec(
                binary,
                "-m",
                str(model),
                "-f",
                str(wav),
                "-l",
                "en",
                "-t",
                "4",
                "-otxt",
                "-of",
                str(output),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                await asyncio.wait_for(process.wait(), timeout=120)
            except (TimeoutError, asyncio.CancelledError):
                process.kill()
                await process.wait()
                raise
            if process.returncode or not output.with_suffix(".txt").exists():
                raise RuntimeError("Whisper не смог распознать речь. Проверьте модель и микрофон.")
            text = output.with_suffix(".txt").read_text().strip()
            if not text:
                raise RuntimeError("Речь не распознана. Попробуйте ещё раз.")
            return text
