"""Audio transport boundary: replace this adapter for a future Realtime transport."""

from io import BytesIO

from openai import AsyncOpenAI


class Audio:
    def __init__(self, settings):
        self.settings = settings
        # Bounded timeouts; avoid opaque SDK retries that bypass the call budget.
        self.client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=40, max_retries=0)

    async def transcribe(self, data: bytes) -> str:
        stream = BytesIO(data)
        stream.name = "speech.ogg"
        result = await self.client.audio.transcriptions.create(
            model=self.settings.stt_model, file=stream
        )
        return result.text

    async def speak(self, text: str) -> bytes:
        response = await self.client.audio.speech.create(
            model=self.settings.tts_model,
            voice=self.settings.voice,
            input=text[:3500],
            response_format="opus",
        )
        return response.content

    async def close(self):
        await self.client.close()
