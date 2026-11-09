from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "sqlite+aiosqlite:///" + str(Path.home() / ".english-life" / "progress.db")
    backend: Literal["ollama", "openai"] = "ollama"
    ollama_url: str = "http://127.0.0.1:11434"
    context_size: int = Field(default=4096, ge=2048, le=8192)
    history_chars: int = Field(default=1800, ge=100, le=4000)
    llm_timeout: int = Field(default=180, ge=5, le=600)
    telegram_bot_token: str = ""
    allowed_telegram_ids: str = ""
    openai_api_key: str = ""
    model: str = "qwen3:4b-q4_K_M"
    stt_model: str = "gpt-4o-mini-transcribe"
    tts_model: str = "gpt-4o-mini-tts"
    voice: str = "alloy"
    whisper_binary: str = "whisper-cli"
    whisper_model: str = "~/.english-life/models/ggml-tiny.en.bin"
    daily_api_calls: int = Field(default=300, ge=1, le=1000)
    max_input_chars: int = Field(default=1200, ge=100, le=10000)
    max_voice_seconds: int = Field(default=90, ge=1, le=300)
    max_output_tokens: int = Field(default=1000, ge=200, le=4000)

    @field_validator("database_url")
    @classmethod
    def async_url(cls, value: str) -> str:
        if value.startswith(("postgres://", "postgresql://")):
            return "postgresql+asyncpg://" + value.split("://", 1)[1]
        return value

    @property
    def allowed_ids(self) -> set[int]:
        return {
            int(value.strip()) for value in self.allowed_telegram_ids.split(",") if value.strip()
        }

    @field_validator("ollama_url")
    @classmethod
    def local_ollama(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("Ollama URL must be a loopback HTTP address")
        return value.rstrip("/")
