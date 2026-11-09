"""Transport-independent agent contract. LearningService knows only respond()."""

from typing import Protocol

from english_life.learning.schemas import Turn


class TutorBackend(Protocol):
    async def respond(self, context: dict, history: list, message: str) -> Turn: ...


class DemoAgent:
    """Offline wiring check; not a language model or assessment."""

    async def respond(self, context, history, message):
        if context["force_finish"]:
            return Turn(
                reply="Демо завершено. Оценка навыков без LLM не выполнялась.",
                finished=True,
                summary="Offline demo; no assessment.",
            )
        return Turn(reply="[Демо без LLM] You are at a café. What would you like to order?")


def create_backend(settings):
    if settings.backend == "ollama":
        from english_life.agents.ollama_backend import OllamaAgent

        return OllamaAgent(settings)
    if settings.backend == "openai":
        try:
            from english_life.agents.openai_backend import TutorAgent
        except ImportError as error:
            raise RuntimeError(
                "API-адаптер необязателен: установите uv sync --extra legacy"
            ) from error

        return TutorAgent(settings)
    raise ValueError("Unknown LLM backend")
