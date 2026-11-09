import json

from agents import (
    Agent,
    ModelSettings,
    OpenAIResponsesModel,
    RunConfig,
    Runner,
    set_tracing_disabled,
)
from openai import AsyncOpenAI

from english_life.agents.prompts import ORCHESTRATOR
from english_life.learning.schemas import Turn

set_tracing_disabled(True)


class TutorAgent:
    def __init__(self, settings):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=45, max_retries=0)
        self.agent = Agent(
            name="English Life Orchestrator",
            instructions=ORCHESTRATOR,
            model=OpenAIResponsesModel(model=settings.model, openai_client=self.client),
            output_type=Turn,
            model_settings=ModelSettings(max_tokens=settings.max_output_tokens),
        )

    async def respond(self, context: dict, history: list, message: str) -> Turn:
        result = await Runner.run(
            self.agent,
            input=json.dumps(
                {"context": context, "dialogue": history, "learner": message}, ensure_ascii=False
            ),
            max_turns=1,
            run_config=RunConfig(tracing_disabled=True),
        )
        return Turn.model_validate(result.final_output)

    async def close(self):
        await self.client.close()
