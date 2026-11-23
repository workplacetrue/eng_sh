import json
import os
import subprocess
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy import select

from english_life.agents.ollama_backend import (
    BackendUnavailable,
    InvalidModelOutput,
    OllamaAgent,
    OllamaTurn,
)
from english_life.agents.runner import DemoAgent
from english_life.cli import chat, single_instance
from english_life.config import Settings
from english_life.learning.schemas import Turn
from english_life.learning.service import LISTENING_QUESTION, LISTENING_SCRIPT, LearningService
from english_life.memory.db import create_local_schema, database
from english_life.memory.models import Lesson, Profile, Receipt, utcnow
from english_life.voice.local import LocalSpeech


def settings_for(tmp_path):
    return Settings(
        _env_file=None, database_url=f"sqlite+aiosqlite:///{tmp_path}/nested/progress.db"
    )


async def test_ollama_contract_and_context_bound(tmp_path):
    settings = settings_for(tmp_path)
    observed = []

    def handler(request):
        payload = json.loads(request.content)
        observed.append(payload)
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"details": {"quantization_level": "Q4_K_M"}})
        turn = OllamaTurn(reply="Yes, oat milk is available. What size would you like?")
        return httpx.Response(
            200, json={"done": True, "message": {"content": turn.model_dump_json()}}
        )

    client = httpx.AsyncClient(base_url=settings.ollama_url, transport=httpx.MockTransport(handler))
    backend = OllamaAgent(settings, client)
    assert (await backend.check())["quantization"] == "Q4_K_M"
    history = [
        {"role": "user", "content": "Begin the session."},
        {"role": "assistant", "content": "Вы покупаете кофе.\n\nWhat can I get you?"},
    ] + [
        {"role": role, "content": f"{role} {i} " + "x" * 1000}
        for i in range(10)
        for role in ("user", "assistant")
    ]
    context = {"skills": {}, "force_finish": False, "topic": "Cafe", "mode": "lesson"}
    assert (await backend.respond(context, history, "Do you have oat milk?")).reply.startswith(
        "Yes, oat milk"
    )
    payload = observed[-1]
    assert payload["think"] is False
    assert payload["stream"] is False
    assert payload["options"]["num_ctx"] == 4096
    assert payload["format"] == OllamaTurn.model_json_schema()
    messages = payload["messages"]
    assert '"topic": "Cafe"' in messages[1]["content"]
    assert messages[2:4] == history[:2]
    assert [turn["role"] for turn in messages[2:-1]] == ["user", "assistant"] * 2
    assert messages[-3]["content"].startswith("user 9 ")
    assert messages[-2]["content"].startswith("assistant 9 ")
    assert sum(len(turn["content"]) for turn in messages[2:-1]) <= settings.history_chars
    assert messages[-1] == {"role": "user", "content": "Do you have oat milk?"}
    await backend.close()


@pytest.mark.parametrize(
    "body",
    [
        {"message": {"content": "not JSON"}},
        {"message": {"content": '{"reply":"Hello", "evidence":[{"skill":"unknown"}]}'}},
        {"done_reason": "length", "message": {"content": '{"reply":"partial"}'}},
        {"message": {"content": '{"reply":"  \\n "}'}},
        {
            "message": {
                "content": OllamaTurn(
                    reply="Where was she?", instruction_ru="Прочитайте текст."
                ).model_dump_json()
            }
        },
        {
            "message": {
                "content": OllamaTurn(
                    reply="Where was she?",
                    instruction_ru="Read the text.",
                    reading_passage="Mia was at the cafe.",
                ).model_dump_json()
            }
        },
        {
            "message": {
                "content": OllamaTurn(
                    reply="Where was she",
                    instruction_ru="Прочитайте текст.",
                    reading_passage="Mia was at the cafe.",
                ).model_dump_json()
            }
        },
        [],
        {},
    ],
)
async def test_invalid_output_never_writes_progress(tmp_path, body):
    settings = settings_for(tmp_path)
    client = httpx.AsyncClient(
        base_url=settings.ollama_url,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)),
    )
    backend = OllamaAgent(settings, client)
    engine, sessions = database(settings.database_url)
    await create_local_schema(engine)
    service = LearningService(sessions, backend, settings)
    with pytest.raises(InvalidModelOutput):
        await service.handle(1, 1, "/start")
    async with sessions() as db:
        assert (await db.get(Profile, 1)).skills == {}
        assert await db.scalar(select(Lesson)) is None
        assert await db.get(Receipt, (1, 1)) is None
    await backend.close()
    await engine.dispose()


async def test_ollama_diagnostic_displays_passage_before_question(tmp_path):
    settings = settings_for(tmp_path)
    turn = OllamaTurn(
        instruction_ru="Прочитайте текст и ответьте по-английски.",
        reading_passage="Mia went to a café on Monday. She ordered tea.",
        reply="What did Mia order?",
    )
    async with httpx.AsyncClient(
        base_url=settings.ollama_url,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"message": {"content": turn.model_dump_json()}}
            )
        ),
    ) as client:
        reply = await OllamaAgent(settings, client).respond({"mode": "diagnostic"}, [], "/start")
    assert reply.reply.split("\n\n") == [turn.instruction_ru, turn.reading_passage, turn.reply]


async def test_ollama_rejects_multiline_exercise_in_lesson(tmp_path):
    settings = settings_for(tmp_path)
    async with httpx.AsyncClient(
        base_url=settings.ollama_url,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "message": {
                        "content": json.dumps(
                            {
                                "reply": "1. Translate this sentence.\n2. Fill in the blank.",
                            }
                        )
                    }
                },
            )
        ),
    ) as client:
        with pytest.raises(InvalidModelOutput):
            await OllamaAgent(settings, client).respond(
                {"mode": "lesson", "topic": "Cafe"},
                [{"role": "user", "content": "Start"}, {"role": "assistant", "content": "Hello!"}],
                "A latte, please.",
            )


async def test_server_error_no_hidden_retry(tmp_path):
    calls = []

    def fail(request):
        calls.append(1)
        return httpx.Response(503)

    client = httpx.AsyncClient(base_url="http://localhost", transport=httpx.MockTransport(fail))
    backend = OllamaAgent(settings_for(tmp_path), client)
    with pytest.raises(BackendUnavailable):
        await backend.respond({}, [], "Hello")
    assert len(calls) == 1
    await backend.close()


def test_loopback_only():
    from pydantic import ValidationError

    for url in [
        "https://api.example.com",
        "http://localhost@evil.com",
        "http://localhost?query=bad",
    ]:
        with pytest.raises(ValidationError):
            Settings(_env_file=None, ollama_url=url)


def test_one_process_per_db(tmp_path):
    settings = settings_for(tmp_path)
    with single_instance(settings.database_url):
        with pytest.raises(RuntimeError, match="Другой English Life"):
            with single_instance(settings.database_url):
                pass
    with single_instance(settings.database_url):
        pass


async def test_cli_persists_without_openai_or_telegram(tmp_path, monkeypatch, capsys):
    settings = settings_for(tmp_path)
    commands = iter(["/start", "A tea, please", "quit"])

    async def read(_):
        return next(commands)

    monkeypatch.setattr("english_life.cli.read_line", read)
    await chat(settings, demo=True)
    engine, sessions = database(settings.database_url)
    async with sessions() as db:
        lesson = await db.scalar(select(Lesson))
        assert lesson.status == "paused"
        assert any("A tea" in x["content"] for x in lesson.state)
    await engine.dispose()
    commands = iter(["/resume", "/finish", "/progress", "quit"])
    await chat(settings, demo=True)
    assert "Карта навыков" in capsys.readouterr().out
    engine, sessions = database(settings.database_url)
    async with sessions() as db:
        lesson = await db.scalar(select(Lesson))
        assert lesson.status == "completed"
    await engine.dispose()
    assert Path(tmp_path / "nested/progress.db").stat().st_mode & 0o077 == 0


def test_macos_installer_syntax_and_linux_refusal():
    subprocess.run(["bash", "-n", "scripts/install-macos.sh", "start.sh"], check=True)
    import platform

    if platform.system() != "Darwin":
        result = subprocess.run(
            ["bash", "scripts/install-macos.sh"], capture_output=True, text=True
        )
        assert result.returncode == 1 and "Apple Silicon" in result.stderr


def test_offline_demo_subprocess_twice(tmp_path):
    env = {**os.environ, "DATABASE_URL": f"sqlite+aiosqlite:///{tmp_path}/smoke.db"}
    for _ in range(2):
        result = subprocess.run(
            [".venv/bin/english-life", "demo", "--smoke"],
            env=env,
            text=True,
            capture_output=True,
            timeout=20,
        )
        assert result.returncode == 0, result.stderr
        assert "Карта навыков" in result.stdout


async def test_cli_closes_expired_lesson_without_user_input(tmp_path, monkeypatch, capsys):
    settings = settings_for(tmp_path)
    engine, sessions = database(settings.database_url)
    await create_local_schema(engine)
    service = LearningService(sessions, DemoAgent(), settings)
    await service.handle(1, 1, "/start")
    async with sessions.begin() as db:
        lesson = await db.scalar(select(Lesson))
        lesson.deadline = utcnow() - timedelta(seconds=1)
    await engine.dispose()

    async def read(_):
        return "quit"

    monkeypatch.setattr("english_life.cli.read_line", read)
    await chat(settings, demo=True)
    assert "Время занятия истекло" in capsys.readouterr().out
    engine, sessions = database(settings.database_url)
    async with sessions() as db:
        assert (await db.scalar(select(Lesson))).status == "completed"
    await engine.dispose()


async def test_cli_failed_timeout_stays_paused_until_finish(tmp_path, monkeypatch, capsys):
    finish_calls = []

    class Agent(DemoAgent):
        async def respond(self, context, history, message):
            if context["force_finish"]:
                finish_calls.append((message, history))
                if message == "/timeout":
                    return Turn(reply="What would you like next?")
            return await super().respond(context, history, message)

    settings = settings_for(tmp_path)
    engine, sessions = database(settings.database_url)
    await create_local_schema(engine)
    agent = Agent()
    service = LearningService(sessions, agent, settings)
    await service.handle(1, 1, "/start")
    await service.handle(1, 2, "A coffee, please.")
    async with sessions.begin() as db:
        lesson = await db.scalar(select(Lesson))
        lesson.deadline = utcnow() - timedelta(seconds=1)

    commands = iter(["/resume", "/finish", "quit"])

    async def read(_):
        return next(commands)

    monkeypatch.setattr("english_life.cli.read_line", read)
    monkeypatch.setattr("english_life.cli.create_backend", lambda _: agent)
    await chat(settings)
    output = capsys.readouterr().out
    assert output.count("Подвожу итоги") == 1
    assert "Автозавершение отложено" in output
    assert "Используйте /finish, чтобы подвести итоги" in output
    assert [message for message, _ in finish_calls] == ["/timeout", "/finish"]
    assert finish_calls[1][1][2]["content"] == "A coffee, please."
    async with sessions() as db:
        assert (await db.scalar(select(Lesson))).status == "completed"
    await engine.dispose()


async def test_cli_listen_plays_english_sample_and_russian_question(tmp_path, monkeypatch, capsys):
    commands = iter(["/voice", "/start", "/listen", "/pause", "/resume", "quit"])

    async def read(_):
        return next(commands)

    speak = AsyncMock()
    monkeypatch.setattr("english_life.cli.read_line", read)
    monkeypatch.setattr(LocalSpeech, "check_tts", lambda self: None)
    monkeypatch.setattr(LocalSpeech, "speak", speak)
    await chat(settings_for(tmp_path), demo=True)
    output = capsys.readouterr().out
    assert LISTENING_SCRIPT not in output
    assert output.count(LISTENING_QUESTION) == 2
    assert [call.args[0] for call in speak.await_args_list][1:3] == [
        LISTENING_SCRIPT,
        LISTENING_QUESTION,
    ]
    assert speak.await_args.args[0].endswith(LISTENING_QUESTION)
