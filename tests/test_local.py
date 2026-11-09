import json
import os
import subprocess
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select

from english_life.agents.ollama_backend import BackendUnavailable, InvalidModelOutput, OllamaAgent
from english_life.cli import chat, single_instance
from english_life.config import Settings
from english_life.learning.schemas import Turn
from english_life.learning.service import LearningService
from english_life.memory.db import create_local_schema, database
from english_life.memory.models import Lesson, Profile, Receipt


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
        return httpx.Response(
            200, json={"done": True, "message": {"content": Turn(reply="Hello!").model_dump_json()}}
        )

    client = httpx.AsyncClient(base_url=settings.ollama_url, transport=httpx.MockTransport(handler))
    backend = OllamaAgent(settings, client)
    assert (await backend.check())["quantization"] == "Q4_K_M"
    history = [{"role": "user", "content": str(i) + "x" * 500} for i in range(10)]
    assert (
        await backend.respond({"skills": {}, "force_finish": False}, history, "Hi")
    ).reply == "Hello!"
    payload = observed[-1]
    assert payload["think"] is False
    assert payload["stream"] is False
    assert payload["options"]["num_ctx"] == 4096
    assert payload["format"] == Turn.model_json_schema()
    assert len(payload["messages"]) < 10
    assert payload["messages"][-1] == {"role": "user", "content": "Hi"}
    await backend.close()


@pytest.mark.parametrize(
    "body",
    [
        {"message": {"content": "not JSON"}},
        {"message": {"content": '{"reply":"Hello", "evidence":[{"skill":"unknown"}]}'}},
        {"done_reason": "length", "message": {"content": '{"reply":"partial"}'}},
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
    from datetime import timedelta

    from english_life.agents.runner import DemoAgent
    from english_life.memory.models import utcnow

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
