from datetime import timedelta

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select

from english_life.config import Settings
from english_life.learning.schemas import Evidence, ReviewItem, Turn
from english_life.learning.service import LISTENING_QUESTION, BudgetExceeded, LearningService, aware
from english_life.memory.db import create_local_schema, database
from english_life.memory.models import Base, Lesson, Profile, Receipt, Review, Snapshot, utcnow


class FakeAgent:
    def __init__(self):
        self.calls = []
        self.fail = False

    async def respond(self, context, history, message):
        self.calls.append((context, history, message))
        if self.fail:
            raise RuntimeError("provider unavailable")
        if not context["force_finish"]:
            return Turn(reply="What would you like?")
        return Turn(
            reply="Well done!",
            finished=True,
            summary="Ordered coffee independently.",
            evidence=[
                Evidence(skill=k, task="Order coffee", demonstrated=True, hints=0, score=40)
                for k in ["communication", "listening", "speaking_independence"]
            ],
            reviews=[ReviewItem(phrase="I'd like", prompt="Order politely", kind="expression")],
            mastered_situations=["Ordering coffee"],
        )


@pytest.fixture
async def env(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path}/test.db"
    engine, sessions = database(url)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
    await create_local_schema(engine)
    settings = Settings(_env_file=None, database_url=url, daily_api_calls=60)
    agent = FakeAgent()
    service = LearningService(sessions, agent, settings)
    yield service, agent, sessions
    await engine.dispose()


async def test_completion_restart_isolation_and_delete(env):
    service, agent, sessions = env
    await service.handle(11, 1, "/start")
    await service.handle(11, 2, "Coffee please")
    # New service instance reads the unfinished conversation from DB.
    restarted = LearningService(sessions, agent, service.settings)
    await restarted.handle(11, 3, "/finish")
    assert len(agent.calls[-1][1]) == 4
    progress = await restarted.handle(11, 4, "/progress")
    assert progress.splitlines()[0] == "Карта навыков · предварительный уровень A2 · цель B1"
    assert "40/100" in progress
    assert "Listening comprehension: пока" in progress
    assert "Speaking independence: пока" in progress
    assert "40/100" not in await restarted.handle(12, 4, "/progress")
    async with sessions() as db:
        lesson = await db.scalar(select(Lesson).where(Lesson.user_id == 11))
        assert lesson.status == "completed" and lesson.state == []
        assert await db.scalar(select(func.count()).select_from(Snapshot)) == 1
        review = await db.scalar(select(Review))
        assert (
            review.due > utcnow().replace(tzinfo=None)
            if review.due.tzinfo is None
            else review.due > utcnow()
        )
    await restarted.handle(11, 5, "/delete_me")
    async with sessions() as db:
        for model in (Profile, Lesson, Review, Snapshot, Receipt):
            assert (
                await db.scalar(select(func.count()).select_from(model).where(model.user_id == 11))
                == 0
            )
        assert await db.get(Profile, 12) is not None


async def test_duplicate_and_api_failure_rollback(env):
    service, agent, sessions = env
    original = await service.handle(11, 1, "/start")
    assert await service.handle(11, 1, "/start") == original
    assert len(agent.calls) == 1
    agent.fail = True
    with pytest.raises(RuntimeError):
        await service.handle(11, 2, "Hello")
    async with sessions() as db:
        lesson = await db.scalar(select(Lesson))
        assert len(lesson.state) == 2
        assert await db.get(Receipt, (11, 2)) is None
        assert (await db.get(Profile, 11)).api_calls == 2
    agent.fail = False
    await service.handle(11, 2, "Hello")
    assert len(agent.calls[-1][1]) == 2


@pytest.mark.parametrize("minutes", [5, 10, 30])
async def test_duration_timeout_and_pause(env, minutes):
    service, agent, sessions = env
    await service.handle(11, 1, f"/lesson {minutes}")
    async with sessions() as db:
        lesson = await db.scalar(select(Lesson))
        assert (lesson.deadline - lesson.started).total_seconds() == minutes * 60
    await service.handle(11, 2, "/pause")
    count = len(agent.calls)
    await service.handle(11, 3, f"/lesson {minutes}")
    assert len(agent.calls) == count
    async with sessions.begin() as db:
        lesson = await db.scalar(select(Lesson))
        lesson.deadline = utcnow() - timedelta(seconds=1)
    await service.handle(11, 4, "Late answer")
    assert agent.calls[-1][0]["force_finish"]
    async with sessions() as db:
        assert (await db.scalar(select(Lesson))).status == "completed"


async def test_budget_due_reviews_and_listening(env):
    service, agent, sessions = env
    await service.handle(11, 1, "/start")
    await service.listening_delivered(11)
    await service.handle(11, 2, "Ten thirty", "voice")
    await service.handle(11, 3, "/finish")
    async with sessions.begin() as db:
        profile = await db.get(Profile, 11)
        assert "listening" in profile.skills
        assert "speaking_independence" in profile.skills
        review = await db.scalar(select(Review))
        review.due = utcnow() - timedelta(days=1)
    await service.handle(11, 4, "/repeat")
    assert "I'd like" in agent.calls[-1][0]["topic"]
    service.settings.daily_api_calls = 4
    with pytest.raises(BudgetExceeded):
        await service.handle(11, 5, "Hello")
    assert "40/100" in await service.handle(11, 6, "/progress")


async def test_resume_restores_fractional_time_and_keeps_scenario(env, monkeypatch):
    service, agent, sessions = env
    now = utcnow()
    monkeypatch.setattr("english_life.learning.service.utcnow", lambda: now)
    respond = agent.respond

    async def slow_opening(context, history, message):
        nonlocal now
        if not history:
            now += timedelta(seconds=45)
        return await respond(context, history, message)

    monkeypatch.setattr(agent, "respond", slow_opening)
    opening = await service.handle(11, 1, "/lesson 10")
    async with sessions() as db:
        original = await service.active(db, 11)
        assert aware(original.deadline) == now + timedelta(minutes=10)
    now += timedelta(minutes=10) - timedelta(milliseconds=400)
    await service.handle(11, 2, "/pause")
    now += timedelta(hours=3)
    assert "на паузе" in await service.handle(11, 3, "/start")
    await service.handle(11, 4, "/pause")
    service.settings.daily_api_calls = 1
    restarted = LearningService(sessions, agent, service.settings)
    assert opening in await restarted.handle(11, 5, "/resume")
    async with sessions() as db:
        resumed = await restarted.active(db, 11)
        assert (resumed.id, resumed.minutes, resumed.topic) == (original.id, 10, original.topic)
        assert aware(resumed.deadline) == now + timedelta(milliseconds=400)
        assert "remaining_seconds" not in resumed.result
        assert (await db.get(Profile, 11)).api_calls == 1
    assert await restarted.handle(11, 6, "/resume") == opening
    assert "ещё идёт" in await restarted.handle(11, 7, "/timeout")
    now += timedelta(milliseconds=400)
    assert "Ответов для оценки пока нет" in await restarted.handle(11, 8, "/timeout")
    assert len(agent.calls) == 1


async def test_finish_paused_without_answers_does_not_assess(env):
    service, agent, sessions = env
    service.settings.daily_api_calls = 1
    await service.handle(11, 1, "/start")
    assert "Введите ответ" in await service.handle(11, 2, "  ")
    await service.handle(11, 3, "/pause")
    assert "Ответов для оценки пока нет" in await service.handle(11, 4, "/finish")
    assert "Нет незавершённого занятия" in await service.handle(11, 5, "/resume")
    assert len(agent.calls) == 1
    async with sessions() as db:
        profile = await db.get(Profile, 11)
        assert not profile.diagnosed and profile.skills == {}
        lesson = await db.scalar(select(Lesson))
        assert lesson.status == "completed" and lesson.result["evidence"] == []


async def test_premature_finish_rolls_back_opening(env, monkeypatch):
    service, agent, sessions = env

    async def premature(*args):
        return Turn(reply="Занятие завершено.", summary="Итог.", finished=True)

    monkeypatch.setattr(agent, "respond", premature)
    with pytest.raises(ValueError, match="до первого задания"):
        await service.handle(11, 1, "/start")
    async with sessions() as db:
        assert await db.scalar(select(Lesson)) is None
        assert await db.get(Receipt, (11, 1)) is None
        assert not (await db.get(Profile, 11)).diagnosed


async def test_listening_question_survives_pause_and_context_keeps_response(env):
    service, agent, sessions = env
    await service.handle(11, 1, "/start")
    await service.listening_delivered(11)
    await service.handle(11, 2, "/pause")
    restarted = LearningService(sessions, agent, service.settings)
    assert LISTENING_QUESTION in await restarted.handle(11, 3, "/resume")
    await restarted.handle(11, 4, "Ten thirty, platform four.", "voice")
    assert agent.calls[-1][0]["listening_question"] == LISTENING_QUESTION
    await restarted.handle(11, 5, "/finish")
    context, history, _ = agent.calls[-1]
    assert "listening_question" not in context
    assert history[-2]["in_reply_to"] == LISTENING_QUESTION


async def test_old_voice_answer_still_counts_after_history_is_trimmed(env):
    service, agent, sessions = env
    await service.handle(11, 1, "/start")
    await service.handle(11, 2, "A coffee, please.", "voice")
    for index in range(3, 14):
        await service.handle(11, index, "Another answer.")
    await service.handle(11, 14, "/finish")
    context, history, _ = agent.calls[-1]
    assert not any(turn.get("mode") == "voice" for turn in history)
    assert context["topic"] == agent.calls[0][0]["topic"]
    assert history[:2] == agent.calls[1][1]
    async with sessions() as db:
        assert "speaking_independence" in (await db.get(Profile, 11)).skills


def test_invalid_result_and_url():
    with pytest.raises(ValidationError):
        Evidence(skill="communication", task="test", demonstrated=True, hints=0, score=101)
    assert (
        Settings(_env_file=None, database_url="postgres://x/db").database_url
        == "postgresql+asyncpg://x/db"
    )
