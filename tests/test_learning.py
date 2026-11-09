from datetime import timedelta

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select

from english_life.config import Settings
from english_life.learning.schemas import Evidence, ReviewItem, Turn
from english_life.learning.service import BudgetExceeded, LearningService
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


def test_invalid_result_and_url():
    with pytest.raises(ValidationError):
        Evidence(skill="communication", task="test", demonstrated=True, hints=0, score=101)
    assert (
        Settings(_env_file=None, database_url="postgres://x/db").database_url
        == "postgresql+asyncpg://x/db"
    )
