import asyncio
from collections import defaultdict
from datetime import UTC, timedelta

from sqlalchemy import delete, func, select

from english_life.learning.planner import choose_scenario
from english_life.memory.models import Lesson, Profile, Receipt, Review, utcnow
from english_life.tools.memory import save_result

SKILLS = {
    "reading": "Reading comprehension",
    "listening": "Listening comprehension",
    "speaking_independence": "Speaking independence",
    "grammar": "Grammar in speech",
    "vocabulary": "Active vocabulary",
    "communication": "Functional communication",
}
LISTENING_SCRIPT = "The train to Brighton leaves at ten thirty from platform four. A return ticket costs twelve pounds."


def aware(value):
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


class BudgetExceeded(Exception):
    pass


class LearningService:
    def __init__(self, sessions, agent, settings):
        self.sessions, self.agent, self.settings = sessions, agent, settings
        self.locks = defaultdict(asyncio.Lock)

    async def reserve_api(self, user_id):
        async with self.sessions.begin() as db:
            profile = await db.get(Profile, user_id, with_for_update=True)
            if profile is None:
                raise ValueError("Сначала /start")
            day = utcnow().date().isoformat()
            if profile.budget_day != day:
                profile.budget_day, profile.api_calls = day, 0
            if profile.api_calls >= self.settings.daily_api_calls:
                raise BudgetExceeded("Дневной лимит запросов исчерпан. Продолжим завтра.")
            profile.api_calls += 1

    async def active(self, db, user_id):
        return await db.scalar(
            select(Lesson)
            .where(Lesson.user_id == user_id, Lesson.status == "active")
            .order_by(Lesson.id.desc())
        )

    async def handle(self, user_id: int, update_id: int, message: str, input_mode="text") -> str:
        async with self.locks[user_id]:
            # Charge before the external call, separately from the lesson transaction:
            # failed API requests count too, bounding retries and daily spend.
            async with self.sessions.begin() as db:
                profile = await db.get(Profile, user_id)
                if profile is None:
                    db.add(Profile(user_id=user_id))
            async with self.sessions() as db:
                receipt = await db.get(Receipt, (user_id, update_id))
                if receipt:
                    return receipt.reply
            if message == "/delete_me":
                async with self.sessions.begin() as db:
                    await db.execute(delete(Profile).where(Profile.user_id == user_id))
                return "Учебные данные удалены. Для нового профиля: /start."
            if len(message) > self.settings.max_input_chars:
                return "Сообщение слишком длинное. Отправьте более короткий ответ."
            async with self.sessions() as db:
                profile = await db.get(Profile, user_id)
                lesson = await self.active(db, user_id)
                local = message in {"/progress", "/weekly", "/voice", "/text", "/lesson", "/pause"}
                if not local and (
                    lesson is not None
                    or message in {"/start", "/repeat", "/lesson 5", "/lesson 10", "/lesson 30"}
                ):
                    needs_call = True
                else:
                    needs_call = False
            if needs_call:
                await self.reserve_api(user_id)
            async with self.sessions.begin() as db:
                profile = await db.get(Profile, user_id, with_for_update=True)
                lesson = await self.active(db, user_id)
                reply = await self._handle(db, profile, lesson, message, input_mode)
                db.add(Receipt(user_id=user_id, update_id=update_id, reply=reply))
                # Keep replay data bounded and separate from educational history.
                await db.execute(
                    delete(Receipt).where(
                        Receipt.user_id == user_id, Receipt.update_id < update_id - 10000
                    )
                )
                return reply

    async def _handle(self, db, profile, lesson, message, input_mode):
        now = utcnow()
        if message in {"/voice", "/text"}:
            profile.audio_reply = message == "/voice"
            return (
                "Голосовые ответы включены."
                if profile.audio_reply
                else "Текстовые ответы включены."
            )
        if message == "/lesson":
            return "Выберите продолжительность: 5, 10 или 30 минут."
        if message in {"/progress", "/weekly"}:
            lines = ["Карта навыков · цель B1 → B2 (не официальный CEFR)."]
            for key, label in SKILLS.items():
                value = profile.skills.get(key)
                if value:
                    delta = (
                        ""
                        if value.get("previous") is None
                        else f" ({value['score'] - value['previous']:+d})"
                    )
                    lines.append(f"{label}: {value['score']}/100{delta} · {value['task']}")
                else:
                    lines.append(f"{label}: пока недостаточно данных")
            if message == "/weekly":
                recent = list(
                    (
                        await db.scalars(
                            select(Lesson).where(
                                Lesson.user_id == profile.user_id,
                                Lesson.status == "completed",
                                Lesson.started >= now - timedelta(days=7),
                            )
                        )
                    ).all()
                )
                lines.append(f"Занятий за 7 дней: {len(recent)}")
                for item in recent[-5:]:
                    lines.append(item.result.get("summary", ""))
                    lines.extend(
                        "Освоено: " + s for s in item.result.get("mastered_situations", [])
                    )
                reviews = list(
                    (
                        await db.scalars(
                            select(Review)
                            .where(Review.user_id == profile.user_id)
                            .order_by(Review.streak)
                            .limit(5)
                        )
                    ).all()
                )
                lines.append(
                    "Фокус и повторяющиеся ошибки: " + "; ".join(x.phrase for x in reviews)
                )
                profile.last_report = now
            return "\n".join(lines)
        if message == "/pause":
            if lesson:
                lesson.result = {
                    **lesson.result,
                    "remaining_seconds": max(
                        0, int((aware(lesson.deadline) - now).total_seconds())
                    ),
                }
                lesson.status = "paused"
            return "Занятие приостановлено. /lesson — продолжить, время паузы не учитывается."
        is_start = message in {"/start", "/repeat", "/lesson 5", "/lesson 10", "/lesson 30"}
        if is_start and lesson:
            return "Занятие уже идёт. Ответьте на вопрос или используйте /pause."
        if is_start and lesson is None:
            paused = await db.scalar(
                select(Lesson)
                .where(Lesson.user_id == profile.user_id, Lesson.status == "paused")
                .order_by(Lesson.id.desc())
            )
            if paused and message.startswith("/lesson"):
                paused.status = "active"
                paused.deadline = now + timedelta(
                    seconds=paused.result.get("remaining_seconds", paused.minutes * 60)
                )
                return "Продолжаем. " + next(
                    (x["content"] for x in reversed(paused.state) if x["role"] == "assistant"),
                    "Tell me more.",
                )
            due = list(
                (
                    await db.scalars(
                        select(Review)
                        .where(Review.user_id == profile.user_id, Review.due <= now)
                        .limit(5)
                    )
                ).all()
            )
            if message == "/repeat" and not due:
                return "Сегодня нет запланированных повторений. /lesson — новое занятие."
            mode = (
                "diagnostic"
                if not profile.diagnosed
                else "review"
                if message == "/repeat"
                else "lesson"
            )
            minutes = (
                int(message.split()[1]) if " " in message else 8 if mode == "diagnostic" else 5
            )
            weak = min(
                profile.skills,
                key=lambda x: profile.skills[x]["score"],
                default="speaking_independence",
            )
            topic = choose_scenario(
                profile.last_topic, weak, [{"phrase": x.phrase, "prompt": x.prompt} for x in due]
            )
            lesson = Lesson(
                user_id=profile.user_id,
                mode=mode,
                minutes=minutes,
                topic=topic,
                started=now,
                deadline=now + timedelta(minutes=minutes),
                state=[],
                result={},
            )
            db.add(lesson)
            await db.flush()
            message = "Begin the session."
        if not lesson:
            return "Начните с /start или /lesson."
        if message.startswith("/") and message not in {"/finish", "/timeout"}:
            return "Команда не распознана. /lesson · /progress · /repeat · /pause"
        force = now >= aware(lesson.deadline) or message in {"/finish", "/timeout"}
        completed = await db.scalar(
            select(func.count())
            .select_from(Lesson)
            .where(Lesson.user_id == profile.user_id, Lesson.status == "completed")
        )
        mistakes = list(
            (
                await db.scalars(
                    select(Review)
                    .where(Review.user_id == profile.user_id)
                    .order_by(Review.streak, Review.due)
                    .limit(3)
                )
            ).all()
        )
        context = {
            "past_errors_and_expressions": [
                {"phrase": item.phrase, "prompt": item.prompt, "kind": item.kind}
                for item in mistakes
            ],
            "mode": lesson.mode,
            "minutes": lesson.minutes,
            "topic": lesson.topic[:900],
            "skills": profile.skills,
            "force_finish": force,
            "input_mode": input_mode,
            "listening_available": lesson.result.get("listening_delivered", False),
            "listening_script": LISTENING_SCRIPT,
            "control_lesson": (completed + 1) % 7 == 0,
            "seconds_remaining": max(0, int((aware(lesson.deadline) - now).total_seconds())),
        }
        # Bound payload and memory growth; preserve first instructions and recent exchanges.
        history = lesson.state[:2] + lesson.state[-18:] if len(lesson.state) > 20 else lesson.state
        turn = await self.agent.respond(context, history, message)
        if force and not turn.finished:
            raise ValueError("Агент не завершил занятие; попробуйте /finish ещё раз.")
        lesson.state = history + [
            {"role": "user", "content": message, "mode": input_mode},
            {"role": "assistant", "content": turn.reply},
        ]
        if turn.finished:
            await save_result(db, profile, lesson, turn, now)
        return turn.reply

    async def listening_delivered(self, user_id):
        async with self.sessions.begin() as db:
            lesson = await self.active(db, user_id)
            if lesson:
                lesson.result = {**lesson.result, "listening_delivered": True}
