import asyncio
from collections import defaultdict
from datetime import UTC, timedelta
from math import ceil

from sqlalchemy import delete, func, select

from english_life.agents.runner import InvalidModelOutput
from english_life.learning.planner import choose_scenario
from english_life.learning.schemas import Turn
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
LISTENING_QUESTION = "Во сколько отправляется поезд и с какой платформы? Ответьте по-английски."
BEGIN_SESSION = "Begin the session."
DURATION_COMMANDS = {f"/lesson {minutes}": minutes for minutes in (5, 10, 30)}
START_COMMANDS = {"/start", "/repeat", *DURATION_COMMANDS}
FINISH_COMMANDS = {"/finish", "/timeout"}


def aware(value):
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def has_answers(lesson):
    # The first stored exchange is the synthetic start request and the opening task.
    return any(turn["role"] == "user" for turn in lesson.state[2:])


def last_prompt(lesson):
    if lesson.result.get("listening_pending"):
        return LISTENING_QUESTION
    return next(
        (turn["content"] for turn in reversed(lesson.state) if turn["role"] == "assistant"),
        "Продолжите ответ по предыдущему заданию.",
    )


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

    async def pending(self, db, user_id):
        return await db.scalar(
            select(Lesson)
            .where(Lesson.user_id == user_id, Lesson.status.in_(("active", "paused")))
            .order_by((Lesson.status == "active").desc(), Lesson.id.desc())
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
            if not message.strip():
                return "Введите ответ, чтобы продолжить занятие."
            now = utcnow()
            async with self.sessions() as db:
                lesson = await self.pending(db, user_id)
                if lesson is None:
                    needs_call = message in START_COMMANDS
                    if message == "/repeat":
                        needs_call = (
                            await db.scalar(
                                select(Review.id)
                                .where(Review.user_id == user_id, Review.due <= now)
                                .limit(1)
                            )
                            is not None
                        )
                elif message == "/finish":
                    needs_call = has_answers(lesson)
                elif lesson.status == "active":
                    needs_call = not message.startswith("/") or (
                        message == "/timeout"
                        and now >= aware(lesson.deadline)
                        and has_answers(lesson)
                    )
                else:
                    needs_call = False
            if needs_call:
                await self.reserve_api(user_id)
            async with self.sessions.begin() as db:
                profile = await db.get(Profile, user_id, with_for_update=True)
                lesson = await self.pending(db, user_id)
                reply = await self._handle(db, profile, lesson, message, input_mode, now)
                db.add(Receipt(user_id=user_id, update_id=update_id, reply=reply))
                # Keep replay data bounded and separate from educational history.
                await db.execute(
                    delete(Receipt).where(
                        Receipt.user_id == user_id, Receipt.update_id < update_id - 10000
                    )
                )
                return reply

    async def _handle(self, db, profile, lesson, message, input_mode, now):
        if message in {"/voice", "/text"}:
            profile.audio_reply = message == "/voice"
            return (
                "Голосовые ответы включены."
                if profile.audio_reply
                else "Текстовые ответы включены."
            )
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
            if not lesson:
                return "Нет занятия, которое можно приостановить."
            if lesson.status == "active":
                lesson.result = {
                    **lesson.result,
                    "remaining_seconds": max(0, (aware(lesson.deadline) - now).total_seconds()),
                }
                lesson.status = "paused"
            if lesson.result.get("remaining_seconds", lesson.minutes * 60) <= 0:
                return "Время занятия истекло. Используйте /finish, чтобы подвести итоги."
            return "Занятие приостановлено. /resume — продолжить, /finish — подвести итоги."
        paused = lesson is not None and lesson.status == "paused"
        if message == "/resume" or (paused and message in {"/lesson", *DURATION_COMMANDS}):
            if not lesson:
                return "Нет незавершённого занятия. Используйте /start или /lesson."
            if paused:
                result = dict(lesson.result)
                remaining = result.pop("remaining_seconds", lesson.minutes * 60)
                if remaining <= 0:
                    return "Время занятия истекло. Используйте /finish, чтобы подвести итоги."
                lesson.status = "active"
                lesson.deadline = now + timedelta(seconds=remaining)
                lesson.result = result
                return "Продолжаем.\n" + last_prompt(lesson)
            if now >= aware(lesson.deadline):
                return "Время занятия истекло. Используйте /finish, чтобы подвести итоги."
            return last_prompt(lesson)
        if message == "/lesson":
            return "Выберите продолжительность: 5, 10 или 30 минут."
        if paused and message != "/finish":
            return "Занятие на паузе. /resume — продолжить, /finish — подвести итоги."
        opening = message in START_COMMANDS
        if opening:
            if lesson:
                return "Занятие уже идёт. Ответьте на вопрос или используйте /pause."
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
            minutes = DURATION_COMMANDS.get(message, 8 if mode == "diagnostic" else 5)
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
            message = BEGIN_SESSION
        if not lesson:
            if message in FINISH_COMMANDS:
                return "Нет незавершённого занятия."
            return "Начните с /start или /lesson."
        if message.startswith("/") and message not in FINISH_COMMANDS:
            return "Команда не распознана. /lesson · /progress · /repeat · /pause · /finish"
        remaining = (
            lesson.result.get("remaining_seconds", lesson.minutes * 60)
            if paused
            else max(0, (aware(lesson.deadline) - now).total_seconds())
        )
        if message == "/timeout" and remaining > 0:
            return "Занятие ещё идёт. Используйте /finish для завершения."
        force = remaining == 0 or message == "/finish"
        answering = not opening and message not in FINISH_COMMANDS
        if force and not (answering or has_answers(lesson)):
            reply = "Занятие завершено. Ответов для оценки пока нет."
            await save_result(
                db, profile, lesson, Turn(reply=reply, summary=reply, finished=True), now
            )
            return reply
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
            "seconds_remaining": ceil(remaining),
        }
        if answering and lesson.result.get("listening_pending"):
            context["listening_question"] = LISTENING_QUESTION
        # Bound payload and memory growth; preserve first instructions and recent exchanges.
        history = lesson.state[:2] + lesson.state[-18:] if len(lesson.state) > 20 else lesson.state
        keep_dialogue = False
        try:
            turn = await self.agent.respond(context, history, message)
        except InvalidModelOutput:
            if not force:
                raise
            reply = "Занятие завершено. Ответы сохранены. Модель не смогла подготовить разбор."
            turn = Turn(reply=reply, summary=reply, finished=True)
            keep_dialogue = True
        if force and not turn.finished:
            raise ValueError(
                "Модель не подвела итоги. Повторите /finish или отправьте ответ ещё раз."
            )
        if opening and turn.finished:
            raise ValueError("Модель завершила занятие до первого задания. Повторите команду.")
        if opening:
            lesson.started = utcnow()
            lesson.deadline = lesson.started + timedelta(minutes=lesson.minutes)
        learner_turn = {"role": "user", "content": message, "mode": input_mode}
        if answering and lesson.result.get("listening_pending"):
            result = dict(lesson.result)
            result.pop("listening_pending")
            result["listening_answered"] = True
            lesson.result = result
            learner_turn["in_reply_to"] = LISTENING_QUESTION
        if answering and input_mode == "voice":
            lesson.result = {**lesson.result, "had_voice_input": True}
        stored_history = lesson.state if keep_dialogue else history
        lesson.state = stored_history + [
            learner_turn,
            {"role": "assistant", "content": turn.reply},
        ]
        if turn.finished:
            await save_result(db, profile, lesson, turn, now, keep_dialogue=keep_dialogue)
        return turn.reply

    async def listening_delivered(self, user_id):
        async with self.sessions.begin() as db:
            lesson = await self.active(db, user_id)
            if lesson:
                lesson.result = {
                    **lesson.result,
                    "listening_delivered": True,
                    "listening_pending": True,
                }
