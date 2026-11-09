import asyncio
import logging
from collections import defaultdict
from datetime import timedelta
from io import BytesIO

from agents import set_default_openai_client
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import BufferedInputFile, KeyboardButton, Message, ReplyKeyboardMarkup
from openai import AsyncOpenAI
from sqlalchemy import select

from english_life.agents.openai_backend import TutorAgent
from english_life.learning.service import LISTENING_SCRIPT, BudgetExceeded, LearningService
from english_life.memory.db import database
from english_life.memory.models import Lesson, Profile, Receipt, utcnow
from english_life.telegram.audio import Audio

log = logging.getLogger(__name__)
KEYBOARD = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton(text="5 минут"),
            KeyboardButton(text="10 минут"),
            KeyboardButton(text="30 минут"),
        ],
        [
            KeyboardButton(text="/progress"),
            KeyboardButton(text="/repeat"),
            KeyboardButton(text="/pause"),
        ],
        [
            KeyboardButton(text="/voice"),
            KeyboardButton(text="/text"),
            KeyboardButton(text="/finish"),
        ],
    ],
    resize_keyboard=True,
)


async def run(settings):
    if not settings.telegram_bot_token or not settings.openai_api_key or not settings.allowed_ids:
        raise ValueError("Set TELEGRAM_BOT_TOKEN, OPENAI_API_KEY and ALLOWED_TELEGRAM_IDS")
    if not settings.database_url.startswith("postgresql+asyncpg://"):
        raise ValueError("Telegram production mode requires PostgreSQL; use demo for SQLite")
    llm_client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=45, max_retries=0)
    set_default_openai_client(llm_client)
    engine, sessions = database(settings.database_url)
    service = LearningService(sessions, TutorAgent(settings), settings)
    audio, bot, dispatcher = Audio(settings), Bot(settings.telegram_bot_token), Dispatcher()
    locks = defaultdict(asyncio.Lock)

    async def deliver(message: Message, text: str):
        for start in range(0, len(text), 3900):
            await message.answer(text[start : start + 3900], reply_markup=KEYBOARD)
        async with sessions() as db:
            profile = await db.get(Profile, message.from_user.id)
            wants_audio = profile and profile.audio_reply
        if wants_audio:
            try:
                await service.reserve_api(message.from_user.id)
                speech = await audio.speak(text)
                await message.answer_voice(BufferedInputFile(speech, filename="reply.ogg"))
            except Exception:
                await message.answer("Озвучка недоступна; текстовый ответ сохранён.")

    @dispatcher.message(Command("delete_me"))
    async def delete_me(message: Message):
        if message.chat.type != "private" or message.from_user.id not in settings.allowed_ids:
            return
        async with locks[message.from_user.id]:
            result = await service.handle(message.from_user.id, message.message_id, "/delete_me")
            await message.answer(result)

    @dispatcher.message(F.text | F.voice)
    async def handle(message: Message):
        if message.chat.type != "private" or message.from_user.id not in settings.allowed_ids:
            return
        user_id = message.from_user.id
        async with locks[user_id]:
            try:
                async with sessions() as db:
                    receipt = await db.get(Receipt, (user_id, message.message_id))
                if receipt:
                    for start in range(0, len(receipt.reply), 3900):
                        await message.answer(
                            receipt.reply[start : start + 3900], reply_markup=KEYBOARD
                        )
                    return
                mode = "text"
                if message.voice:
                    if (
                        message.voice.duration > settings.max_voice_seconds
                        or (message.voice.file_size or 0) > 5_000_000
                    ):
                        await message.answer(
                            f"Голосовое должно быть короче {settings.max_voice_seconds} секунд и 5 МБ."
                        )
                        return
                    await service.reserve_api(user_id)
                    stream = BytesIO()
                    await bot.download(message.voice, destination=stream)
                    text, mode = await audio.transcribe(stream.getvalue()), "voice"
                else:
                    text = message.text.strip()
                    if text in {"5 минут", "10 минут", "30 минут"}:
                        text = "/lesson " + text.split()[0]
                if text == "/listen":
                    async with sessions() as db:
                        lesson = await service.active(db, user_id)
                    if not lesson or lesson.mode != "diagnostic":
                        await message.answer("Аудиозадание доступно во время диагностики /start.")
                        return
                    await service.reserve_api(user_id)
                    speech = await audio.speak(LISTENING_SCRIPT)
                    await message.answer_voice(
                        BufferedInputFile(speech, filename="listening.ogg"),
                        caption="Listen: what time does the train leave, and which platform?",
                    )
                    await service.listening_delivered(user_id)
                    return
                reply = await service.handle(user_id, message.message_id, text, mode)
                await deliver(message, reply)
                if text == "/start":
                    await message.answer(
                        "Для проверки понимания на слух отправьте /listen во время диагностики. "
                        "Отвечать можно текстом или голосом. /weekly — отчёт за неделю."
                    )
            except BudgetExceeded as error:
                await message.answer(str(error))
            except Exception as error:
                # Do not log prompts, tokens, transcripts, or provider error bodies.
                log.warning("Request failed: %s", type(error).__name__)
                await message.answer(
                    "Не удалось обработать запрос. Состояние занятия сохранено. "
                    "Повторите ответ чуть позже или используйте /progress."
                )

    async def maintenance():
        while True:
            await asyncio.sleep(20)
            now = utcnow()
            async with sessions() as db:
                expired = list(
                    (
                        await db.scalars(
                            select(Lesson).where(Lesson.status == "active", Lesson.deadline <= now)
                        )
                    ).all()
                )
                weekly = list(
                    (await db.scalars(select(Profile).where(Profile.diagnosed.is_(True)))).all()
                )
            for lesson in expired:
                if lesson.user_id not in settings.allowed_ids:
                    continue
                async with locks[lesson.user_id]:
                    try:
                        reply = await service.handle(lesson.user_id, -lesson.id, "/timeout")
                        await bot.send_message(lesson.user_id, reply)
                    except Exception as error:
                        log.warning("Finalization deferred: %s", type(error).__name__)
                        # Avoid retry storms; no further tutor turns until closure succeeds.
                        await asyncio.sleep(1)
            for profile in weekly:
                if profile.user_id not in settings.allowed_ids:
                    continue
                from english_life.learning.service import aware

                if profile.last_report and now - aware(profile.last_report) < timedelta(days=7):
                    continue
                async with locks[profile.user_id]:
                    try:
                        text = await service.handle(
                            profile.user_id, -int(now.timestamp()), "/weekly"
                        )
                        await bot.send_message(profile.user_id, text[:4000])
                    except Exception as error:
                        log.warning("Weekly report deferred: %s", type(error).__name__)

    task = asyncio.create_task(maintenance())
    try:
        await dispatcher.start_polling(bot, allowed_updates=["message"], handle_as_tasks=False)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await audio.close()
        await llm_client.close()
        await bot.session.close()
        await engine.dispose()
