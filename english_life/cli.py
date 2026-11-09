import argparse
import asyncio
import contextlib
import fcntl
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

from sqlalchemy.engine import make_url

from english_life.agents.runner import DemoAgent, create_backend
from english_life.config import Settings
from english_life.learning.service import BudgetExceeded, LearningService, aware
from english_life.memory.db import create_local_schema, database
from english_life.memory.models import Profile, utcnow

HELP = """English Life · локальная практика английского
/start — диагностика · /lesson 5|10|30 — занятие · /resume — продолжить
/progress — навыки · /repeat — повторение · /weekly — отчёт
/pause — пауза · /finish — завершить · /delete_me — удалить данные
/voice | /text — озвучка ответов · /record 10 — записать речь (необязательный модуль)
/listen — локальное аудиозадание диагностики · quit — выход
"""


@contextlib.contextmanager
def single_instance(url):
    filename = make_url(url).database
    if not url.startswith("sqlite") or not filename or filename == ":memory:":
        raise ValueError("Локальный CLI требует файловую SQLite БД.")
    path = Path(filename).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Lock inode is stable: never remove a lock file while another process can open it.
    with open(str(path) + ".lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Другой English Life уже использует эту БД.") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


async def read_line(prompt):
    """Cancellable macOS/Linux terminal input without accumulating blocked input threads."""
    print(prompt, end="", flush=True)
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def ready():
        if not future.done():
            future.set_result(sys.stdin.readline())

    loop.add_reader(sys.stdin.fileno(), ready)
    try:
        line = await future
        if not line:
            raise EOFError
        return line.strip()
    finally:
        loop.remove_reader(sys.stdin.fileno())


async def chat(settings, demo=False, smoke=False):
    engine, sessions = database(settings.database_url)
    await create_local_schema(engine)
    filename = make_url(settings.database_url).database
    if filename and filename != ":memory:":
        os.chmod(filename, 0o600)
    backend = DemoAgent() if demo else create_backend(settings)
    service = LearningService(sessions, backend, settings)
    sequence = time.time_ns() // 1000
    from english_life.voice.local import LocalSpeech

    speech = LocalSpeech(settings)
    spoken = False
    try:
        if not demo and hasattr(backend, "check"):
            info = await backend.check()
            print(f"Ollama: {info['model']} ({info['quantization']})")
        if smoke:
            for index, text in enumerate(["/start", "A coffee, please", "/finish", "/progress"]):
                print(await service.handle(1, sequence + index, text))
            return
        print(HELP)
        if demo:
            print("Демо: детерминированный ответ, без LLM и оценки навыков.")
        async with sessions() as db:
            lesson = await service.active(db, 1)
            profile = await db.get(Profile, 1)
            if lesson:
                print(
                    "Найдено незавершённое занятие. /resume — последний вопрос; /finish — закончить."
                )
            report_due = (
                profile
                and profile.diagnosed
                and (
                    not profile.last_report
                    or utcnow() - aware(profile.last_report) >= timedelta(days=7)
                )
            )
        if report_due:
            sequence += 1
            print(await service.handle(1, sequence, "/weekly"))
        while True:
            async with sessions() as db:
                lesson = await service.active(db, 1)
                wait = (
                    max(0, (aware(lesson.deadline) - utcnow()).total_seconds()) if lesson else None
                )
            mode = "text"
            try:
                try:
                    text = await asyncio.wait_for(read_line("You > "), timeout=wait)
                except TimeoutError:
                    print("\nВремя занятия истекло. Подвожу итоги…")
                    text = "/timeout"
                if text in {"quit", "exit"}:
                    if lesson:
                        sequence += 1
                        print(await service.handle(1, sequence, "/pause"))
                    break
                if not text:
                    continue
                if text == "/help":
                    print(HELP)
                    continue
                if text == "/delete_me":
                    if await read_line("Удалить всю учебную историю? Введите DELETE: ") != "DELETE":
                        print("Отменено.")
                        continue
                if text == "/resume":
                    if lesson:
                        print(
                            next(
                                (
                                    x["content"]
                                    for x in reversed(lesson.state)
                                    if x["role"] == "assistant"
                                ),
                                "Начните с /start.",
                            )
                        )
                        continue
                    text = "/lesson 5"  # Service resumes paused session with its remaining time.
                if text in {"/voice", "/text"}:
                    spoken = text == "/voice"
                    if spoken:
                        speech.check_tts()
                    print("Озвучка включена." if spoken else "Текстовый режим.")
                    continue
                if text == "/listen":
                    if not lesson or lesson.mode != "diagnostic":
                        print("Сначала начните диагностику /start.")
                        continue
                    from english_life.learning.service import LISTENING_SCRIPT

                    await speech.speak(LISTENING_SCRIPT)
                    await service.listening_delivered(1)
                    print("What time does the train leave, and which platform?")
                    continue
                if text.startswith("/record"):
                    parts = text.split()
                    seconds = int(parts[1]) if len(parts) == 2 else 10
                    text = await speech.record_and_transcribe(seconds)
                    print("Распознано:", text)
                    mode = "voice"
                sequence += 1
                reply = await service.handle(1, sequence, text, mode)
                print("Tutor >", reply)
                if spoken:
                    await speech.speak(reply)
            except (EOFError, KeyboardInterrupt):
                break
            except Exception as error:
                # Local errors are actionable but provider response bodies are never dumped.
                if isinstance(error, (ValueError, RuntimeError, BudgetExceeded)):
                    print(str(error))
                else:
                    print(f"Ошибка {type(error).__name__}. Прогресс сохранён; повторите запрос.")
                if text == "/timeout":
                    print("Автозавершение отложено. /finish — повторить, /pause — пауза.")
                    # Avoid a zero-timeout busy loop if inference or validation failed.
                    sequence += 1
                    await service.handle(1, sequence, "/pause")
    finally:
        if hasattr(backend, "close"):
            await backend.close()
        await engine.dispose()


def main():
    parser = argparse.ArgumentParser(description="Local English Life tutor")
    parser.add_argument("mode", nargs="?", default="chat", choices=["chat", "demo", "doctor"])
    parser.add_argument("--smoke", action="store_true", help="Offline demo workflow check")
    args = parser.parse_args()
    settings = Settings()
    if args.smoke and args.mode != "demo":
        parser.error("--smoke is supported only in demo mode")
    try:
        if args.mode == "doctor":

            async def doctor():
                backend = create_backend(settings)
                try:
                    print(await backend.check())
                finally:
                    await backend.close()

            asyncio.run(doctor())
        else:
            with single_instance(settings.database_url):
                asyncio.run(chat(settings, demo=args.mode == "demo", smoke=args.smoke))
    except KeyboardInterrupt:
        print("\nПрогресс сохранён. До встречи.")
    except (ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
