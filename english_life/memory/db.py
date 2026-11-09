from pathlib import Path

from sqlalchemy import event
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from english_life.memory.models import Base


def database(url: str):
    if url.startswith("sqlite"):
        filename = make_url(url).database
        if filename and filename != ":memory:":
            parent = Path(filename).expanduser().parent
            parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    engine = create_async_engine(url, pool_pre_ping=True)
    if url.startswith("sqlite"):

        @event.listens_for(engine.sync_engine, "connect")
        def foreign_keys(connection, _):
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute("PRAGMA journal_mode=WAL")

    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def create_local_schema(engine):
    # Production schema is managed by Alembic, never implicitly by the bot.
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
