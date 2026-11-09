from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow():
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Profile(Base):
    __tablename__ = "profiles"
    user_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    skills: Mapped[dict] = mapped_column(JSON, default=dict)
    diagnosed: Mapped[bool] = mapped_column(default=False)
    audio_reply: Mapped[bool] = mapped_column(default=False)
    last_topic: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_report: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    budget_day: Mapped[str] = mapped_column(String(10), default="")
    api_calls: Mapped[int] = mapped_column(default=0)


class Lesson(Base):
    __tablename__ = "lessons"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("profiles.user_id", ondelete="CASCADE"), index=True
    )
    mode: Mapped[str] = mapped_column(String(20))
    minutes: Mapped[int]
    topic: Mapped[str] = mapped_column(Text)
    started: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    deadline: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(20), default="active")
    # Only unfinished lessons retain bounded conversation state. Cleared at completion.
    state: Mapped[list] = mapped_column(JSON, default=list)
    result: Mapped[dict] = mapped_column(JSON, default=dict)


class Review(Base):
    __tablename__ = "reviews"
    __table_args__ = (UniqueConstraint("user_id", "phrase"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("profiles.user_id", ondelete="CASCADE"), index=True
    )
    phrase: Mapped[str] = mapped_column(String(160))
    prompt: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String(30))
    streak: Mapped[int] = mapped_column(default=0)
    due: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Snapshot(Base):
    __tablename__ = "snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("profiles.user_id", ondelete="CASCADE"), index=True
    )
    created: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    skills: Mapped[dict] = mapped_column(JSON)


class Receipt(Base):
    __tablename__ = "receipts"
    user_id: Mapped[int] = mapped_column(
        ForeignKey("profiles.user_id", ondelete="CASCADE"), primary_key=True
    )
    update_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    reply: Mapped[str] = mapped_column(Text)
