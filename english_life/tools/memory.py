"""Validated, user-scoped persistence operations. The model never gets SQL or user IDs as tools."""

from sqlalchemy import select

from english_life.learning.planner import next_review
from english_life.learning.schemas import Turn
from english_life.memory.models import Review, Snapshot


async def save_result(db, profile, lesson, turn: Turn, now):
    turn = Turn.model_validate(turn)
    skills = dict(profile.skills)
    accepted = []
    modes = {entry.get("mode", "text") for entry in lesson.state if entry["role"] == "user"}
    for evidence in turn.evidence:
        if evidence.skill in {"speaking_independence", "grammar"} and "voice" not in modes:
            continue  # Text-only production cannot establish grammar IN SPEECH.
        if evidence.skill == "listening" and not lesson.result.get("listening_delivered"):
            continue
        accepted.append(evidence.model_dump())
        previous = skills.get(evidence.skill, {})
        skills[evidence.skill] = {
            "score": evidence.score,
            "previous": previous.get("score"),
            "task": evidence.task,
            "hints": evidence.hints,
            "demonstrated": evidence.demonstrated,
            "date": now.isoformat(),
        }
    profile.skills = skills
    profile.diagnosed = profile.diagnosed or lesson.mode == "diagnostic"
    profile.last_topic = lesson.topic
    if profile.last_report is None:
        profile.last_report = now
    for item in turn.reviews:
        review = await db.scalar(
            select(Review).where(Review.user_id == profile.user_id, Review.phrase == item.phrase)
        )
        existing = review is not None
        if review is None:
            review = Review(
                user_id=profile.user_id,
                phrase=item.phrase,
                prompt=item.prompt,
                kind=item.kind,
                streak=0,
                due=now,
            )
            db.add(review)
        review.streak, review.due = next_review(now, review.streak, item.recalled and existing)
    lesson.result = {
        "summary": turn.summary,
        "evidence": accepted,
        "mastered_situations": turn.mastered_situations,
    }
    lesson.status = "completed"
    lesson.state = []
    db.add(Snapshot(user_id=profile.user_id, skills=skills))
    await db.flush()  # A reply claiming completion is sent only after the caller commits.
