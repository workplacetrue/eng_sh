import random
from datetime import datetime, timedelta

SCENARIOS = [
    "Airport: your flight changed and you need another route",
    "Taxi: negotiate a price and explain where to stop",
    "Cafe: order lunch, then resolve a wrong order",
    "Renting: ask a landlord about costs and negotiate a repair",
    "New friend: discover shared interests and suggest plans",
    "Opinions: explain a preference to someone who disagrees",
    "Work: introduce your experience and negotiate a deadline",
]


def choose_scenario(last_topic: str | None, weak_skill: str, due: list[dict]) -> str:
    topic = random.choice([x for x in SCENARIOS if not last_topic or not last_topic.startswith(x)])
    return f"{topic}. Focus: {weak_skill}. Due practice: {due[:5]}"


def next_review(now: datetime, streak: int, recalled: bool) -> tuple[int, datetime]:
    streak = min(streak + 1, 6) if recalled else 0
    return streak, now + timedelta(days=[1, 2, 4, 7, 14, 30, 60][streak])
