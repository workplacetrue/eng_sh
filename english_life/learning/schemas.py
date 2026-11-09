from typing import Literal

from pydantic import BaseModel, Field

SkillName = Literal[
    "reading", "listening", "speaking_independence", "grammar", "vocabulary", "communication"
]


class Evidence(BaseModel):
    skill: SkillName
    task: str = Field(min_length=1, max_length=300)
    demonstrated: bool
    hints: int = Field(ge=0, le=10)
    # Rubric evidence, never an official CEFR certification.
    score: int = Field(ge=0, le=100)


class ReviewItem(BaseModel):
    phrase: str = Field(min_length=1, max_length=160)
    prompt: str = Field(min_length=1, max_length=300)
    kind: Literal["expression", "grammar", "communication"]
    recalled: bool = False


class Turn(BaseModel):
    reply: str = Field(min_length=1, max_length=3500)
    finished: bool = False
    summary: str = Field(default="", max_length=1500)
    evidence: list[Evidence] = Field(default_factory=list, max_length=6)
    reviews: list[ReviewItem] = Field(default_factory=list, max_length=5)
    mastered_situations: list[str] = Field(default_factory=list, max_length=5)
