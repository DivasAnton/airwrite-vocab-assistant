from __future__ import annotations

from pydantic import BaseModel, Field


class Candidate(BaseModel):
    label: str = Field(min_length=1, max_length=1)
    confidence: float = Field(ge=0, le=1)


class SegmentInput(BaseModel):
    prediction: str = Field(min_length=1, max_length=1)
    confidence: float = Field(ge=0, le=1)
    margin: float = Field(ge=0, le=1)
    candidates: list[Candidate] = Field(default_factory=list, max_length=3)


class DraftUpdate(BaseModel):
    draft: str = Field(min_length=0, max_length=128)


class AnswerInput(BaseModel):
    answer: str = Field(max_length=128)


class LearningSessionInput(BaseModel):
    collection_id: int | None = None
    mode: str = Field(default="en_to_vi", pattern="^(en_to_vi|vi_to_en|airwrite|cloze)$")

