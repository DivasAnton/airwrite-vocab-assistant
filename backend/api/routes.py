from __future__ import annotations

import re
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.storage.database import Database, utc_now
from backend.domain.airwrite_runtime import AirWriteRuntime
from app.utils.config import settings


def normalize(value: str) -> str:
    return re.sub(r"[^a-z]", "", value.casefold())


class Candidate(BaseModel):
    label: str = Field(min_length=1, max_length=1)
    confidence: float = Field(ge=0, le=1)


class Segment(BaseModel):
    prediction: str = Field(min_length=1, max_length=1)
    confidence: float = Field(ge=0, le=1)
    margin: float = Field(ge=0, le=1)
    candidates: list[Candidate] = Field(default_factory=list, max_length=3)


class Draft(BaseModel):
    draft: str = Field(min_length=1, max_length=64)


class CanvasInput(BaseModel):
    image: str = Field(min_length=32)


class LearningInput(BaseModel):
    mode: str = Field(pattern="^(en_to_vi|vi_to_en|airwrite|cloze)$")
    collection_id: int | None = None


class Answer(BaseModel):
    answer: str = Field(max_length=128)


def create_router(database: Database, airwrite: AirWriteRuntime | None = None) -> APIRouter:
    router = APIRouter()
    recognition: dict[str, dict] = {}

    @router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "mode": "local", "input": "isolated_whole_word", "airwrite": "ready" if airwrite else "unavailable"}

    @router.get("/camera/settings")
    def camera_settings() -> dict:
        return {
            "width": settings.camera_width,
            "height": settings.camera_height,
            "fps": settings.camera_fps,
            "mirror": settings.camera_mirror,
            "camera_index": settings.camera_index,
            "hands": settings.hand_num_hands,
            "detection_confidence": settings.hand_min_detection_confidence,
            "presence_confidence": settings.hand_min_presence_confidence,
            "tracking_confidence": settings.hand_min_tracking_confidence,
        }

    @router.post("/recognition/sessions")
    def start_recognition() -> dict:
        session = {"session_id": uuid4().hex, "mode": "isolated_whole_word", "segments": [], "draft": "", "status": "reviewing"}
        recognition[session["session_id"]] = session
        return session

    @router.get("/recognition/sessions/{session_id}")
    def get_recognition(session_id: str) -> dict:
        session = recognition.get(session_id)
        if session is None:
            raise HTTPException(404, "Recognition session not found")
        return session

    @router.post("/recognition/sessions/{session_id}/segments")
    def add_segment(session_id: str, payload: Segment) -> dict:
        session = recognition.get(session_id)
        if session is None:
            raise HTTPException(404, "Recognition session not found")
        status = "accepted" if payload.confidence >= 0.60 and payload.margin >= 0.15 else "uncertain"
        item = {"position": len(session["segments"]), **payload.model_dump(), "status": status}
        session["segments"].append(item)
        session["draft"] = "".join(segment["prediction"] for segment in session["segments"])
        return item

    @router.post("/recognition/sessions/{session_id}/canvas")
    def recognize_canvas(session_id: str, payload: CanvasInput) -> dict:
        session = recognition.get(session_id)
        if session is None:
            raise HTTPException(404, "Recognition session not found")
        if airwrite is None:
            raise HTTPException(503, "AirWrite model is unavailable")
        try:
            result = airwrite.recognize_canvas(payload.image)
        except (ValueError, RuntimeError) as error:
            raise HTTPException(422, str(error)) from error
        session["segments"] = [
            {
                "position": character.position,
                "prediction": character.rendered_character,
                "confidence": character.confidence,
                "margin": character.candidates[0].confidence - character.candidates[1].confidence if len(character.candidates) > 1 else character.confidence,
                "status": character.status.value.lower(),
                "candidates": [{"label": candidate.rendered_character, "confidence": candidate.confidence} for candidate in character.candidates],
            }
            for character in result.characters
        ]
        session["draft"] = result.predicted_word
        session["recognition_status"] = result.status.value
        session["message"] = result.message
        return session

    @router.post("/recognition/sessions/{session_id}/frames")
    def camera_frame(session_id: str, payload: CanvasInput) -> dict:
        if session_id not in recognition:
            raise HTTPException(404, "Recognition session not found")
        if airwrite is None:
            raise HTTPException(503, "AirWrite model is unavailable")
        try:
            return airwrite.process_camera_frame(session_id, payload.image)
        except (ValueError, RuntimeError) as error:
            raise HTTPException(422, str(error)) from error

    @router.post("/recognition/sessions/{session_id}/finish")
    def finish_camera(session_id: str) -> dict:
        session = recognition.get(session_id)
        if session is None:
            raise HTTPException(404, "Recognition session not found")
        if airwrite is None:
            raise HTTPException(503, "AirWrite model is unavailable")
        try:
            result = airwrite.finish_camera_capture(session_id)
        except (ValueError, RuntimeError) as error:
            raise HTTPException(422, str(error)) from error
        session["draft"] = result.predicted_word
        session["recognition_status"] = result.status.value
        session["message"] = result.message
        session["segments"] = [{"position": c.position, "prediction": c.rendered_character, "confidence": c.confidence, "status": c.status.value.lower(), "candidates": [{"label": x.rendered_character, "confidence": x.confidence} for x in c.candidates]} for c in result.characters]
        return session

    @router.post("/recognition/sessions/{session_id}/clear")
    def clear_to_pause(session_id: str) -> dict[str, str]:
        if session_id not in recognition:
            raise HTTPException(404, "Recognition session not found")
        if airwrite is None:
            raise HTTPException(503, "AirWrite model is unavailable")
        try:
            return airwrite.clear_to_last_pause(session_id)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @router.patch("/recognition/sessions/{session_id}")
    def update_draft(session_id: str, payload: Draft) -> dict:
        session = recognition.get(session_id)
        if session is None:
            raise HTTPException(404, "Recognition session not found")
        session["draft"] = payload.draft
        return session

    @router.post("/recognition/sessions/{session_id}/commit")
    def commit(session_id: str) -> dict:
        session = recognition.get(session_id)
        if session is None:
            raise HTTPException(404, "Recognition session not found")
        if any(segment["status"] == "uncertain" for segment in session["segments"]):
            raise HTTPException(409, "Resolve uncertain segments before committing")
        word = normalize(session["draft"])
        if not word:
            raise HTTPException(422, "The draft is empty")
        session.update(status="committed", word=word)
        return {"session_id": session_id, "word": word, "entry": database.vocabulary(word)}

    @router.get("/dictionary/{word}")
    def dictionary(word: str) -> dict:
        entry = database.vocabulary(word)
        if entry is None:
            raise HTTPException(404, "Dictionary entry not found")
        return entry

    @router.post("/vocabulary/{word}/save")
    def save_word(word: str) -> dict:
        entry = database.vocabulary(word)
        if entry is None:
            raise HTTPException(404, "Dictionary entry not found")
        database.save(entry["id"])
        return {"saved": True, "word": entry["normalized_word"]}

    @router.get("/vocabulary")
    def saved_words() -> list[dict]:
        rows = database.connection.execute("SELECT v.* FROM vocabulary v JOIN saved_vocabulary s ON s.vocabulary_id=v.id ORDER BY s.saved_at DESC").fetchall()
        return [database.vocabulary(row["normalized_word"]) for row in rows]

    @router.get("/collections")
    def collections(category: str | None = None) -> list[dict]:
        query = "SELECT * FROM collections"
        params: tuple = ()
        if category:
            query += " WHERE category=?"
            params = (category,)
        return [dict(row) for row in database.connection.execute(query + " ORDER BY category, name", params).fetchall()]

    @router.get("/collections/{collection_id}")
    def collection(collection_id: int) -> dict:
        row = database.connection.execute("SELECT * FROM collections WHERE id=?", (collection_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Collection not found")
        words = database.connection.execute("SELECT v.normalized_word FROM vocabulary v JOIN collection_items i ON i.vocabulary_id=v.id WHERE i.collection_id=? ORDER BY i.position", (collection_id,)).fetchall()
        return {**dict(row), "vocabulary": [database.vocabulary(item["normalized_word"]) for item in words]}

    @router.post("/learning/sessions")
    def create_learning(payload: LearningInput) -> dict:
        session_id = uuid4().hex
        database.connection.execute("INSERT INTO learning_sessions (id, collection_id, mode, created_at) VALUES (?, ?, ?, ?)", (session_id, payload.collection_id, payload.mode, utc_now()))
        database.connection.commit()
        return learning_state(database, session_id)

    @router.get("/learning/sessions/{session_id}")
    def get_learning(session_id: str) -> dict:
        return learning_state(database, session_id)

    @router.post("/learning/sessions/{session_id}/answer")
    def answer_learning(session_id: str, payload: Answer) -> dict:
        state = learning_state(database, session_id)
        question = state["question"]
        expected = question["expected"]
        correct = normalize(payload.answer) == normalize(expected) if state["mode"] != "en_to_vi" else payload.answer.strip().casefold() == expected.strip().casefold()
        database.connection.execute("INSERT INTO learning_attempts (session_id, vocabulary_id, expected, answer, correct, created_at) VALUES (?, ?, ?, ?, ?, ?)", (session_id, question["vocabulary_id"], expected, payload.answer, int(correct), utc_now()))
        database.connection.execute("INSERT OR IGNORE INTO progress (vocabulary_id, next_review) VALUES (?, ?)", (question["vocabulary_id"], utc_now()))
        database.connection.execute("UPDATE progress SET correct_count=correct_count+?, incorrect_count=incorrect_count+?, mastery=MIN(1.0, (correct_count+?)*1.0/(correct_count+incorrect_count+1)), last_reviewed=?, state=? WHERE vocabulary_id=?", (int(correct), int(not correct), int(correct), utc_now(), "mastered" if correct else "learning", question["vocabulary_id"]))
        database.connection.execute("UPDATE learning_sessions SET question_index=question_index+1, score=score+? WHERE id=?", (int(correct), session_id))
        database.connection.commit()
        return {"correct": correct, "expected": expected, "answer": payload.answer, "session": learning_state(database, session_id)}

    @router.get("/progress")
    def progress() -> list[dict]:
        return [dict(row) for row in database.connection.execute("SELECT p.*, v.word FROM progress p JOIN vocabulary v ON v.id=p.vocabulary_id ORDER BY p.mastery DESC, v.word").fetchall()]

    return router


def learning_state(database: Database, session_id: str) -> dict:
    row = database.connection.execute("SELECT * FROM learning_sessions WHERE id=?", (session_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Learning session not found")
    if row["collection_id"]:
        question = database.connection.execute("SELECT v.normalized_word FROM vocabulary v JOIN collection_items i ON i.vocabulary_id=v.id WHERE i.collection_id=? ORDER BY i.position LIMIT 1 OFFSET ?", (row["collection_id"], row["question_index"])).fetchone()
    else:
        question = database.connection.execute("SELECT v.normalized_word FROM vocabulary v JOIN saved_vocabulary s ON s.vocabulary_id=v.id ORDER BY s.saved_at LIMIT 1 OFFSET ?", (row["question_index"],)).fetchone()
    item = database.vocabulary(question["normalized_word"]) if question else None
    if item:
        item["vocabulary_id"] = item["id"]
        item["expected"] = item["word"] if row["mode"] != "en_to_vi" else item["vietnamese_meaning"]
    return {**dict(row), "question": item}
