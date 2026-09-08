from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query

from app.web.db import Database, utc_now
from app.web.schemas import AnswerInput, DraftUpdate, LearningSessionInput, SegmentInput


def normalize_word(value: str) -> str:
    return re.sub(r"[^a-z]", "", value.casefold())


def create_router(database: Database) -> APIRouter:
    router = APIRouter()
    recognition: dict[str, dict] = {}

    @router.post("/recognition/sessions")
    def start_recognition(mode: str = Query("word", pattern="^(character|word)$")):
        session_id = uuid4().hex
        recognition[session_id] = {"session_id": session_id, "mode": mode, "segments": [], "draft": "", "status": "reviewing"}
        return recognition[session_id]

    @router.get("/recognition/sessions/{session_id}")
    def get_recognition(session_id: str):
        if session_id not in recognition:
            raise HTTPException(404, "Recognition session not found")
        return recognition[session_id]

    @router.post("/recognition/sessions/{session_id}/segments")
    def add_segment(session_id: str, payload: SegmentInput):
        session = recognition.get(session_id)
        if session is None:
            raise HTTPException(404, "Recognition session not found")
        status = "accepted" if payload.confidence >= 0.60 and payload.margin >= 0.15 else "uncertain"
        candidates = [candidate.model_dump() for candidate in payload.candidates[:3]]
        if not candidates:
            candidates = [{"label": payload.prediction, "confidence": payload.confidence}]
        segment = {"position": len(session["segments"]), "prediction": payload.prediction, "confidence": payload.confidence, "margin": payload.margin, "status": status, "candidates": candidates}
        session["segments"].append(segment)
        session["draft"] = "".join(item["prediction"] for item in session["segments"])
        return segment

    @router.patch("/recognition/sessions/{session_id}")
    def update_recognition(session_id: str, payload: DraftUpdate):
        session = recognition.get(session_id)
        if session is None:
            raise HTTPException(404, "Recognition session not found")
        session["draft"] = payload.draft
        return session

    @router.post("/recognition/sessions/{session_id}/commit")
    def commit_recognition(session_id: str):
        session = recognition.get(session_id)
        if session is None:
            raise HTTPException(404, "Recognition session not found")
        if any(item["status"] == "uncertain" for item in session["segments"]):
            raise HTTPException(409, "Resolve uncertain segments before committing")
        word = normalize_word(session["draft"])
        if not word:
            raise HTTPException(422, "The draft is empty")
        session["status"] = "committed"
        session["word"] = word
        return {"session_id": session_id, "word": word, "dictionary": lookup_word(database, word)}

    @router.get("/dictionary/{word}")
    def dictionary(word: str):
        result = lookup_word(database, word)
        if result is None:
            raise HTTPException(404, "Dictionary entry not found")
        return result

    @router.post("/vocabulary/{word}/save")
    def save_vocabulary(word: str):
        result = lookup_word(database, word)
        if result is None:
            raise HTTPException(404, "Dictionary entry not found")
        database.save_word(result["id"])
        return {"saved": True, "word": result["normalized_word"]}

    @router.get("/vocabulary")
    def my_vocabulary():
        rows = database.connection.execute("SELECT v.* FROM vocabulary v JOIN user_vocabulary u ON u.vocabulary_id=v.id ORDER BY u.saved_at DESC").fetchall()
        return [database.vocabulary_json(row) for row in rows]

    @router.get("/collections")
    def collections(category: str | None = None):
        if category:
            rows = database.connection.execute("SELECT * FROM collections WHERE category=? ORDER BY name", (category,)).fetchall()
        else:
            rows = database.connection.execute("SELECT * FROM collections ORDER BY category, name").fetchall()
        return [dict(row) for row in rows]

    @router.get("/collections/{collection_id}")
    def collection(collection_id: int):
        row = database.connection.execute("SELECT * FROM collections WHERE id=?", (collection_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Collection not found")
        items = database.connection.execute("SELECT v.* FROM vocabulary v JOIN collection_items ci ON ci.vocabulary_id=v.id WHERE ci.collection_id=? ORDER BY ci.position", (collection_id,)).fetchall()
        return {**dict(row), "vocabulary": [database.vocabulary_json(item) for item in items]}

    @router.post("/learning/sessions")
    def create_learning_session(payload: LearningSessionInput):
        session_id = uuid4().hex
        total = database.connection.execute("SELECT COUNT(*) FROM collection_items WHERE collection_id=?", (payload.collection_id,)).fetchone()[0] if payload.collection_id else database.connection.execute("SELECT COUNT(*) FROM user_vocabulary").fetchone()[0]
        database.connection.execute("INSERT INTO learning_sessions (id, collection_id, mode, total, created_at) VALUES (?, ?, ?, ?, ?)", (session_id, payload.collection_id, payload.mode, total, utc_now()))
        database.connection.commit()
        return learning_state(database, session_id)

    @router.get("/learning/sessions/{session_id}")
    def get_learning_session(session_id: str):
        return learning_state(database, session_id)

    @router.post("/learning/sessions/{session_id}/answer")
    def answer_learning_session(session_id: str, payload: AnswerInput):
        state = learning_state(database, session_id)
        question = state["question"]
        expected = question["expected"]
        correct = normalize_word(payload.answer) == normalize_word(expected) if state["mode"] in {"vi_to_en", "airwrite", "cloze"} else payload.answer.strip().casefold() == expected.strip().casefold()
        database.connection.execute("INSERT INTO learning_attempts (session_id, vocabulary_id, expected, answer, correct, created_at) VALUES (?, ?, ?, ?, ?, ?)", (session_id, question["vocabulary_id"], expected, payload.answer, int(correct), utc_now()))
        database.connection.execute("INSERT OR IGNORE INTO progress (vocabulary_id, next_review) VALUES (?, ?)", (question["vocabulary_id"], utc_now()))
        database.connection.execute("UPDATE progress SET correct_count=correct_count+?, incorrect_count=incorrect_count+?, mastery=MIN(1.0, (correct_count+?)*1.0/(correct_count+incorrect_count+1)), last_reviewed=?, next_review=?, state=? WHERE vocabulary_id=?", (int(correct), int(not correct), int(correct), utc_now(), utc_now(), "mastered" if correct else "learning", question["vocabulary_id"]))
        database.connection.execute("UPDATE learning_sessions SET question_index=question_index+1, score=score+? WHERE id=?", (int(correct), session_id))
        database.connection.commit()
        return {"correct": correct, "expected": expected, "answer": payload.answer, "session": learning_state(database, session_id)}

    @router.get("/progress")
    def progress():
        rows = database.connection.execute("SELECT p.*, v.word FROM progress p JOIN vocabulary v ON v.id=p.vocabulary_id ORDER BY p.mastery DESC, v.word").fetchall()
        return [dict(row) for row in rows]

    @router.get("/health")
    def health():
        return {"status": "ok", "airwrite_pipeline": "reused", "database": str(Path(database.path))}

    return router


def lookup_word(database: Database, word: str) -> dict | None:
    row = database.vocabulary_row(word)
    return database.vocabulary_json(row) if row else None


def learning_state(database: Database, session_id: str) -> dict:
    row = database.connection.execute("SELECT * FROM learning_sessions WHERE id=?", (session_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Learning session not found")
    data = dict(row)
    if row["collection_id"]:
        question = database.connection.execute("SELECT v.id, v.word, v.vietnamese_meaning, ci.position FROM vocabulary v JOIN collection_items ci ON ci.vocabulary_id=v.id WHERE ci.collection_id=? ORDER BY ci.position LIMIT 1 OFFSET ?", (row["collection_id"], row["question_index"])).fetchone()
    else:
        question = database.connection.execute("SELECT v.id, v.word, v.vietnamese_meaning, 0 AS position FROM vocabulary v JOIN user_vocabulary u ON u.vocabulary_id=v.id ORDER BY v.word LIMIT 1 OFFSET ?", (row["question_index"],)).fetchone()
    if question:
        q = dict(question)
        q["vocabulary_id"] = q.pop("id")
        q["expected"] = q["word"] if row["mode"] in {"vi_to_en", "airwrite", "cloze"} else q["vietnamese_meaning"]
    else:
        q = None
    data["question"] = q
    return data

