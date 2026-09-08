from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: str | Path = "data/vocabulary.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.initialize()

    def initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS vocabulary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT NOT NULL,
                normalized_word TEXT NOT NULL UNIQUE,
                language TEXT NOT NULL DEFAULT 'en',
                vietnamese_meaning TEXT NOT NULL,
                definition TEXT NOT NULL,
                part_of_speech TEXT,
                pronunciation TEXT,
                ipa TEXT,
                synonyms_json TEXT NOT NULL DEFAULT '[]',
                antonyms_json TEXT NOT NULL DEFAULT '[]',
                examples_json TEXT NOT NULL DEFAULT '[]',
                cefr TEXT,
                ielts_relevance INTEGER NOT NULL DEFAULT 0,
                toeic_relevance INTEGER NOT NULL DEFAULT 0,
                topics_json TEXT NOT NULL DEFAULT '[]',
                source TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS collections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                UNIQUE(category, name)
            );
            CREATE TABLE IF NOT EXISTS collection_items (
                collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
                vocabulary_id INTEGER NOT NULL REFERENCES vocabulary(id) ON DELETE CASCADE,
                position INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(collection_id, vocabulary_id)
            );
            CREATE TABLE IF NOT EXISTS user_vocabulary (
                vocabulary_id INTEGER PRIMARY KEY REFERENCES vocabulary(id) ON DELETE CASCADE,
                saved_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS learning_sessions (
                id TEXT PRIMARY KEY,
                collection_id INTEGER REFERENCES collections(id),
                mode TEXT NOT NULL,
                question_index INTEGER NOT NULL DEFAULT 0,
                score INTEGER NOT NULL DEFAULT 0,
                total INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS learning_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES learning_sessions(id) ON DELETE CASCADE,
                vocabulary_id INTEGER NOT NULL REFERENCES vocabulary(id),
                expected TEXT NOT NULL,
                answer TEXT NOT NULL,
                correct INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS progress (
                vocabulary_id INTEGER PRIMARY KEY REFERENCES vocabulary(id) ON DELETE CASCADE,
                correct_count INTEGER NOT NULL DEFAULT 0,
                incorrect_count INTEGER NOT NULL DEFAULT 0,
                mastery REAL NOT NULL DEFAULT 0,
                last_reviewed TEXT,
                next_review TEXT,
                state TEXT NOT NULL DEFAULT 'new'
            );
            """
        )
        self.seed()
        self.connection.commit()

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value or [], ensure_ascii=True)

    def seed(self) -> None:
        words = [
            ("environment", "môi trường", "the natural world and surroundings", "noun", ["surroundings", "ecosystem"], ["We must protect the environment."], ["Environment"], "IELTS"),
            ("pollution", "ô nhiễm", "harmful substances in the environment", "noun", ["contamination"], ["Air pollution affects our health."], ["Environment"], "IELTS"),
            ("sustainable", "bền vững", "able to continue without exhausting resources", "adjective", ["viable", "renewable"], ["We need sustainable energy."], ["Environment", "Business"], "IELTS"),
            ("beautiful", "đẹp", "pleasing to the senses or mind aesthetically", "adjective", ["attractive", "lovely", "pretty"], ["She lives in a beautiful house."], ["Daily English"], "general"),
        ]
        for word, meaning, definition, pos, synonyms, examples, topics, source in words:
            self.connection.execute(
                """INSERT OR IGNORE INTO vocabulary
                (word, normalized_word, vietnamese_meaning, definition, part_of_speech,
                 synonyms_json, examples_json, topics_json, source, ielts_relevance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (word, word.casefold(), meaning, definition, pos, self._json(synonyms),
                 self._json(examples), self._json(topics), source, int(source == "IELTS")),
            )
        for category, name, description in (
            ("IELTS", "Environment", "High-frequency environmental vocabulary"),
            ("TOEIC", "Workplace", "Practical English for work"),
            ("Topics", "Environment", "Everyday environmental terms"),
        ):
            self.connection.execute("INSERT OR IGNORE INTO collections (name, category, description) VALUES (?, ?, ?)", (name, category, description))
        environment = self.connection.execute("SELECT id FROM collections WHERE category='IELTS' AND name='Environment'").fetchone()[0]
        ids = self.connection.execute("SELECT id FROM vocabulary WHERE normalized_word IN ('environment','pollution','sustainable') ORDER BY id").fetchall()
        for position, row in enumerate(ids):
            self.connection.execute("INSERT OR IGNORE INTO collection_items VALUES (?, ?, ?)", (environment, row[0], position))

    def vocabulary_row(self, normalized_word: str) -> sqlite3.Row | None:
        return self.connection.execute("SELECT * FROM vocabulary WHERE normalized_word = ?", (normalized_word.casefold().strip(),)).fetchone()

    def vocabulary_json(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        for key in ("synonyms_json", "antonyms_json", "examples_json", "topics_json", "metadata_json"):
            item[key.removesuffix("_json")] = json.loads(item.pop(key))
        return item

    def save_word(self, vocabulary_id: int) -> None:
        self.connection.execute("INSERT OR IGNORE INTO user_vocabulary VALUES (?, ?)", (vocabulary_id, utc_now()))
        self.connection.execute("INSERT OR IGNORE INTO progress (vocabulary_id, next_review) VALUES (?, ?)", (vocabulary_id, utc_now()))
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

