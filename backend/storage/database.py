from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class Database:
    def __init__(self, path: str | Path = "data/learn_english.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._create_schema()
        self._seed()

    def close(self) -> None:
        self.connection.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS vocabulary (
                id INTEGER PRIMARY KEY,
                word TEXT NOT NULL,
                normalized_word TEXT NOT NULL UNIQUE,
                vietnamese_meaning TEXT NOT NULL,
                definition TEXT NOT NULL,
                part_of_speech TEXT,
                synonyms TEXT NOT NULL DEFAULT '[]',
                antonyms TEXT NOT NULL DEFAULT '[]',
                examples TEXT NOT NULL DEFAULT '[]'
            );
            CREATE TABLE IF NOT EXISTS collections (
                id INTEGER PRIMARY KEY,
                category TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS collection_items (
                collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
                vocabulary_id INTEGER NOT NULL REFERENCES vocabulary(id) ON DELETE CASCADE,
                position INTEGER NOT NULL,
                PRIMARY KEY(collection_id, vocabulary_id)
            );
            CREATE TABLE IF NOT EXISTS saved_vocabulary (
                vocabulary_id INTEGER PRIMARY KEY REFERENCES vocabulary(id) ON DELETE CASCADE,
                saved_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS learning_sessions (
                id TEXT PRIMARY KEY,
                collection_id INTEGER REFERENCES collections(id),
                mode TEXT NOT NULL,
                question_index INTEGER NOT NULL DEFAULT 0,
                score INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS learning_attempts (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES learning_sessions(id),
                vocabulary_id INTEGER NOT NULL REFERENCES vocabulary(id),
                expected TEXT NOT NULL,
                answer TEXT NOT NULL,
                correct INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS progress (
                vocabulary_id INTEGER PRIMARY KEY REFERENCES vocabulary(id),
                correct_count INTEGER NOT NULL DEFAULT 0,
                incorrect_count INTEGER NOT NULL DEFAULT 0,
                mastery REAL NOT NULL DEFAULT 0,
                state TEXT NOT NULL DEFAULT 'new',
                last_reviewed TEXT,
                next_review TEXT
            );
            """
        )
        self.connection.commit()
        # Migration: add antonyms column if it doesn't exist yet (for existing databases)
        try:
            self.connection.execute("ALTER TABLE vocabulary ADD COLUMN antonyms TEXT NOT NULL DEFAULT '[]'")
            self.connection.commit()
        except Exception:
            pass  # Column already exists

    def _seed(self) -> None:
        words = (
            # --- Home ---
            ("house", "ngôi nhà", "a building for human habitation, especially one lived in by a family", "noun", '["home", "residence", "dwelling"]', '["workplace"]', '["They bought a new house.", "The house has three bedrooms."]'),
            ("kitchen", "nhà bếp", "a room where food is prepared and cooked", "noun", '["cooking area", "galley"]', '[]', '["She cooks dinner in the kitchen.", "The kitchen smells wonderful."]'),
            ("bedroom", "phòng ngủ", "a room used for sleeping", "noun", '["sleeping room", "chamber"]', '[]', '["The children share a bedroom.", "He reads before bed in his bedroom."]'),
            ("clean", "sạch sẽ", "free from dirt, marks, or stains", "adjective", '["tidy", "spotless", "neat"]', '["dirty", "messy", "filthy"]', '["Keep your room clean.", "She cleaned the house every weekend."]'),
            ("furniture", "đồ nội thất", "large movable equipment used to make a room suitable for living", "noun", '["furnishings", "fittings"]', '[]', '["They bought new furniture.", "The furniture is made of wood."]'),
            ("garden", "khu vườn", "a piece of ground adjoining a house, used for growing flowers or vegetables", "noun", '["yard", "lawn", "backyard"]', '[]', '["She planted roses in the garden.", "The children play in the garden."]'),
            # --- Work ---
            ("workplace", "nơi làm việc", "a place where people work, such as an office or factory", "noun", '["office", "work environment", "workspace"]', '[]', '["A healthy workplace supports employees.", "The workplace should be safe."]'),
            ("meeting", "cuộc họp", "an assembly of people for a particular purpose, especially for discussion", "noun", '["conference", "gathering", "session"]', '[]', '["We have a meeting at 9am.", "The meeting lasted two hours."]'),
            ("deadline", "hạn chót", "the latest time by which something should be completed", "noun", '["due date", "time limit", "cutoff"]', '[]', '["The deadline is tomorrow.", "She always meets her deadlines."]'),
            ("salary", "lương", "a fixed regular payment for employment, paid monthly or yearly", "noun", '["wage", "pay", "income", "earnings"]', '[]', '["He received a salary raise.", "The salary is paid every month."]'),
            ("career", "sự nghiệp", "an occupation undertaken for a significant period with opportunities for progress", "noun", '["profession", "vocation", "occupation"]', '["hobby"]', '["She built a successful career in medicine.", "Choose a career you love."]'),
            ("colleague", "đồng nghiệp", "a person with whom one works in the same organization", "noun", '["coworker", "associate", "teammate"]', '["competitor"]', '["My colleague helped me finish the report.", "We celebrated with our colleagues."]'),
            # --- School ---
            ("student", "học sinh", "a person who is studying at a school or university", "noun", '["pupil", "learner", "scholar"]', '["teacher"]', '["She is an excellent student.", "The students worked on the project together."]'),
            ("teacher", "giáo viên", "a person who teaches, especially in a school", "noun", '["instructor", "educator", "professor"]', '["student"]', '["The teacher explained the lesson clearly.", "She became a teacher after university."]'),
            ("homework", "bài tập về nhà", "school work that a student is required to do at home", "noun", '["assignment", "task", "coursework"]', '[]', '["He forgot to do his homework.", "The teacher gave them homework every day."]'),
            ("exam", "kỳ thi", "a formal test of a person knowledge or ability in a subject", "noun", '["test", "quiz", "assessment"]', '[]', '["She studied hard for the final exam.", "The exam results were published online."]'),
            ("library", "thư viện", "a building containing collections of books for reading or borrowing", "noun", '["reading room", "archive"]', '[]', '["He spent the afternoon in the library.", "The library has thousands of books."]'),
            ("education", "giáo dục", "the process of receiving or giving systematic instruction", "noun", '["instruction", "schooling", "learning"]', '["ignorance"]', '["Education opens new opportunities.", "Quality education is a human right."]'),
            # --- Food ---
            ("breakfast", "bữa sáng", "the first meal of the day, eaten in the morning", "noun", '["morning meal"]', '["dinner"]', '["She eats breakfast at 7am.", "A healthy breakfast includes fruit and eggs."]'),
            ("delicious", "ngon", "highly pleasing to the taste", "adjective", '["tasty", "yummy", "scrumptious", "flavorful"]', '["tasteless", "bland"]', '["The soup was absolutely delicious.", "That cake smells delicious."]'),
            ("hungry", "đói", "having a feeling of discomfort caused by lack of food", "adjective", '["famished", "starving", "ravenous"]', '["full", "satisfied"]', '["I am hungry after the long walk.", "He was so hungry he ate two plates."]'),
            ("cook", "nấu ăn", "to prepare food by heating it", "verb", '["prepare", "make", "bake", "fry"]', '[]', '["She loves to cook on weekends.", "He learned to cook from his mother."]'),
            ("restaurant", "nhà hàng", "a place where people pay to eat meals cooked and served on site", "noun", '["diner", "eatery", "bistro", "cafe"]', '[]', '["We went to an Italian restaurant.", "The restaurant was fully booked."]'),
            # --- Travel ---
            ("travel", "du lịch", "to make a journey, typically of some length", "verb", '["journey", "voyage", "trip", "tour"]', '["stay", "remain"]', '["She loves to travel abroad.", "We traveled by train to the coast."]'),
            ("airport", "sân bay", "a complex of runways and buildings for aircraft takeoff and landing", "noun", '["airfield", "terminal", "aerodrome"]', '[]', '["We arrived at the airport two hours early.", "The airport is very busy on weekends."]'),
            ("hotel", "khách sạn", "an establishment providing accommodation and meals for travellers", "noun", '["inn", "motel", "lodging", "accommodation"]', '[]', '["We stayed at a five-star hotel.", "The hotel room had a sea view."]'),
            ("passport", "hộ chiếu", "an official document certifying your identity for international travel", "noun", '["travel document", "ID"]', '[]', '["She renewed her passport before the trip.", "Do not forget your passport!"]'),
            ("adventure", "cuộc phiêu lưu", "an unusual and exciting experience or activity", "noun", '["expedition", "journey", "quest", "exploration"]', '["routine", "boredom"]', '["Traveling solo is a great adventure.", "Every trip is a new adventure."]'),
            # --- Nature ---
            ("environment", "môi trường", "the natural world around us, including the air, water, and land", "noun", '["surroundings", "ecosystem", "habitat"]', '[]', '["We must protect the environment.", "The environment is affected by climate change."]'),
            ("pollution", "ô nhiễm", "harmful substances introduced into the natural environment, causing damage", "noun", '["contamination", "toxicity"]', '["cleanliness", "purity"]', '["Pollution affects public health.", "Air pollution is a major problem in cities."]'),
            ("sustainable", "bền vững", "able to be maintained without harming resources or the environment", "adjective", '["viable", "renewable", "eco-friendly"]', '["unsustainable", "destructive"]', '["We need sustainable solutions.", "Sustainable farming protects the soil."]'),
            ("nature", "thiên nhiên", "the phenomena of the physical world collectively, including plants, animals, and landscape", "noun", '["wilderness", "environment", "natural world"]', '["urban", "artificial"]', '["She loves spending time in nature.", "Nature provides everything we need."]'),
            ("wildlife", "động vật hoang dã", "wild animals living in their natural environment", "noun", '["fauna", "animals", "creatures"]', '[]', '["The national park protects local wildlife.", "Wildlife photography is her hobby."]'),
        )
        self.connection.executemany(
            "INSERT OR IGNORE INTO vocabulary (word, normalized_word, vietnamese_meaning, definition, part_of_speech, synonyms, antonyms, examples) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(w, w.casefold(), vi, d, pos, syn, ant, ex) for w, vi, d, pos, syn, ant, ex in words],
        )

        collections_data = [
            ("🏠 Nhà cửa", "Nhà & Gia đình", "Từ vựng về ngôi nhà, phòng ốc và sinh hoạt hằng ngày.",
             ["house", "kitchen", "bedroom", "clean", "furniture", "garden"]),
            ("💼 Công việc", "Nơi làm việc", "Từ vựng thường dùng trong môi trường văn phòng và công sở.",
             ["workplace", "meeting", "deadline", "salary", "career", "colleague"]),
            ("🏫 Trường học", "Trường & Lớp học", "Từ vựng dành cho học sinh, sinh viên và môi trường học đường.",
             ["student", "teacher", "homework", "exam", "library", "education"]),
            ("🍜 Ăn uống", "Đồ ăn & Thức uống", "Từ vựng về bữa ăn, nấu ăn và nhà hàng.",
             ["breakfast", "delicious", "hungry", "cook", "restaurant"]),
            ("✈️ Du lịch", "Khám phá & Di chuyển", "Từ vựng cần thiết khi đi du lịch trong và ngoài nước.",
             ["travel", "airport", "hotel", "passport", "adventure"]),
            ("🌿 Thiên nhiên", "Môi trường & Thiên nhiên", "Từ vựng về thiên nhiên và bảo vệ môi trường sống.",
             ["environment", "pollution", "sustainable", "nature", "wildlife"]),
        ]

        for category, name, description, word_list in collections_data:
            existing = self.connection.execute(
                "SELECT id FROM collections WHERE name=?", (name,)
            ).fetchone()
            if existing is None:
                self.connection.execute(
                    "INSERT INTO collections (category, name, description) VALUES (?, ?, ?)",
                    (category, name, description),
                )
                self.connection.commit()

            col_row = self.connection.execute(
                "SELECT id FROM collections WHERE name=?", (name,)
            ).fetchone()
            if col_row is None:
                continue
            col_id = col_row["id"]

            for position, word_key in enumerate(word_list):
                vocab_row = self.connection.execute(
                    "SELECT id FROM vocabulary WHERE normalized_word=?", (word_key,)
                ).fetchone()
                if vocab_row:
                    self.connection.execute(
                        "INSERT OR IGNORE INTO collection_items (collection_id, vocabulary_id, position) VALUES (?, ?, ?)",
                        (col_id, vocab_row["id"], position),
                    )
        self.connection.commit()

    def vocabulary(self, normalized_word: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM vocabulary WHERE normalized_word=?", (normalized_word.casefold().strip(),)
        ).fetchone()
        if row is None:
            return None
        import json

        item = dict(row)
        item["synonyms"] = json.loads(item.get("synonyms") or "[]")
        item["antonyms"] = json.loads(item.get("antonyms") or "[]")
        item["examples"] = json.loads(item.get("examples") or "[]")
        return item

    def save(self, vocabulary_id: int) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO saved_vocabulary (vocabulary_id, saved_at) VALUES (?, ?)",
            (vocabulary_id, utc_now()),
        )
        self.connection.execute(
            "INSERT OR IGNORE INTO progress (vocabulary_id, next_review) VALUES (?, ?)",
            (vocabulary_id, utc_now()),
        )
        self.connection.commit()
