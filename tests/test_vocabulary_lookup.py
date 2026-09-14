import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.storage.database import Database

sys.stdout.reconfigure(encoding='utf-8')
db = Database(Path('data/learn_english.sqlite3'))

words = [
    'is', 'am', 'are', 'i', 'a', 'cannot', 'have', 'do', 'can', 'should',
    'walks', 'studying', 'played', 'shoes', 'boxes',
    'helo', 'technology', 'database', 'algorithm', 'opportunity'
]

def test_standard_vocabulary_lookups():
    cur = db.connection.cursor()
    cur.execute("SELECT count(*) FROM vocabulary")
    total_count = cur.fetchone()[0]
    assert 3000 <= total_count <= 3500, f"Expected between 3000 and 3500 words, got {total_count}"

    for w in words:
        v = db.vocabulary(w)
        assert v is not None, f"Word '{w}' should be found in vocabulary"
        assert len(v.get("vietnamese_meaning", "")) > 0, f"Word '{w}' must have a Vietnamese meaning"
