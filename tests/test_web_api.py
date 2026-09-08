from fastapi.testclient import TestClient

from app.web.app import create_app


def client(tmp_path):
    return TestClient(create_app(str(tmp_path / "vocab.sqlite3")))


def test_recognition_requires_uncertain_resolution(tmp_path):
    c = client(tmp_path)
    session = c.post("/api/recognition/sessions").json()
    c.post(f"/api/recognition/sessions/{session['session_id']}/segments", json={"prediction":"e","confidence":0.57,"margin":0.08,"candidates":[{"label":"e","confidence":0.57},{"label":"a","confidence":0.22}]})
    response = c.post(f"/api/recognition/sessions/{session['session_id']}/commit")
    assert response.status_code == 409


def test_dictionary_save_learning_and_progress(tmp_path):
    c = client(tmp_path)
    entry = c.get("/api/dictionary/environment")
    assert entry.status_code == 200
    assert entry.json()["vietnamese_meaning"] == "môi trường"
    assert c.post("/api/vocabulary/environment/save").json()["saved"] is True
    assert c.get("/api/vocabulary").json()[0]["word"] == "environment"
    session = c.post("/api/learning/sessions", json={"mode":"airwrite"}).json()
    answer = c.post(f"/api/learning/sessions/{session['id']}/answer", json={"answer":"environment"})
    assert answer.json()["correct"] is True
    assert c.get("/api/progress").json()[0]["mastery"] > 0


def test_collection_is_seeded(tmp_path):
    c = client(tmp_path)
    collections = c.get("/api/collections?category=IELTS").json()
    assert collections[0]["name"] == "Environment"
    assert c.get(f"/api/collections/{collections[0]['id']}").json()["vocabulary"]

