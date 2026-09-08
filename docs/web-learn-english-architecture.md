# AirWrite Learn English

This web layer extends the existing Python AirWrite V2 runtime instead of creating a second recognition engine.

## Architecture

- **AirWrite core:** existing MediaPipe, AirCanvas, shared 28x28 preprocessing, E02 identity model, top-3 policy, case resolver, and word segmentation remain the source of truth. `app/web/airwrite_adapter.py` injects the already-configured `CharacterRecognitionService`; the web recognition API models the same session/segment/prediction/draft/review/commit contract.
- **Backend:** `app/web/app.py` exposes FastAPI routes under `/api`; `app/web/api.py` owns recognition, dictionary, vocabulary, collections, learning, and progress orchestration.
- **Database:** local SQLite (`data/vocabulary.sqlite3`) stores vocabulary, many-to-many collection items, saved words, learning sessions/attempts, and progress. It is intentionally user-ready by using IDs and a future user boundary.
- **Frontend:** `app/web/static` is a responsive static client. It uses browser camera permission and a drawing canvas, presents uncertainty, and calls the backend rather than embedding dictionary data.
- **Learning engine:** a single session model supports `en_to_vi`, `vi_to_en`, `airwrite`, and `cloze`; attempts update mastery and review timestamps.

## API

`POST/GET/PATCH /api/recognition/sessions`, `POST /api/recognition/sessions/{id}/segments`, `POST /api/recognition/sessions/{id}/commit`, `GET /api/dictionary/{word}`, `POST /api/vocabulary/{word}/save`, `GET /api/vocabulary`, `GET /api/collections`, `GET /api/collections/{id}`, `POST/GET /api/learning/sessions`, `POST /api/learning/sessions/{id}/answer`, and `GET /api/progress`.

Recognition accepts a segment only as **accepted** when confidence is at least 0.60 and margin at least 0.15. Otherwise it remains **uncertain** and commit returns `409` until corrected.

## Run

```powershell
python -m pip install -r requirements.txt -r requirements-web.txt
python -m uvicorn app.web.app:app --reload
```

Open `http://127.0.0.1:8000`. Raw webcam frames are never uploaded or stored by this layer.

## Limitations

The current browser demo sends prediction events to the API; wiring MediaPipe frame events to those calls is the next adapter step. Dictionary data is a small local seed, authentication is intentionally deferred, and camera support depends on browser permission and HTTPS/localhost policy.
