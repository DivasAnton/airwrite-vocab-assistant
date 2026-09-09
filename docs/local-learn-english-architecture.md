# Local Learn English Architecture

## Boundaries

The repository has three deliberate boundaries:

- `app/` is the AirWrite core: camera, hand tracking, canvas, segmentation, preprocessing, E02 identity inference, top-3 policy, and isolated whole-word draft review.
- `backend/` is the local learning application: recognition sessions, dictionary lookup, vocabulary storage, collections, learning sessions, answers, and progress.
- `frontend/` is a standalone browser client. It owns camera permission, canvas interaction, navigation, and learning feedback. It does not duplicate preprocessing or model inference.

The backend is local-only and listens on `127.0.0.1`. SQLite is the local source of truth. Raw camera video is never stored.

## Recognition boundary

Every recognition flow uses `isolated_whole_word`:

```text
Browser camera frames
  -> local `/api/recognition/sessions/{id}/frames`
  -> existing Python MediaPipe hand tracking + AirCanvas
  -> `/finish` invokes the existing whole-word pipeline
  -> segment predictions and top-3 candidates
  -> POST /api/recognition/sessions/{id}/segments
  -> draft review
  -> atomic commit
  -> dictionary / learning engine
```

The frontend never reimplements 28x28 preprocessing or loads the E02 model. A future browser camera adapter may call the existing Python pipeline through a local process boundary, but it must keep the same session and segment contract.

## Local API

- `POST/GET /api/recognition/sessions`
- `POST /api/recognition/sessions/{id}/frames`
- `POST /api/recognition/sessions/{id}/finish`
- `POST/PATCH/GET /api/recognition/sessions/{id}/...`
- `GET /api/dictionary/{word}`
- `POST /api/vocabulary/{word}/save`
- `GET /api/vocabulary`
- `GET /api/collections` and `/api/collections/{id}`
- `POST/GET /api/learning/sessions`
- `POST /api/learning/sessions/{id}/answer`
- `GET /api/progress`

## Data model

SQLite stores vocabulary entries, collections, collection items, saved vocabulary, learning sessions, attempts, and progress. Definitions, synonyms, and examples are stored as JSON arrays so one word can have one-to-many data without hard-coding it in the frontend.

## Run locally

Backend:

```powershell
python -m backend.run
```

Frontend, in a second terminal:

```powershell
python -m http.server 5173 --directory frontend
```

Open `http://127.0.0.1:5173`.
