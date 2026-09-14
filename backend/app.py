from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes import create_router
from backend.domain.airwrite_runtime import AirWriteRuntime
from backend.storage.database import Database

# Resolve frontend directory relative to this file's location
_FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


def create_app(database_path: str = "data/learn_english.sqlite3") -> FastAPI:
    application = FastAPI(title="AirWrite Local Learn English", version="2.0.0")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # allow same-origin requests from the mounted frontend
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
    database = Database(database_path)
    application.state.database = database
    try:
        airwrite = AirWriteRuntime()
        application.state.airwrite_status = "ready"
    except Exception as error:  # model artifacts may be unavailable in a fresh checkout
        airwrite = None
        application.state.airwrite_status = f"unavailable: {error}"

    # --- API routes (must be registered BEFORE static files) ---
    application.include_router(create_router(database, airwrite), prefix="/api")

    # --- Serve frontend index.html at root ---
    @application.get("/", include_in_schema=False)
    def serve_index() -> FileResponse:
        return FileResponse(_FRONTEND_DIR / "index.html")

    # --- Serve frontend static files at /static (CSS, JS, assets) ---
    # Mounted at /static to avoid conflicting with /docs, /api, /redoc
    if _FRONTEND_DIR.exists():
        application.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="frontend")

    @application.on_event("shutdown")
    def close_database() -> None:
        database.close()

    return application


app = create_app()

