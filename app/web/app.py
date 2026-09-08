from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.web.api import create_router
from app.web.db import Database


def create_app(database_path: str = "data/vocabulary.sqlite3") -> FastAPI:
    application = FastAPI(title="AirWrite Learn English", version="1.0.0")
    application.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    database = Database(database_path)
    application.state.database = database
    application.include_router(create_router(database), prefix="/api")
    static_dir = Path(__file__).with_name("static")
    application.mount("/static", StaticFiles(directory=static_dir), name="static")

    @application.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    return application


app = create_app()

