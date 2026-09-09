from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import create_router
from backend.domain.airwrite_runtime import AirWriteRuntime
from backend.storage.database import Database


def create_app(database_path: str = "data/learn_english.sqlite3") -> FastAPI:
    application = FastAPI(title="AirWrite Local Learn English", version="2.0.0")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ],
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
    application.include_router(create_router(database, airwrite), prefix="/api")

    @application.on_event("shutdown")
    def close_database() -> None:
        database.close()

    return application


app = create_app()
