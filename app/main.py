from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.rate_outfit import router as outfit_router

def create_app() -> FastAPI:
    app = FastAPI(
        title = "ToneFit API",
        version = "0.0.1"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(outfit_router)

    return app

app = create_app()