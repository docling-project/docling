from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

def cors_config(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"),
        allow_headers=['*'],
    )

