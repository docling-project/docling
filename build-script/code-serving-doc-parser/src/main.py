import json
import os
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from common.logger import Logger
from service import service

logger = Logger.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    lifespan context manager
    """
    logger.info('lifespan start')

    yield

    logger.info('lifespan end')


app: FastAPI = FastAPI(lifespan=lifespan, title="CDN API", root_path="/api/cdn")
# config.router_config(app)


LOG_BODY_LIMIT = int(os.environ.get("LOG_BODY_LIMIT", 4096))


class LoggingMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        req_chunks = bytearray()

        async def recv_wrapper() -> Message:
            msg = await receive()
            if msg["type"] == "http.request":
                remaining = LOG_BODY_LIMIT - len(req_chunks)
                if remaining > 0:
                    req_chunks.extend(msg.get("body", b"")[:remaining])
            return msg

        resp_chunks = bytearray()

        async def send_wrapper(msg: Message) -> None:
            await send(msg)  # 즉시 흘려보냄 — 스트리밍 보존
            if msg["type"] == "http.response.body" and len(resp_chunks) < LOG_BODY_LIMIT:
                remaining = LOG_BODY_LIMIT - len(resp_chunks)
                resp_chunks.extend(msg.get("body", b"")[:remaining])

        path = scope.get("path", "")
        client = scope.get("client") or ("?", 0)
        method = scope.get("method", "?")
        query = scope.get("query_string", b"").decode("latin1")

        try:
            await self.app(scope, recv_wrapper, send_wrapper)
        except Exception:
            req_preview = "[FILE UPLOAD]" if "upload" in path else bytes(req_chunks).decode("utf-8", errors="replace")
            logger.error(
                f"[LoggingMiddleware] unhandled exception: ip={client[0]} method={method} "
                f"path={path} params={query} req_body={req_preview}\n{traceback.format_exc()}"
            )
            raise

        req_preview = "[FILE UPLOAD]" if "upload" in path else bytes(req_chunks).decode("utf-8", errors="replace")
        resp_preview = bytes(resp_chunks).decode("utf-8", errors="replace")
        logger.info(
            f"req: ip={client[0]} method={method} path={path} params={query} req_body={req_preview}"
        )
        logger.info(f"resp: {resp_preview}")


app.add_middleware(LoggingMiddleware)


@app.get(path="")
@app.get(path="/health")
async def health(request: Request):
    return {"status": "ok"}


@app.post(path="")
@app.post(path="/json")
async def run_json(request: Request):
    config = {}
    try:
        data = await request.json()
    except json.decoder.JSONDecodeError:
        data = await request.body()

    return await service.service(config=config, data=data)


@app.post(path="/multipart")
async def run_multipart(request: Request):
    config = {}
    data = await request.form()
    return await service.service(config=config, data=data)


if __name__ == '__main__':
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8086, reload=True)
