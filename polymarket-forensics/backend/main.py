"""Polymarket Forensics Dashboard - FastAPI entry point."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import (
    routes_alerts,
    routes_clusters,
    routes_export,
    routes_markets,
    routes_search,
    routes_stats,
    routes_wallets,
    websocket,
)
from config import get_settings
from exceptions import (
    AlertNotFoundError,
    ClusterNotFoundError,
    ForensicsError,
    WalletNotFoundError,
)
from services.scheduler import scheduler
from utils.logging import get_logger, setup_logging

settings = get_settings()
setup_logging(settings.log_level)
log = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    log.info("app_start",
             alchemy_configured=bool(settings.alchemy_key),
             cors_origins=settings.cors_origins)
    scheduler.start()
    yield
    await scheduler.stop()


app = FastAPI(
    title="Polymarket Forensics",
    description="Real-time on-chain forensics for Polymarket insider trading.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["*"],
)


@app.middleware("http")
async def correlation_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    cid = request.headers.get("x-correlation-id") or uuid.uuid4().hex
    structlog.contextvars.bind_contextvars(correlation_id=cid)
    try:
        response = await call_next(request)
    finally:
        structlog.contextvars.unbind_contextvars("correlation_id")
    response.headers["x-correlation-id"] = cid
    return response


@app.exception_handler(WalletNotFoundError)
async def _wallet_not_found(_: Request, exc: WalletNotFoundError) -> JSONResponse:
    return JSONResponse({"detail": exc.message, "context": exc.context}, status_code=404)


@app.exception_handler(ClusterNotFoundError)
async def _cluster_not_found(_: Request, exc: ClusterNotFoundError) -> JSONResponse:
    return JSONResponse({"detail": exc.message, "context": exc.context}, status_code=404)


@app.exception_handler(AlertNotFoundError)
async def _alert_not_found(_: Request, exc: AlertNotFoundError) -> JSONResponse:
    return JSONResponse({"detail": exc.message, "context": exc.context}, status_code=404)


@app.exception_handler(ForensicsError)
async def _forensics_error(_: Request, exc: ForensicsError) -> JSONResponse:
    log.warning("api_forensics_error", error=exc.message, context=exc.context)
    return JSONResponse({"detail": exc.message, "context": exc.context}, status_code=500)


app.include_router(routes_stats.router)
app.include_router(routes_wallets.router)
app.include_router(routes_clusters.router)
app.include_router(routes_alerts.router)
app.include_router(routes_markets.router)
app.include_router(routes_search.router)
app.include_router(routes_export.router)
app.include_router(websocket.router)


@app.get("/", tags=["meta"])
async def root() -> dict[str, object]:
    return {
        "name": "polymarket-forensics",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
