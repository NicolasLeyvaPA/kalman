"""
Polymarket Forensics Dashboard - FastAPI entry point.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import (
    routes_alerts, routes_clusters, routes_markets, routes_search,
    routes_stats, routes_wallets, websocket,
)
from config import get_settings
from services.scheduler import scheduler
from utils.logging import setup_logging

setup_logging("INFO")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    allow_origins=[settings.frontend_origin, "http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_stats.router)
app.include_router(routes_wallets.router)
app.include_router(routes_clusters.router)
app.include_router(routes_alerts.router)
app.include_router(routes_markets.router)
app.include_router(routes_search.router)
app.include_router(websocket.router)


@app.get("/")
async def root() -> dict:
    return {
        "name": "polymarket-forensics",
        "version": "1.0.0",
        "endpoints": [
            "/stats/overview",
            "/stats/recent-trades",
            "/wallets",
            "/wallets/{address}",
            "/wallets/{address}/trades",
            "/wallets/{address}/funding",
            "/wallets/{address}/trace [POST]",
            "/wallets/{address}/rescore [POST]",
            "/clusters",
            "/clusters/{id}",
            "/clusters/graph/edges",
            "/alerts",
            "/alerts/{id}/dismiss [POST]",
            "/markets",
            "/markets/{id}/forensics",
            "/search?q=",
            "/ws  (WebSocket)",
        ],
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
