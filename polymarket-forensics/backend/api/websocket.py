"""WebSocket broadcast hub for real-time alert push to the frontend."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from utils.logging import get_logger

log = get_logger(__name__)

HEARTBEAT_SEC = 25


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections.add(ws)
        log.info("ws_connected", total=len(self._connections))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(ws)
        log.info("ws_disconnected", total=len(self._connections))

    async def broadcast(self, payload: dict[str, Any]) -> None:
        msg = json.dumps(payload, default=str)
        async with self._lock:
            targets = list(self._connections)
        dead: list[WebSocket] = []
        for ws in targets:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._connections.discard(ws)
            log.info("ws_dropped_dead", count=len(dead))


manager = ConnectionManager()
router = APIRouter()


async def _heartbeat(ws: WebSocket) -> None:
    while True:
        await asyncio.sleep(HEARTBEAT_SEC)
        try:
            await ws.send_text('{"type":"ping"}')
        except Exception:
            return


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await manager.connect(ws)
    heartbeat_task = asyncio.create_task(_heartbeat(ws))
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("ws_receive_error")
    finally:
        heartbeat_task.cancel()
        await manager.disconnect(ws)
