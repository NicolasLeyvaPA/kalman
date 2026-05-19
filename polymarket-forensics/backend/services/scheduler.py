"""Orchestrate all background services as concurrent asyncio tasks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from services import (
    chain_tracer,
    cluster_detector,
    resolution_tracker,
    scoring_engine,
    trade_ingester,
)
from utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ServiceStatus:
    name: str
    running: bool
    last_started: str | None


class Scheduler:
    """Container for background service tasks. Single-instance."""

    def __init__(self) -> None:
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task[None]] = []
        self._started_at: str | None = None

    def start(self) -> None:
        if self._tasks:
            return
        from utils.time import utc_now
        self._started_at = utc_now().isoformat()
        log.info("scheduler_start")
        self._tasks = [
            asyncio.create_task(trade_ingester.run_loop(self._stop), name="ingester"),
            asyncio.create_task(scoring_engine.run_loop(self._stop), name="scoring"),
            asyncio.create_task(cluster_detector.run_loop(self._stop), name="cluster"),
            asyncio.create_task(resolution_tracker.run_loop(self._stop), name="resolution"),
            asyncio.create_task(chain_tracer.run_worker(self._stop), name="tracer"),
        ]

    async def stop(self) -> None:
        log.info("scheduler_stop")
        self._stop.set()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    def status(self) -> list[ServiceStatus]:
        return [
            ServiceStatus(
                name=t.get_name(),
                running=not t.done(),
                last_started=self._started_at,
            )
            for t in self._tasks
        ]


scheduler = Scheduler()
