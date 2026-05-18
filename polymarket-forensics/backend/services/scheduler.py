"""
Orchestrates all background services as concurrent asyncio tasks.
"""
from __future__ import annotations

import asyncio

from services import (
    chain_tracer, cluster_detector, resolution_tracker,
    scoring_engine, trade_ingester,
)
from utils.logging import get_logger

log = get_logger("scheduler")


class Scheduler:
    def __init__(self) -> None:
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

    def start(self) -> None:
        if self._tasks:
            return
        log.info("starting background services")
        self._tasks = [
            asyncio.create_task(trade_ingester.run_loop(self._stop), name="ingester"),
            asyncio.create_task(scoring_engine.run_loop(self._stop), name="scoring"),
            asyncio.create_task(cluster_detector.run_loop(self._stop), name="cluster"),
            asyncio.create_task(resolution_tracker.run_loop(self._stop), name="resolution"),
            asyncio.create_task(chain_tracer.run_worker(self._stop), name="tracer"),
        ]

    async def stop(self) -> None:
        log.info("stopping background services")
        self._stop.set()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []


scheduler = Scheduler()
