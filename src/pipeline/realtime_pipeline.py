"""Real-time filtering pipeline.

Connects to Polymarket's WebSocket, receives live price updates,
runs the Kalman filter in real-time, monitors for regime changes,
and logs results to SQLite.
"""

import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
from loguru import logger

from src.data.models import MarketObservation
from src.data.polymarket_client import PolymarketClient
from src.data.storage import MarketStorage
from src.data.websocket_client import PolymarketWebSocket
from src.detection.regime_detector import RegimeAlert, RegimeDetector
from src.filters.adaptive_kalman import AdaptiveKalmanFilter
from src.filters.noise_estimation import compute_observation_noise

# Default order book polling interval in seconds
ORDER_BOOK_POLL_INTERVAL: float = 30.0


class RealTimeFilterPipeline:
    """Live filtering pipeline for Polymarket prediction markets.

    Pipeline steps per price update:
    1. Receive price update via WebSocket
    2. Compute observation noise R_t from latest order book
    3. Run Kalman predict + update
    4. Check regime detector
    5. Log state to SQLite
    6. Emit alert if regime change detected

    Parameters
    ----------
    market_ids : list[str]
        Token IDs to subscribe to.
    Q_base : float
        Baseline process noise for the adaptive filter.
    R_default : float
        Default observation noise (used when order book unavailable).
    storage : MarketStorage or None
        Storage backend for logging. Created with defaults if None.

    Examples
    --------
    >>> pipeline = RealTimeFilterPipeline(market_ids=["0x..."])
    >>> pipeline.run(duration_minutes=30)
    """

    def __init__(
        self,
        market_ids: list[str],
        Q_base: float = 1e-4,
        R_default: float = 1e-3,
        storage: MarketStorage | None = None,
    ) -> None:
        self.market_ids = market_ids
        self.storage = storage or MarketStorage()

        # One filter and detector per market
        self.filters: dict[str, AdaptiveKalmanFilter] = {}
        self.detectors: dict[str, RegimeDetector] = {}

        for mid in market_ids:
            self.filters[mid] = AdaptiveKalmanFilter(Q_base=Q_base, R=R_default)
            self.detectors[mid] = RegimeDetector()

        self._client = PolymarketClient()
        self._last_book_poll: float = 0.0
        self._latest_R: dict[str, float] = {mid: R_default for mid in market_ids}
        self._update_count: int = 0
        self._alerts: list[RegimeAlert] = []

    def run(self, duration_minutes: int = 60) -> None:
        """Start the real-time pipeline.

        Parameters
        ----------
        duration_minutes : int
            How long to run the pipeline.
        """
        logger.info(
            "Starting real-time pipeline for {} markets, duration={}min",
            len(self.market_ids), duration_minutes,
        )

        ws = PolymarketWebSocket(
            on_price_update=self._on_price_update,
            market_ids=self.market_ids,
        )
        ws.connect(duration_seconds=duration_minutes * 60)

    def _on_price_update(
        self, market_id: str, price: float, timestamp: datetime,
    ) -> None:
        """Process a single price update.

        Parameters
        ----------
        market_id : str
            Token ID of the market.
        price : float
            New observed price.
        timestamp : datetime
            When the update occurred.
        """
        if market_id not in self.filters:
            return

        self._update_count += 1

        # Periodically poll order book for R_t
        self._maybe_update_orderbook(market_id)

        # Run filter
        R_t = self._latest_R.get(market_id, 1e-3)
        state = self.filters[market_id].step(price, R_t=R_t)

        # Check regime detector
        alert = self.detectors[market_id].check(
            innovation=state.innovation,
            S=state.S,
            timestamp=timestamp,
        )

        if alert.detected:
            self._alerts.append(alert)
            logger.warning(
                "REGIME CHANGE: market={}, method={}, severity={:.2f}: {}",
                market_id[:12], alert.method, alert.severity, alert.description,
            )

        # Log to storage
        obs = MarketObservation(
            timestamp=timestamp,
            market_id=market_id,
            market_question="",
            yes_price=price,
            midpoint=price,
            spread=0.0,
        )
        self.storage.save_observation(obs)

        if self._update_count % 100 == 0:
            logger.info(
                "Pipeline: {} updates processed, {} alerts",
                self._update_count, len(self._alerts),
            )

    def _maybe_update_orderbook(self, market_id: str) -> None:
        """Poll order book if enough time has passed since last poll."""
        now = time.time()
        if now - self._last_book_poll < ORDER_BOOK_POLL_INTERVAL:
            return

        try:
            book = self._client.get_book(market_id)
            # Build a minimal observation for noise estimation
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            best_bid = float(bids[0]["price"]) if bids else 0.4
            best_ask = float(asks[0]["price"]) if asks else 0.6

            obs = MarketObservation(
                timestamp=datetime.now(timezone.utc),
                market_id=market_id,
                market_question="",
                yes_price=(best_bid + best_ask) / 2,
                spread=best_ask - best_bid,
                total_depth=1000.0,
                imbalance=0.5,
                num_trades_1h=50,
            )
            self._latest_R[market_id] = compute_observation_noise(obs)
            self._last_book_poll = now
        except Exception as e:
            logger.debug("Order book poll failed: {}", e)

    def get_alerts(self) -> list[RegimeAlert]:
        """Return all regime change alerts.

        Returns
        -------
        list[RegimeAlert]
            All alerts emitted during the pipeline run.
        """
        return self._alerts.copy()

    def get_filter_states(self) -> dict[str, float]:
        """Return current state estimates for all markets.

        Returns
        -------
        dict[str, float]
            Market ID -> current filtered probability.
        """
        return {mid: f.x for mid, f in self.filters.items()}
