"""WebSocket client for real-time Polymarket price updates.

Connects to the Polymarket CLOB WebSocket to receive live price changes.
Includes reconnection logic with exponential backoff.

WebSocket endpoint: wss://ws-subscriptions-clob.polymarket.com/ws/
"""

import json
import time
from datetime import datetime, timezone
from typing import Any, Callable

from loguru import logger

try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False

# WebSocket endpoint
WS_ENDPOINT: str = "wss://ws-subscriptions-clob.polymarket.com/ws/"

# Reconnection parameters
MAX_RECONNECT_ATTEMPTS: int = 5
INITIAL_BACKOFF_SECONDS: float = 2.0
MAX_BACKOFF_SECONDS: float = 60.0


class PolymarketWebSocket:
    """WebSocket client for real-time Polymarket price updates.

    Subscribes to market channels and dispatches price update messages
    to a callback function.

    Parameters
    ----------
    on_price_update : callable
        Callback function(market_id: str, price: float, timestamp: datetime).
    market_ids : list[str]
        Token IDs to subscribe to.

    Examples
    --------
    >>> def callback(market_id, price, ts):
    ...     print(f"{market_id}: {price}")
    >>> ws = PolymarketWebSocket(on_price_update=callback, market_ids=["0x..."])
    >>> ws.connect(duration_seconds=60)
    """

    def __init__(
        self,
        on_price_update: Callable[[str, float, datetime], None],
        market_ids: list[str],
    ) -> None:
        if not HAS_WEBSOCKET:
            raise ImportError("websocket-client is required for real-time updates")

        self.on_price_update = on_price_update
        self.market_ids = market_ids
        self._ws: Any = None
        self._running: bool = False
        self._reconnect_count: int = 0

    def connect(self, duration_seconds: int = 3600) -> None:
        """Connect to the WebSocket and listen for updates.

        Parameters
        ----------
        duration_seconds : int
            Maximum connection duration in seconds.
        """
        self._running = True
        start_time = time.time()

        while self._running and (time.time() - start_time) < duration_seconds:
            try:
                self._connect_once(duration_seconds - (time.time() - start_time))
            except Exception as e:
                logger.warning("WebSocket error: {}", e)
                if self._should_reconnect():
                    backoff = self._get_backoff()
                    logger.info("Reconnecting in {:.1f}s...", backoff)
                    time.sleep(backoff)
                else:
                    break

    def _connect_once(self, remaining_seconds: float) -> None:
        """Establish a single WebSocket connection.

        Parameters
        ----------
        remaining_seconds : float
            Time remaining before timeout.
        """
        ws = websocket.WebSocketApp(
            WS_ENDPOINT,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._ws = ws

        # Run with timeout
        ws.run_forever(
            ping_interval=30,
            ping_timeout=10,
        )

    def _on_open(self, ws: Any) -> None:
        """Subscribe to market channels on connection."""
        self._reconnect_count = 0
        logger.info("WebSocket connected, subscribing to {} markets", len(self.market_ids))

        for token_id in self.market_ids:
            subscribe_msg = json.dumps({
                "type": "subscribe",
                "channel": "market",
                "market": token_id,
            })
            ws.send(subscribe_msg)

    def _on_message(self, ws: Any, message: str) -> None:
        """Parse incoming price update messages."""
        try:
            data = json.loads(message)
            event_type = data.get("event_type", data.get("type", ""))

            if event_type in ("price_change", "trade", "book"):
                market_id = data.get("market", data.get("asset_id", ""))
                price_str = data.get("price", data.get("last_price", ""))
                ts_str = data.get("timestamp", "")

                if market_id and price_str:
                    price = float(price_str)
                    if ts_str:
                        try:
                            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        except ValueError:
                            ts = datetime.now(timezone.utc)
                    else:
                        ts = datetime.now(timezone.utc)

                    self.on_price_update(market_id, price, ts)
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug("Failed to parse message: {}", e)

    def _on_error(self, ws: Any, error: Exception) -> None:
        """Log WebSocket errors."""
        logger.error("WebSocket error: {}", error)

    def _on_close(self, ws: Any, close_status_code: int | None,
                   close_msg: str | None) -> None:
        """Handle connection close."""
        logger.info("WebSocket closed: code={}, msg={}", close_status_code, close_msg)

    def _should_reconnect(self) -> bool:
        """Check if we should attempt reconnection."""
        self._reconnect_count += 1
        return self._reconnect_count <= MAX_RECONNECT_ATTEMPTS

    def _get_backoff(self) -> float:
        """Calculate exponential backoff delay."""
        delay = INITIAL_BACKOFF_SECONDS * (2 ** (self._reconnect_count - 1))
        return min(delay, MAX_BACKOFF_SECONDS)

    def stop(self) -> None:
        """Stop the WebSocket connection."""
        self._running = False
        if self._ws:
            self._ws.close()
