"""High-level market data fetcher.

Provides convenience functions for finding liquid markets and building
MarketObservation time series from Polymarket API data.
"""

from datetime import datetime, timezone
from typing import Any

import pandas as pd
from loguru import logger

from src.data.models import MarketInfo, PriceHistory, PricePoint
from src.data.polymarket_client import PolymarketClient

# Minimum volume threshold (USD) for considering a market "liquid"
MIN_LIQUID_VOLUME_USD: float = 100_000.0


class MarketFetcher:
    """Fetches and structures market data from Polymarket.

    Wraps the low-level API client with higher-level operations like
    finding liquid markets and building price history DataFrames.

    Parameters
    ----------
    client : PolymarketClient or None
        API client instance. Created with defaults if None.
    """

    def __init__(self, client: PolymarketClient | None = None) -> None:
        self.client = client or PolymarketClient()

    def find_liquid_markets(self, n: int = 10,
                            min_volume: float = MIN_LIQUID_VOLUME_USD) -> list[MarketInfo]:
        """Find the most liquid active markets.

        Parameters
        ----------
        n : int
            Number of markets to return.
        min_volume : float
            Minimum 24h volume in USD to qualify as liquid.

        Returns
        -------
        list[MarketInfo]
            Markets sorted by volume (descending).
        """
        raw_markets = self.client.get_markets(limit=100, active=True)
        markets: list[MarketInfo] = []

        for m in raw_markets:
            volume = _safe_float(m.get("volume", 0))
            if volume < min_volume:
                continue

            # Extract token ID — markets have a tokens array with YES/NO outcomes
            token_id = ""
            tokens = m.get("clobTokenIds")
            if tokens and isinstance(tokens, str):
                # Sometimes it's a JSON string
                import json
                try:
                    token_list = json.loads(tokens)
                    token_id = token_list[0] if token_list else ""
                except (json.JSONDecodeError, IndexError):
                    token_id = tokens
            elif tokens and isinstance(tokens, list):
                token_id = tokens[0] if tokens else ""

            end_date = None
            if m.get("endDate"):
                try:
                    end_date = datetime.fromisoformat(
                        m["endDate"].replace("Z", "+00:00")
                    )
                except (ValueError, AttributeError):
                    pass

            info = MarketInfo(
                market_id=m.get("conditionId", m.get("id", "")),
                question=m.get("question", m.get("title", "Unknown")),
                token_id=token_id,
                slug=m.get("slug", ""),
                category=m.get("category", ""),
                end_date=end_date,
                active=bool(m.get("active", True)),
                description=m.get("description", ""),
            )
            markets.append(info)

        markets.sort(key=lambda mi: _market_volume(mi, raw_markets), reverse=True)
        logger.info("Found {} liquid markets (min volume ${:,.0f})", len(markets), min_volume)
        return markets[:n]

    def fetch_price_history(self, market_id: str, question: str = "",
                            interval: str = "max",
                            fidelity: str = "1h") -> PriceHistory:
        """Fetch historical prices for a market.

        Parameters
        ----------
        market_id : str
            Market condition ID or token ID.
        question : str
            Market question for labeling.
        interval : str
            Time range for history.
        fidelity : str
            Data granularity.

        Returns
        -------
        PriceHistory
            Structured price history with timestamps and prices.
        """
        raw = self.client.get_price_history(market_id, interval=interval, fidelity=fidelity)
        points: list[PricePoint] = []

        if isinstance(raw, list):
            for item in raw:
                ts = _parse_timestamp(item)
                price = _safe_float(item.get("p", item.get("price", 0)))
                if ts is not None and 0.0 <= price <= 1.0:
                    points.append(PricePoint(timestamp=ts, price=price))

        points.sort(key=lambda pp: pp.timestamp)
        logger.info(
            "Fetched {} price points for market {}",
            len(points), market_id[:20]
        )
        return PriceHistory(market_id=market_id, question=question, points=points)

    def price_history_to_dataframe(self, history: PriceHistory) -> pd.DataFrame:
        """Convert PriceHistory to a pandas DataFrame.

        Parameters
        ----------
        history : PriceHistory
            Price history to convert.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: timestamp, price. Indexed by timestamp.
        """
        if not history.points:
            return pd.DataFrame(columns=["timestamp", "price"])

        data = [{"timestamp": p.timestamp, "price": p.price} for p in history.points]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        return df


def _safe_float(value: Any) -> float:
    """Safely convert a value to float, returning 0.0 on failure."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def _parse_timestamp(item: dict[str, Any]) -> datetime | None:
    """Parse a timestamp from various API response formats."""
    for key in ("t", "timestamp", "time"):
        val = item.get(key)
        if val is None:
            continue
        if isinstance(val, (int, float)):
            try:
                return datetime.fromtimestamp(val, tz=timezone.utc)
            except (OSError, ValueError):
                # Try millisecond timestamp
                try:
                    return datetime.fromtimestamp(val / 1000, tz=timezone.utc)
                except (OSError, ValueError):
                    continue
        if isinstance(val, str):
            try:
                return datetime.fromisoformat(val.replace("Z", "+00:00"))
            except ValueError:
                continue
    return None


def _market_volume(market_info: MarketInfo, raw_markets: list[dict]) -> float:
    """Look up the volume for a MarketInfo from the raw API response."""
    for m in raw_markets:
        cid = m.get("conditionId", m.get("id", ""))
        if cid == market_info.market_id:
            return _safe_float(m.get("volume", 0))
    return 0.0
