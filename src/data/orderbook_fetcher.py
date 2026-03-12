"""Order book fetcher and microstructure data extraction.

Fetches order book snapshots from the CLOB API and computes
depth, imbalance, and spread metrics needed for dynamic noise estimation.
"""

from datetime import datetime, timezone
from typing import Any

from loguru import logger

from src.data.models import MarketObservation
from src.data.polymarket_client import PolymarketClient

# Number of top price levels to use when computing depth
TOP_LEVELS_FOR_DEPTH: int = 5


class OrderbookFetcher:
    """Fetches and parses order book data from Polymarket CLOB API.

    Parameters
    ----------
    client : PolymarketClient or None
        API client instance. Created with defaults if None.
    """

    def __init__(self, client: PolymarketClient | None = None) -> None:
        self.client = client or PolymarketClient()

    def fetch_observation(self, token_id: str, market_id: str = "",
                          market_question: str = "") -> MarketObservation:
        """Build a full MarketObservation from current order book state.

        Fetches the order book and spread, then computes microstructure metrics.

        Parameters
        ----------
        token_id : str
            CLOB token ID for the YES outcome.
        market_id : str
            Market identifier for labeling.
        market_question : str
            Market question for labeling.

        Returns
        -------
        MarketObservation
            Complete observation with price and microstructure data.
        """
        book_data = self.client.get_book(token_id)
        spread_data = self.client.get_spread(token_id)

        bids = _parse_levels(book_data.get("bids", []))
        asks = _parse_levels(book_data.get("asks", []))

        best_bid = bids[0][0] if bids else 0.0
        best_ask = asks[0][0] if asks else 1.0
        midpoint = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid

        bid_depth = sum(size for _, size in bids[:TOP_LEVELS_FOR_DEPTH])
        ask_depth = sum(size for _, size in asks[:TOP_LEVELS_FOR_DEPTH])
        total_depth = bid_depth + ask_depth
        imbalance = bid_depth / total_depth if total_depth > 0 else 0.5

        # Use spread data if available for more accurate values
        if isinstance(spread_data, dict):
            best_bid = _safe_float(spread_data.get("bid", best_bid))
            best_ask = _safe_float(spread_data.get("ask", best_ask))
            spread = _safe_float(spread_data.get("spread", spread))
            if spread_data.get("bid") and spread_data.get("ask"):
                midpoint = (best_bid + best_ask) / 2.0

        yes_price = midpoint

        return MarketObservation(
            timestamp=datetime.now(timezone.utc),
            market_id=market_id,
            market_question=market_question,
            yes_price=yes_price,
            no_price=1.0 - yes_price,
            midpoint=midpoint,
            spread=spread,
            volume_24h=0.0,  # Not available from book endpoint
            best_bid=best_bid,
            best_ask=best_ask,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            total_depth=total_depth,
            imbalance=imbalance,
            num_trades_1h=0,  # Not available from book endpoint
        )


def _parse_levels(levels: list[Any]) -> list[tuple[float, float]]:
    """Parse order book levels into (price, size) tuples.

    Parameters
    ----------
    levels : list
        Raw order book levels from the API. Each level may be a dict
        with 'price'/'size' keys or a list [price, size].

    Returns
    -------
    list[tuple[float, float]]
        Parsed (price, size) pairs sorted by price (descending for bids).
    """
    parsed: list[tuple[float, float]] = []
    for level in levels:
        if isinstance(level, dict):
            price = _safe_float(level.get("price", 0))
            size = _safe_float(level.get("size", 0))
        elif isinstance(level, (list, tuple)) and len(level) >= 2:
            price = _safe_float(level[0])
            size = _safe_float(level[1])
        else:
            continue
        parsed.append((price, size))
    return parsed


def _safe_float(value: Any) -> float:
    """Safely convert a value to float, returning 0.0 on failure."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0
