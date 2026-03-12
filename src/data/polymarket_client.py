"""Low-level API client for Polymarket public endpoints.

Wraps the Gamma (market discovery), CLOB (pricing/order book), and Data
(historical) APIs with response caching and rate limiting.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import requests
from loguru import logger

# API base URLs
GAMMA_API_BASE: str = "https://gamma-api.polymarket.com"
CLOB_API_BASE: str = "https://clob.polymarket.com"
DATA_API_BASE: str = "https://data-api.polymarket.com"

# Default rate limit: minimum seconds between requests
DEFAULT_RATE_LIMIT_SECONDS: float = 1.0

# Default cache directory
DEFAULT_CACHE_DIR: str = "data/cache"

# Request timeout in seconds
REQUEST_TIMEOUT_SECONDS: int = 30


class PolymarketClient:
    """HTTP client for Polymarket public APIs with caching and rate limiting.

    Parameters
    ----------
    cache_dir : str
        Directory for cached API responses. Created if it doesn't exist.
    rate_limit : float
        Minimum seconds between consecutive API requests.
    use_cache : bool
        Whether to use cached responses. Set False for fresh data.

    Examples
    --------
    >>> client = PolymarketClient()
    >>> markets = client.get_markets(limit=5)
    """

    def __init__(
        self,
        cache_dir: str = DEFAULT_CACHE_DIR,
        rate_limit: float = DEFAULT_RATE_LIMIT_SECONDS,
        use_cache: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit
        self.use_cache = use_cache
        self._last_request_time: float = 0.0
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": "polymarket-kalman/0.1.0",
        })

    def _cache_key(self, url: str, params: dict[str, Any] | None) -> str:
        """Generate a deterministic cache key from URL and parameters."""
        key_str = url + json.dumps(params or {}, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _get_cached(self, cache_key: str) -> Any | None:
        """Load cached response if available."""
        cache_path = self.cache_dir / f"{cache_key}.json"
        if self.use_cache and cache_path.exists():
            logger.debug("Cache hit: {}", cache_key[:12])
            with open(cache_path) as f:
                return json.load(f)
        return None

    def _save_cache(self, cache_key: str, data: Any) -> None:
        """Save response to cache."""
        cache_path = self.cache_dir / f"{cache_key}.json"
        with open(cache_path, "w") as f:
            json.dump(data, f)

    def _rate_limit_wait(self) -> None:
        """Wait if needed to respect rate limit."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            wait_time = self.rate_limit - elapsed
            logger.debug("Rate limiting: waiting {:.2f}s", wait_time)
            time.sleep(wait_time)

    def _get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        """Execute GET request with caching and rate limiting.

        Parameters
        ----------
        url : str
            Full URL to request.
        params : dict or None
            Query parameters.

        Returns
        -------
        Any
            Parsed JSON response.

        Raises
        ------
        requests.HTTPError
            If the request fails with an HTTP error status.
        """
        cache_key = self._cache_key(url, params)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        self._rate_limit_wait()
        logger.debug("GET {} params={}", url, params)
        response = self._session.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        self._last_request_time = time.time()
        response.raise_for_status()
        data = response.json()
        self._save_cache(cache_key, data)
        return data

    # --- Gamma API (Market Discovery) ---

    def get_events(self, limit: int = 100, offset: int = 0,
                   active: bool = True) -> list[dict[str, Any]]:
        """Fetch prediction events from the Gamma API.

        Parameters
        ----------
        limit : int
            Maximum number of events to return.
        offset : int
            Pagination offset.
        active : bool
            If True, only return active (unresolved) events.

        Returns
        -------
        list[dict]
            List of event objects.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if active:
            params["active"] = "true"
        return self._get(f"{GAMMA_API_BASE}/events", params)

    def get_markets(self, limit: int = 100, offset: int = 0,
                    active: bool = True) -> list[dict[str, Any]]:
        """Fetch markets from the Gamma API.

        Parameters
        ----------
        limit : int
            Maximum number of markets to return.
        offset : int
            Pagination offset.
        active : bool
            If True, only return active markets.

        Returns
        -------
        list[dict]
            List of market objects with pricing and metadata.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if active:
            params["active"] = "true"
        return self._get(f"{GAMMA_API_BASE}/markets", params)

    def get_market(self, market_id: str) -> dict[str, Any]:
        """Fetch a single market by ID.

        Parameters
        ----------
        market_id : str
            The market's condition ID.

        Returns
        -------
        dict
            Market object with full metadata.
        """
        return self._get(f"{GAMMA_API_BASE}/markets/{market_id}")

    # --- CLOB API (Pricing / Order Book) ---

    def get_midpoint(self, token_id: str) -> dict[str, Any]:
        """Get the order book midpoint price for a token.

        Parameters
        ----------
        token_id : str
            CLOB token ID (YES outcome).

        Returns
        -------
        dict
            Midpoint price data.
        """
        return self._get(f"{CLOB_API_BASE}/midpoint", {"token_id": token_id})

    def get_price(self, token_id: str, side: str = "BUY") -> dict[str, Any]:
        """Get the best price for a side (BUY or SELL).

        Parameters
        ----------
        token_id : str
            CLOB token ID.
        side : str
            Order side: "BUY" or "SELL".

        Returns
        -------
        dict
            Price data for the requested side.
        """
        return self._get(f"{CLOB_API_BASE}/price", {"token_id": token_id, "side": side})

    def get_spread(self, token_id: str) -> dict[str, Any]:
        """Get the bid-ask spread for a token.

        Parameters
        ----------
        token_id : str
            CLOB token ID.

        Returns
        -------
        dict
            Spread data including best bid, best ask, and spread.
        """
        return self._get(f"{CLOB_API_BASE}/spread", {"token_id": token_id})

    def get_book(self, token_id: str) -> dict[str, Any]:
        """Get the full order book for a token.

        Parameters
        ----------
        token_id : str
            CLOB token ID.

        Returns
        -------
        dict
            Order book with bids and asks arrays.
        """
        return self._get(f"{CLOB_API_BASE}/book", {"token_id": token_id})

    # --- Data API (Historical) ---

    def get_price_history(self, market_id: str, interval: str = "max",
                          fidelity: str = "1h") -> list[dict[str, Any]]:
        """Fetch historical price data for a market.

        Parameters
        ----------
        market_id : str
            Market condition ID or token ID.
        interval : str
            Time range: "1d", "1w", "1m", "3m", "max".
        fidelity : str
            Data granularity: "1m", "5m", "1h", "1d".

        Returns
        -------
        list[dict]
            List of price history points with timestamp and price.
        """
        params = {"market": market_id, "interval": interval, "fidelity": fidelity}
        return self._get(f"{DATA_API_BASE}/prices-history", params)

    def get_trades(self, market_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """Fetch recent trades for a market.

        Parameters
        ----------
        market_id : str
            Market condition ID.
        limit : int
            Maximum number of trades to return.

        Returns
        -------
        list[dict]
            List of trade objects.
        """
        return self._get(f"{DATA_API_BASE}/trades", {"market": market_id, "limit": limit})

    def get_activity(self, market_id: str) -> list[dict[str, Any]]:
        """Fetch recent activity for a market.

        Parameters
        ----------
        market_id : str
            Market condition ID.

        Returns
        -------
        list[dict]
            List of activity events.
        """
        return self._get(f"{DATA_API_BASE}/activity", {"market": market_id})
