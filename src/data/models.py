"""Data models for market observations and metadata.

Defines the core dataclasses used throughout the system for representing
market state, order book snapshots, and price history.
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MarketInfo:
    """Static metadata about a Polymarket prediction market.

    Parameters
    ----------
    market_id : str
        Unique identifier for the market (condition ID or slug).
    question : str
        The question the market is predicting (e.g., "Will BTC exceed $100K?").
    token_id : str
        CLOB token ID for the YES outcome, used for order book queries.
    slug : str
        URL-friendly market identifier.
    category : str
        Market category (e.g., "politics", "crypto", "sports").
    end_date : datetime or None
        When the market is scheduled to resolve.
    active : bool
        Whether the market is currently accepting trades.
    description : str
        Detailed description of the resolution criteria.
    """

    market_id: str
    question: str
    token_id: str
    slug: str = ""
    category: str = ""
    end_date: datetime | None = None
    active: bool = True
    description: str = ""


@dataclass
class MarketObservation:
    """A single observation of market state at a point in time.

    Captures both price and microstructure data needed for the Kalman filter.
    Price fields are probabilities in [0, 1]. Dollar amounts are in USD.

    Parameters
    ----------
    timestamp : datetime
        When this observation was recorded.
    market_id : str
        Identifier of the market.
    market_question : str
        Human-readable market question.
    yes_price : float
        Last traded YES price, in [0, 1].
    no_price : float
        Last traded NO price, in [0, 1].
    midpoint : float
        Order book midpoint: (best_bid + best_ask) / 2.
    spread : float
        Order book spread: best_ask - best_bid.
    volume_24h : float
        Trading volume over the last 24 hours in USD.
    best_bid : float
        Highest buy order price.
    best_ask : float
        Lowest sell order price.
    bid_depth : float
        Total USD on the bid side (top 5 levels).
    ask_depth : float
        Total USD on the ask side (top 5 levels).
    total_depth : float
        bid_depth + ask_depth.
    imbalance : float
        Order book imbalance: bid_depth / total_depth. 0.5 = balanced.
    num_trades_1h : int
        Number of trades in the last hour.
    """

    timestamp: datetime
    market_id: str
    market_question: str
    yes_price: float
    no_price: float = 0.0
    midpoint: float = 0.0
    spread: float = 0.0
    volume_24h: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    total_depth: float = 0.0
    imbalance: float = 0.5
    num_trades_1h: int = 0


@dataclass
class PricePoint:
    """A single historical price data point.

    Parameters
    ----------
    timestamp : datetime
        When this price was recorded.
    price : float
        The price at this time, in [0, 1].
    """

    timestamp: datetime
    price: float


@dataclass
class PriceHistory:
    """Historical price series for a market.

    Parameters
    ----------
    market_id : str
        Identifier of the market.
    question : str
        Human-readable market question.
    points : list[PricePoint]
        Time-ordered list of price observations.
    """

    market_id: str
    question: str
    points: list[PricePoint] = field(default_factory=list)
