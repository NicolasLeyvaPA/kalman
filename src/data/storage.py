"""SQLite persistence layer for market data.

Stores market metadata, observations, and price history in a local SQLite
database. Supports querying by market ID and time range, and exporting to CSV.
"""

import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from src.data.models import MarketInfo, MarketObservation, PriceHistory, PricePoint

# Default database path
DEFAULT_DB_PATH: str = "data/markets.db"


class MarketStorage:
    """SQLite storage for market data.

    Creates the database and tables on initialization if they don't exist.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS markets (
                market_id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                token_id TEXT,
                slug TEXT,
                category TEXT,
                end_date TEXT,
                active INTEGER DEFAULT 1,
                description TEXT
            );

            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                market_id TEXT NOT NULL,
                market_question TEXT,
                yes_price REAL,
                no_price REAL,
                midpoint REAL,
                spread REAL,
                volume_24h REAL,
                best_bid REAL,
                best_ask REAL,
                bid_depth REAL,
                ask_depth REAL,
                total_depth REAL,
                imbalance REAL,
                num_trades_1h INTEGER,
                FOREIGN KEY (market_id) REFERENCES markets(market_id)
            );

            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                price REAL NOT NULL,
                FOREIGN KEY (market_id) REFERENCES markets(market_id)
            );

            CREATE INDEX IF NOT EXISTS idx_obs_market_time
                ON observations(market_id, timestamp);

            CREATE INDEX IF NOT EXISTS idx_price_market_time
                ON price_history(market_id, timestamp);
        """)
        self._conn.commit()

    def save_market(self, market: MarketInfo) -> None:
        """Insert or update market metadata.

        Parameters
        ----------
        market : MarketInfo
            Market metadata to save.
        """
        end_date_str = market.end_date.isoformat() if market.end_date else None
        self._conn.execute(
            """INSERT OR REPLACE INTO markets
               (market_id, question, token_id, slug, category, end_date, active, description)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (market.market_id, market.question, market.token_id, market.slug,
             market.category, end_date_str, int(market.active), market.description),
        )
        self._conn.commit()

    def save_observation(self, obs: MarketObservation) -> None:
        """Insert a single market observation.

        Parameters
        ----------
        obs : MarketObservation
            Observation to save.
        """
        self._conn.execute(
            """INSERT INTO observations
               (timestamp, market_id, market_question, yes_price, no_price,
                midpoint, spread, volume_24h, best_bid, best_ask,
                bid_depth, ask_depth, total_depth, imbalance, num_trades_1h)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (obs.timestamp.isoformat(), obs.market_id, obs.market_question,
             obs.yes_price, obs.no_price, obs.midpoint, obs.spread,
             obs.volume_24h, obs.best_bid, obs.best_ask, obs.bid_depth,
             obs.ask_depth, obs.total_depth, obs.imbalance, obs.num_trades_1h),
        )
        self._conn.commit()

    def save_price_history(self, history: PriceHistory) -> None:
        """Insert price history points for a market.

        Parameters
        ----------
        history : PriceHistory
            Price history to save. Existing data for this market is cleared first.
        """
        self._conn.execute(
            "DELETE FROM price_history WHERE market_id = ?", (history.market_id,)
        )
        for point in history.points:
            self._conn.execute(
                "INSERT INTO price_history (market_id, timestamp, price) VALUES (?, ?, ?)",
                (history.market_id, point.timestamp.isoformat(), point.price),
            )
        self._conn.commit()
        logger.info(
            "Saved {} price points for market {}",
            len(history.points), history.market_id[:20]
        )

    def get_observations(self, market_id: str,
                         start: datetime | None = None,
                         end: datetime | None = None) -> list[MarketObservation]:
        """Query observations for a market within a time range.

        Parameters
        ----------
        market_id : str
            Market to query.
        start : datetime or None
            Start of time range (inclusive). None means no lower bound.
        end : datetime or None
            End of time range (inclusive). None means no upper bound.

        Returns
        -------
        list[MarketObservation]
            Observations sorted by timestamp ascending.
        """
        query = "SELECT * FROM observations WHERE market_id = ?"
        params: list = [market_id]

        if start:
            query += " AND timestamp >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND timestamp <= ?"
            params.append(end.isoformat())

        query += " ORDER BY timestamp ASC"
        rows = self._conn.execute(query, params).fetchall()
        return [_row_to_observation(row) for row in rows]

    def get_price_history(self, market_id: str) -> PriceHistory:
        """Load price history for a market.

        Parameters
        ----------
        market_id : str
            Market to query.

        Returns
        -------
        PriceHistory
            Price history with all stored points.
        """
        rows = self._conn.execute(
            "SELECT timestamp, price FROM price_history WHERE market_id = ? ORDER BY timestamp",
            (market_id,),
        ).fetchall()

        # Get question from markets table
        market_row = self._conn.execute(
            "SELECT question FROM markets WHERE market_id = ?", (market_id,)
        ).fetchone()
        question = market_row["question"] if market_row else ""

        points = [
            PricePoint(
                timestamp=datetime.fromisoformat(row["timestamp"]),
                price=row["price"],
            )
            for row in rows
        ]
        return PriceHistory(market_id=market_id, question=question, points=points)

    def export_to_csv(self, market_id: str, output_path: str) -> None:
        """Export price history for a market to CSV.

        Parameters
        ----------
        market_id : str
            Market to export.
        output_path : str
            Path for the output CSV file.
        """
        history = self.get_price_history(market_id)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "price"])
            for point in history.points:
                writer.writerow([point.timestamp.isoformat(), point.price])

        logger.info("Exported {} points to {}", len(history.points), output_path)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


def _row_to_observation(row: sqlite3.Row) -> MarketObservation:
    """Convert a database row to a MarketObservation."""
    return MarketObservation(
        timestamp=datetime.fromisoformat(row["timestamp"]),
        market_id=row["market_id"],
        market_question=row["market_question"] or "",
        yes_price=row["yes_price"] or 0.0,
        no_price=row["no_price"] or 0.0,
        midpoint=row["midpoint"] or 0.0,
        spread=row["spread"] or 0.0,
        volume_24h=row["volume_24h"] or 0.0,
        best_bid=row["best_bid"] or 0.0,
        best_ask=row["best_ask"] or 0.0,
        bid_depth=row["bid_depth"] or 0.0,
        ask_depth=row["ask_depth"] or 0.0,
        total_depth=row["total_depth"] or 0.0,
        imbalance=row["imbalance"] or 0.5,
        num_trades_1h=row["num_trades_1h"] or 0,
    )
