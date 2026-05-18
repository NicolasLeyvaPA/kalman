"""Canonical enumerations for categorical fields.

These values are persisted to the database as text. Changing a value here
is a schema migration — bump the value and write a SQL migration that
updates existing rows.
"""

from __future__ import annotations

from enum import Enum


class Classification(str, Enum):
    """Coarse-grained wallet classification derived from the insider score."""

    UNKNOWN = "unknown"
    NORMAL = "normal"
    WATCH = "watch"
    SUSPICIOUS = "suspicious"
    INSIDER_SUSPECT = "insider_suspect"
    CONFIRMED_INSIDER = "confirmed_insider"
    CLEARED = "cleared"
    SMART = "smart"


class Severity(str, Enum):
    """Alert severity. Critical pings the on-call channel immediately."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Every alert generator writes exactly one of these values."""

    FRESH_WHALE = "FRESH_WHALE"
    IMPOSSIBLE_WIN_RATE = "IMPOSSIBLE_WIN_RATE"
    PRE_EVENT_CLUSTER = "PRE_EVENT_CLUSTER"
    SINGLE_MARKET_ALL_IN = "SINGLE_MARKET_ALL_IN"
    FUNDING_CHAIN_MATCH = "FUNDING_CHAIN_MATCH"
    SENSITIVE_MARKET_SURGE = "SENSITIVE_MARKET_SURGE"
    RESOLUTION_SNIPE = "RESOLUTION_SNIPE"
    INSIDER_SCORE_SPIKE = "INSIDER_SCORE_SPIKE"


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class Outcome(str, Enum):
    YES = "YES"
    NO = "NO"


class ClusterType(str, Enum):
    FUNDING_LINKED = "funding_linked"
    TEMPORAL = "temporal"
    BEHAVIORAL = "behavioral"
    MIXED = "mixed"


class ClusterStatus(str, Enum):
    ACTIVE = "active"
    CONFIRMED = "confirmed"
    CLEARED = "cleared"
    WATCHING = "watching"


class SourceType(str, Enum):
    """Funding-chain source classification."""

    EXCHANGE = "exchange"
    BRIDGE = "bridge"
    CONTRACT = "contract"
    WALLET = "wallet"
    UNKNOWN = "unknown"
