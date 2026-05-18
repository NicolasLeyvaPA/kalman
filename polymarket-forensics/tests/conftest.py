"""Shared test fixtures."""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))


def utc(year: int = 2026, month: int = 5, day: int = 18, hour: int = 12) -> datetime:
    return datetime(year, month, day, hour, tzinfo=UTC)


@pytest.fixture
def clean_wallet():
    """A diversified, long-running, smart trader.

    The wallet was created well before its first trade and has hundreds of
    diversified positions with a modestly above-market win rate.
    """
    from scoring.types import WalletProfile
    return WalletProfile(
        address="0x" + "a" * 40,
        first_seen=utc() - timedelta(days=400),
        first_trade=utc() - timedelta(days=300),  # 100-day setup gap
        total_trades=200,
        total_volume=Decimal("250000"),
        total_pnl=Decimal("25000"),
        wins=80,
        total_resolved=120,
        avg_entry_price=Decimal("0.52"),
        avg_trade_size=Decimal("1250"),
        avg_pnl_per_trade=Decimal("125"),
        avg_hours_before_resolution=Decimal("250"),
        top_market_volume=Decimal("12000"),
        top_category_volume=Decimal("40000"),
        political_military_volume=Decimal("15000"),
        unique_protocols=12,
        total_tx_count=800,
        markets_traded=85,
        cluster_id=None,
        cluster_insider_prob=Decimal("0"),
    )


@pytest.fixture
def insider_wallet():
    """A textbook insider profile."""
    from scoring.types import WalletProfile
    return WalletProfile(
        address="0x" + "b" * 40,
        first_seen=utc() - timedelta(days=12),
        first_trade=utc() - timedelta(days=11, hours=12),
        total_trades=14,
        total_volume=Decimal("184000"),
        total_pnl=Decimal("127000"),
        wins=14,
        total_resolved=14,
        avg_entry_price=Decimal("0.18"),
        avg_trade_size=Decimal("13142.86"),
        avg_pnl_per_trade=Decimal("9071.43"),
        avg_hours_before_resolution=Decimal("4"),
        top_market_volume=Decimal("174800"),
        top_category_volume=Decimal("184000"),
        political_military_volume=Decimal("184000"),
        unique_protocols=2,
        total_tx_count=18,
        markets_traded=3,
        cluster_id="C-fun-deadbeef",
        cluster_insider_prob=Decimal("0.8"),
    )
