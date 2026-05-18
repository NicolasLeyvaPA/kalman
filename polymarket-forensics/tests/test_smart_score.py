"""Tests for the smart-money score."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

from scoring.smart_score import compute_smart_score


def test_diversified_long_history_wallet_scores_high(clean_wallet):
    result = compute_smart_score(clean_wallet)
    assert result.composite >= Decimal("0.50")


def test_insider_wallet_scores_low_on_smart(insider_wallet):
    # Insider profile: short history, concentrated, few markets — not "smart".
    result = compute_smart_score(insider_wallet)
    assert result.composite < Decimal("0.55"), \
        f"insider should not score smart, got {result.composite}"


def test_bounded(clean_wallet):
    result = compute_smart_score(clean_wallet)
    assert Decimal("0") <= result.composite <= Decimal("1")


def test_zero_trades_doesnt_crash(clean_wallet):
    w = replace(
        clean_wallet, total_trades=0, total_resolved=0, wins=0,
        markets_traded=0, total_volume=Decimal("0"), total_pnl=Decimal("0"),
    )
    result = compute_smart_score(w)
    assert result.composite >= Decimal("0")
