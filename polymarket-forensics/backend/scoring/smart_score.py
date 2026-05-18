"""
Smart-money score: profile of a legitimately skilled trader.

Distinct from insider_score: a smart trader has good win rate AND high
diversification across many markets AND a long operating history.
An insider has good win rate AND low diversification AND a short history.
"""
from __future__ import annotations

import math

from scoring.insider_score import WalletProfile


def compute_smart_score(p: WalletProfile) -> float:
    score = 0.0

    if p.total_resolved >= 50:
        score += 0.25
    elif p.total_resolved >= 20:
        score += 0.15

    if p.markets_traded >= 30:
        score += 0.20
    elif p.markets_traded >= 10:
        score += 0.10

    if p.total_resolved > 0:
        wr = p.wins / p.total_resolved
        if wr > 0.65:
            score += 0.20
        elif wr > 0.55:
            score += 0.10

    if p.total_pnl > 0 and p.total_volume > 0:
        roi = p.total_pnl / p.total_volume
        if roi > 0.10:
            score += min(0.20, math.log10(1 + roi * 10) * 0.20)

    if p.first_seen and p.first_trade:
        days = (p.first_trade - p.first_seen).total_seconds() / 86400.0
        if days > 90:
            score += 0.15

    return round(min(score, 1.0), 4)
