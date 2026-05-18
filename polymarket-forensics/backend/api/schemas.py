"""Pydantic response models for the public API.

Using typed response models gives us free OpenAPI schema generation,
runtime validation of every payload we send, and an unambiguous contract
that the React frontend can rely on.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class WalletResponse(_Base):
    address: str
    first_seen: datetime | None
    last_active: datetime | None
    total_trades: int
    total_volume: Decimal
    total_pnl: Decimal
    win_rate: Decimal
    win_rate_p_value: Decimal | None = None
    total_resolved: int
    wins: int
    markets_traded: int
    avg_entry_price: Decimal | None = None
    avg_trade_size: Decimal | None = None
    avg_hours_before_resolution: Decimal | None = None
    top_market_volume: Decimal
    top_category_volume: Decimal
    political_military_volume: Decimal
    insider_score: Decimal
    smart_score: Decimal
    score_breakdown: dict[str, Any] | None = None
    cluster_id: str | None = None
    classification: str
    funding_source: str | None = None
    funding_exchange: str | None = None
    ens_name: str | None = None
    notes: str | None = None


class WalletListResponse(_Base):
    total: int
    limit: int
    offset: int
    wallets: list[WalletResponse]


class TradeResponse(_Base):
    id: int
    wallet_address: str | None = None
    market_id: str
    market_question: str | None = None
    market_category: str | None = None
    side: str
    outcome: str
    size: Decimal
    price: Decimal
    timestamp: datetime
    tx_hash: str | None = None
    is_large: bool
    hours_before_resolution: Decimal | None = None
    resolution_outcome: str | None = None
    trade_won: bool | None = None
    pnl: Decimal | None = None


class RecentTradeResponse(_Base):
    id: int
    wallet_address: str
    wallet_ens: str | None = None
    wallet_insider_score: Decimal
    wallet_classification: str
    market_id: str
    market_question: str | None = None
    market_category: str | None = None
    side: str
    outcome: str
    size: Decimal
    price: Decimal
    timestamp: datetime
    is_large: bool


class FundingHopResponse(_Base):
    source_address: str
    source_type: str | None = None
    source_exchange: str | None = None
    amount: Decimal | None = None
    asset: str | None = None
    timestamp: datetime | None = None
    tx_hash: str | None = None
    depth: int


class ClusterResponse(_Base):
    id: str
    wallets: list[str]
    cluster_type: str | None = None
    evidence: str | None = None
    total_pnl: Decimal | None = None
    combined_win_rate: Decimal | None = None
    markets_in_common: list[str] = Field(default_factory=list)
    insider_probability: Decimal | None = None
    first_detected: datetime | None = None
    last_updated: datetime | None = None
    status: str


class ClusterWalletSummary(_Base):
    address: str
    insider_score: Decimal
    total_pnl: Decimal
    total_volume: Decimal
    classification: str
    funding_exchange: str | None = None


class ClusterDetailResponse(ClusterResponse):
    wallet_details: list[ClusterWalletSummary] = Field(default_factory=list)


class ClusterGraphNode(_Base):
    id: str
    label: str
    insider_score: Decimal
    total_volume: Decimal
    cluster_id: str | None = None
    classification: str


class ClusterGraphEdge(_Base):
    source: str
    target: str
    cluster_id: str
    type: str | None = None
    weight: Decimal


class ClusterGraphResponse(_Base):
    nodes: list[ClusterGraphNode]
    edges: list[ClusterGraphEdge]


class AlertResponse(_Base):
    id: int
    alert_type: str
    severity: str
    title: str
    description: str
    wallet_address: str | None = None
    cluster_id: str | None = None
    market_id: str | None = None
    data: dict[str, Any] | None = None
    dismissed: bool
    created_at: datetime


class AlertListResponse(_Base):
    total: int
    limit: int
    offset: int
    alerts: list[AlertResponse]


class MarketResponse(_Base):
    id: str
    question: str
    category: str | None = None
    status: str | None = None
    resolution_outcome: str | None = None
    current_price: Decimal | None = None
    volume_total: Decimal | None = None
    resolved_at: datetime | None = None


class MarketForensicsWallet(_Base):
    address: str
    insider_score: Decimal
    classification: str
    funding_exchange: str | None = None
    volume: Decimal
    trades: int


class MarketForensicsResponse(_Base):
    market: MarketResponse
    dirty_volume: Decimal
    clean_volume: Decimal
    wallets: list[MarketForensicsWallet]


class StatsResponse(_Base):
    wallets_total: int
    trades_total: int
    clusters_total: int
    insider_suspects: int
    suspicious: int
    open_alerts: int
    critical_alerts: int
    volume_24h: Decimal
    large_trades_24h: int


class ServiceStatusResponse(_Base):
    name: str
    running: bool
    last_started: str | None = None


class SchedulerStatusResponse(_Base):
    services: list[ServiceStatusResponse]


class WalletPatch(_Base):
    notes: str | None = Field(None, max_length=2000)
    classification: str | None = None
    ens_name: str | None = Field(None, max_length=128)


class SearchResponse(_Base):
    wallets: list[WalletResponse]
    markets: list[MarketResponse]


class TraceResponse(_Base):
    queued: bool
    address: str


class DismissResponse(_Base):
    dismissed: bool
    id: int
