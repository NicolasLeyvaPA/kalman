from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    BigInteger, Boolean, DateTime, ForeignKey, Integer, Numeric, String, Text,
    ARRAY, func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Wallet(Base):
    __tablename__ = "wallets"

    address: Mapped[str] = mapped_column(Text, primary_key=True)
    first_seen: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_active: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    total_trades: Mapped[int] = mapped_column(Integer, default=0)
    total_volume: Mapped[Decimal] = mapped_column(Numeric, default=0)
    total_pnl: Mapped[Decimal] = mapped_column(Numeric, default=0)
    win_rate: Mapped[Decimal] = mapped_column(Numeric, default=0)
    win_rate_p_value: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    total_resolved: Mapped[int] = mapped_column(Integer, default=0)
    wins: Mapped[int] = mapped_column(Integer, default=0)
    markets_traded: Mapped[int] = mapped_column(Integer, default=0)
    avg_entry_price: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    avg_trade_size: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    avg_hours_before_resolution: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    top_market_volume: Mapped[Decimal] = mapped_column(Numeric, default=0)
    top_category_volume: Mapped[Decimal] = mapped_column(Numeric, default=0)
    political_military_volume: Mapped[Decimal] = mapped_column(Numeric, default=0)
    unique_protocols: Mapped[int] = mapped_column(Integer, default=0)
    total_tx_count: Mapped[int] = mapped_column(Integer, default=0)
    smart_score: Mapped[Decimal] = mapped_column(Numeric, default=0)
    insider_score: Mapped[Decimal] = mapped_column(Numeric, default=0)
    score_breakdown: Mapped[Optional[dict]] = mapped_column(JSONB)
    cluster_id: Mapped[Optional[str]] = mapped_column(Text)
    classification: Mapped[str] = mapped_column(Text, default="unknown")
    funding_source: Mapped[Optional[str]] = mapped_column(Text)
    funding_exchange: Mapped[Optional[str]] = mapped_column(Text)
    ens_name: Mapped[Optional[str]] = mapped_column(Text)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    wallet_address: Mapped[str] = mapped_column(Text, ForeignKey("wallets.address", ondelete="CASCADE"))
    market_id: Mapped[str] = mapped_column(Text, nullable=False)
    market_question: Mapped[Optional[str]] = mapped_column(Text)
    market_category: Mapped[Optional[str]] = mapped_column(Text)
    side: Mapped[str] = mapped_column(Text, nullable=False)
    outcome: Mapped[str] = mapped_column(Text, nullable=False)
    size: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    price: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    tx_hash: Mapped[Optional[str]] = mapped_column(Text)
    is_large: Mapped[bool] = mapped_column(Boolean, default=False)
    hours_before_resolution: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    resolution_outcome: Mapped[Optional[str]] = mapped_column(Text)
    trade_won: Mapped[Optional[bool]] = mapped_column(Boolean)
    pnl: Mapped[Optional[Decimal]] = mapped_column(Numeric)


class FundingChain(Base):
    __tablename__ = "funding_chains"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    wallet_address: Mapped[str] = mapped_column(Text, ForeignKey("wallets.address", ondelete="CASCADE"))
    source_address: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[Optional[str]] = mapped_column(Text)
    source_exchange: Mapped[Optional[str]] = mapped_column(Text)
    amount: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    asset: Mapped[Optional[str]] = mapped_column(Text)
    timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    tx_hash: Mapped[Optional[str]] = mapped_column(Text)
    depth: Mapped[int] = mapped_column(Integer, default=0)
    traced_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Cluster(Base):
    __tablename__ = "clusters"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    wallets: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    cluster_type: Mapped[Optional[str]] = mapped_column(Text)
    evidence: Mapped[Optional[str]] = mapped_column(Text)
    total_pnl: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    combined_win_rate: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    markets_in_common: Mapped[Optional[list[str]]] = mapped_column(ARRAY(Text))
    insider_probability: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    first_detected: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    status: Mapped[str] = mapped_column(Text, default="active")


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    alert_type: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    wallet_address: Mapped[Optional[str]] = mapped_column(Text)
    cluster_id: Mapped[Optional[str]] = mapped_column(Text)
    market_id: Mapped[Optional[str]] = mapped_column(Text)
    data: Mapped[Optional[dict]] = mapped_column(JSONB)
    dismissed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Market(Base):
    __tablename__ = "markets"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[Optional[str]] = mapped_column(Text)
    resolution_outcome: Mapped[Optional[str]] = mapped_column(Text)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    current_price: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    volume_total: Mapped[Optional[Decimal]] = mapped_column(Numeric)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class KnownAddress(Base):
    __tablename__ = "known_addresses"

    address: Mapped[str] = mapped_column(Text, primary_key=True)
    label: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(Text, nullable=False)
    chain: Mapped[str] = mapped_column(Text, default="polygon")
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
