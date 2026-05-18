# Polymarket Forensics Dashboard

A personal on-chain intelligence tool for detecting insider trading on Polymarket.

Monitors Polymarket wallets in real time, scores them for insider-trading probability,
traces funding chains on Polygon, clusters suspicious wallets, and alerts you when
statistically impossible trading patterns emerge.

## Stack

- **Backend**: FastAPI (Python 3.11+), SQLAlchemy, asyncio
- **Frontend**: React 18 + Vite + Tailwind CSS, d3-force for graph viz
- **Database**: PostgreSQL 15
- **Data**: Polymarket Gamma/CLOB/Data APIs, Alchemy Polygon RPC

## Quick Start

```bash
# 1. Copy and edit environment
cp .env.example .env
# Fill in ALCHEMY_API_KEY and DATABASE_URL

# 2. Bring up Postgres + backend + frontend
docker compose up -d

# 3. Initialize database
docker compose exec backend python -m scripts.init_db

# 4. Seed exchange address lists
docker compose exec backend python -m scripts.seed_exchanges

# 5. Open the dashboard
open http://localhost:5173
```

## Running Without Docker

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/forensics
export ALCHEMY_API_KEY=your_key_here
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              React Frontend (port 5173)                     │
│  LiveFeed / WalletExplorer / ClusterMap / MarketIntel /     │
│  Leaderboard / Settings                                     │
└──────────────────────────┬──────────────────────────────────┘
                           │ REST + WebSocket
┌──────────────────────────▼──────────────────────────────────┐
│              FastAPI Backend (port 8000)                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Services (background loops)                        │    │
│  │  - TradeIngester    (every 60s)                     │    │
│  │  - ScoringEngine    (every 5m)                      │    │
│  │  - ClusterDetector  (every 15m)                     │    │
│  │  - ChainTracer      (on-demand queue)               │    │
│  │  - ResolutionTracker (every 1h)                     │    │
│  └─────────────────────────────────────────────────────┘    │
└──────┬──────────────┬──────────────┬───────────────────────┘
       │              │              │
   ┌───▼───┐    ┌────▼────┐    ┌────▼─────┐
   │Alchemy│    │Polymarket│   │Postgres  │
   │Polygon│    │  APIs    │   │ + JSONB  │
   └───────┘    └──────────┘   └──────────┘
```

## Scoring

Each wallet receives an **insider score** (0.0 to 1.0) composed of 8 weighted sub-signals:

| Signal             | Weight | What it measures                                    |
|--------------------|--------|-----------------------------------------------------|
| Win-rate anomaly   | 0.20   | Binomial p-value against avg implied odds           |
| Timing             | 0.20   | Avg hours before market resolution                  |
| Freshness          | 0.15   | Age of wallet at time of first large trade          |
| Concentration      | 0.15   | % of volume in a single market / category           |
| Size vs odds       | 0.10   | Aggressive sizing at long odds                      |
| Single-purpose     | 0.08   | Wallet only touches CEX + Polymarket                |
| Sensitive markets  | 0.07   | % volume in political / military categories         |
| Cluster membership | 0.05   | Linked to other flagged wallets                     |

## Alert Types

- `FRESH_WHALE` — new wallet (< 7 days) places trade > $10K
- `IMPOSSIBLE_WIN_RATE` — win rate p < 0.001
- `PRE_EVENT_CLUSTER` — 3+ wallets bet same side within 2h
- `SINGLE_MARKET_ALL_IN` — 90%+ volume in one market
- `FUNDING_CHAIN_MATCH` — flagged wallets share funding source
- `SENSITIVE_MARKET_SURGE` — volume spike in political/military from new wallets
- `RESOLUTION_SNIPE` — trade < 6h pre-resolution at extreme odds
- `INSIDER_SCORE_SPIKE` — wallet score crosses 0.7

## Disclaimer

This is a **personal analysis tool**. Do not publish specific wallet addresses as
"insider" publicly without legal review. Do not trade on insider information you
detect. Use for research, monitoring, and defensive purposes only.
