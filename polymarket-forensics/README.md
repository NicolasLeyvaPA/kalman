# Polymarket Forensics Dashboard

A fund-grade on-chain intelligence tool for detecting insider trading on
Polymarket. Monitors every Polymarket wallet in real time, scores them for
insider-trading probability against eight statistical sub-signals, traces
funding chains on Polygon, clusters suspicious wallets together, and pushes
critical alerts over a WebSocket feed.

## Stack

- **Backend**: FastAPI · SQLAlchemy async · asyncpg · Pydantic · structlog
- **Frontend**: React 18 · Vite · Tailwind · d3-force
- **Database**: PostgreSQL 15 (JSONB for score breakdowns)
- **Data sources**: Polymarket Gamma/CLOB/Data APIs · Alchemy Polygon RPC

## Quick Start

```bash
cp .env.example .env                # set ALCHEMY_API_KEY
docker compose up -d                # postgres + backend + frontend
docker compose exec backend python -m scripts.seed_exchanges
open http://localhost:5173
```

The Postgres container applies the schema on first boot from
`scripts/init_db.sql`. If you bring your own database, run
`python -m scripts.init_db` once.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                React Frontend  (port 5173)                          │
│   LiveFeed / WalletExplorer / ClusterMap / MarketIntel /            │
│   Leaderboard / Settings   ← custom hooks · skeleton + error states │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ REST + WebSocket (typed schemas)
┌──────────────────────────▼──────────────────────────────────────────┐
│                FastAPI Backend  (port 8000)                         │
│   Routes  →  Repositories  →  SQLAlchemy async                      │
│   ───────────────────────────────────────────────                   │
│   Services (scheduler-managed background loops)                     │
│     • TradeIngester      every 60s  · ingest + surge alerts         │
│     • ScoringEngine      every 5m   · 8-signal score + alerts       │
│     • ClusterDetector    every 15m  · 3-method merge + alerts       │
│     • ResolutionTracker  every 1h   · PnL with 2% fee model         │
│     • ChainTracerWorker  on-demand  · BFS funding chain trace       │
└──────┬──────────────┬──────────────┬───────────────────────────────┘
       │              │              │
   ┌───▼───┐    ┌────▼────┐    ┌────▼─────┐
   │Alchemy│    │Polymarket│   │Postgres  │
   │Polygon│    │  APIs    │   │ + JSONB  │
   └───────┘    └──────────┘   └──────────┘
```

## Insider scoring

Each wallet receives a composite insider score in `[0, 1]` built from eight
weighted sub-signals. Every threshold and weight lives in
`backend/scoring/types.ScoringConfig`. The composite is pure: same inputs
always yield the same output.

| Signal             | Weight | Captures                                            |
|--------------------|--------|-----------------------------------------------------|
| Win-rate anomaly   | 0.20   | Binomial p-value against avg implied odds           |
| Timing             | 0.20   | Avg hours before market resolution                  |
| Freshness          | 0.15   | Days between wallet creation and first trade        |
| Concentration      | 0.15   | % of volume in a single market / category           |
| Size vs odds       | 0.10   | Aggressive sizing at long odds                      |
| Single-purpose     | 0.08   | Wallet only touches CEX + Polymarket                |
| Sensitive markets  | 0.07   | % volume in political / military categories         |
| Cluster membership | 0.05   | Linked to other flagged wallets                     |

Coarse classification: `normal` (<0.30) · `watch` (≥0.30) · `suspicious`
(≥0.50) · `insider_suspect` (≥0.70).

## Alerts (all eight types are wired up)

| Type                       | Severity | Trigger                                                   |
|----------------------------|----------|-----------------------------------------------------------|
| `FRESH_WHALE`              | medium   | Wallet < 7 days old places a trade ≥ $10K                 |
| `IMPOSSIBLE_WIN_RATE`      | critical | Binomial p < 0.001 with ≥ 10 resolved trades              |
| `PRE_EVENT_CLUSTER`        | high     | 3+ wallets bet the same side within 2h on a market        |
| `SINGLE_MARKET_ALL_IN`     | high     | ≥ 90% of volume in one market across ≤ 2 markets traded   |
| `FUNDING_CHAIN_MATCH`      | critical | Flagged wallets share a funding source                    |
| `SENSITIVE_MARKET_SURGE`   | critical | Sudden volume spike in political/military markets         |
| `RESOLUTION_SNIPE`         | high     | Trade < 6h pre-resolution at price < 0.20                 |
| `INSIDER_SCORE_SPIKE`      | high     | Insider score crosses the 0.70 threshold                  |

Identical alerts within a 6-hour window are deduped by the alert generator.

## Project layout

```
polymarket-forensics/
├── README.md                   ← you are here
├── docker-compose.yml
├── pyproject.toml              ← ruff + mypy + pytest config
├── .pre-commit-config.yaml
├── .env.example
│
├── backend/
│   ├── main.py                 ← FastAPI app + correlation middleware
│   ├── config.py               ← Pydantic Settings (env-driven)
│   ├── enums.py                ← Classification / AlertType / Severity / …
│   ├── exceptions.py           ← typed exception hierarchy
│   ├── requirements.txt
│   │
│   ├── api/
│   │   ├── routes_wallets.py
│   │   ├── routes_clusters.py
│   │   ├── routes_alerts.py
│   │   ├── routes_markets.py
│   │   ├── routes_search.py
│   │   ├── routes_stats.py     ← overview + scheduler status
│   │   ├── routes_export.py    ← CSV / JSON exports
│   │   ├── schemas.py          ← Pydantic response models
│   │   └── websocket.py        ← broadcast hub + heartbeat
│   │
│   ├── repositories/           ← all SQL lives here, nowhere else
│   │   ├── wallets.py
│   │   └── alerts.py
│   │
│   ├── services/               ← background loops + alert generator
│   │   ├── scheduler.py
│   │   ├── trade_ingester.py
│   │   ├── scoring_engine.py
│   │   ├── cluster_detector.py
│   │   ├── chain_tracer.py
│   │   ├── resolution_tracker.py
│   │   └── alert_generator.py  ← single point of alert emission
│   │
│   ├── scoring/                ← pure functions, fully unit-tested
│   │   ├── types.py            ← WalletProfile / InsiderScoreResult / ScoringConfig
│   │   ├── insider_score.py
│   │   ├── smart_score.py
│   │   ├── statistical_tests.py
│   │   └── anomaly_detection.py
│   │
│   ├── chain/                  ← Polygon funding-chain tracer
│   ├── data/                   ← SQLAlchemy models + API clients
│   └── utils/                  ← logging + tz-aware time helpers
│
├── frontend/
│   └── src/
│       ├── App.jsx             ← error boundary + connection indicator
│       ├── pages/              ← 6 dashboard pages, all hook-driven
│       ├── components/         ← shared, presentational
│       ├── hooks/              ← useAsync, useWallet, useAlerts, useDebounced
│       ├── services/           ← typed API client + websocket w/ heartbeat
│       └── utils/
│
├── scripts/
│   ├── init_db.sql
│   ├── init_db.py
│   ├── seed_exchanges.py
│   └── backfill_trades.py
│
├── tests/                      ← 62 tests, scoring + cluster + utils
└── docs/
    └── AUDIT.md                ← record of the hardening pass
```

## Development

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend
cd frontend && npm install && npm run dev

# Tests + lint
python -m pytest tests/         # 62 tests pass
python -m ruff check backend tests
python -m mypy backend          # strict on pure modules
```

## Hardening pass

Every file in this codebase has been audited against fund-grade standards.
The audit findings and prioritized remediation list live in
[`docs/AUDIT.md`](docs/AUDIT.md). Key improvements over the initial spec:

- Typed exception hierarchy (`ForensicsError`, `PolymarketAPIError`,
  `RateLimitError`, …) replaces bare `except` blocks.
- Enums for every categorical field (`Classification`, `AlertType`,
  `Severity`, `Side`, `Outcome`, `ClusterType`, `SourceType`).
- `Decimal` propagates through scoring, monetary stats, and PnL.
  No `float` for money anywhere.
- `WalletProfile` / `InsiderScoreResult` / `ScoringConfig` dataclasses
  make scoring pure, deterministic, and trivially testable.
- All 8 alert types are wired up, with dedup within a 6-hour window via
  `services/alert_generator.emit()`.
- Repository pattern (`WalletRepository`, `AlertRepository`) cleanly
  separates SQL from HTTP handlers.
- Token-bucket rate limiter on the Polymarket client; `Retry-After`-aware
  rate-limit handling on Alchemy.
- WebSocket has a server-sent heartbeat; client reconnects with backoff
  and exposes connection status to the UI.
- React pages use `useAsync` / `useWallet` / `useAlerts` hooks; every
  page has skeleton loading states + retryable error states + an
  ErrorBoundary at the router level.
- d3-force graph is responsive via `ResizeObserver` and supports drag,
  zoom, pan, and node-click navigation.
- 62 unit tests cover scoring sub-signals, statistical primitives,
  cluster-merge transitivity, and timezone-aware time parsing.

## Disclaimer

This is a personal analysis tool. The insider score is a heuristic, not
proof. Do not publish wallet attributions without legal review. Trading
on insider information you detect may be illegal in your jurisdiction.
Use for research, monitoring, and defensive purposes only.
