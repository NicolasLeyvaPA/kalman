# Audit Findings — Polymarket Forensics Dashboard

Systematic review of every file in the codebase against fund-grade standards.
This document is the input to the hardening pass that follows.

## Backend Audit

### `backend/config.py`
- ✗ `political_categories` is a mutable list default on a Pydantic model. Should be a `tuple` or use `Field(default_factory=...)`.
- ✗ No validation on `large_trade_usd`, `insider_trace_threshold`, `*_interval` values (could be negative).
- ✗ `ALCHEMY_API_KEY` allowed to be empty string. Should be `Optional[SecretStr]`.
- ✗ Magic strings duplicated for service base URLs (gamma/data/clob) — fine, but could be grouped in a sub-model.

### `backend/data/database.py`
- ✗ `get_db` async generator is not properly typed as `AsyncGenerator[AsyncSession, None]`.
- ✗ `engine`, `AsyncSessionLocal` are module-level globals. Not dependency-injectable, hard to mock.
- ✗ `db_session` context manager doesn't log query errors.
- ✗ Module imports `get_settings()` at import time — couples DB module to environment.

### `backend/data/models.py`
- ✓ Types good, `Decimal` used for monetary fields.
- ✗ `Wallet.classification` is `str`, should be `Enum`.
- ✗ `Trade.side` and `Trade.outcome` are `str`, should be `Enum`.
- ✗ `Alert.severity` is `str`, should be `Enum`.
- ✗ `Cluster.cluster_type` is `str`, should be `Enum`.
- ✗ No `__repr__` overrides — debugging is painful.

### `backend/data/polymarket_client.py`
- ✗ Bare `except httpx.HTTPStatusError: return []` swallows real errors silently. Caller can't distinguish "no data" from "API broken".
- ✗ No custom exception types — caller has no way to handle 429 vs 500 vs timeout differently.
- ✗ `_singleton` global + asyncio lock pattern. Fine, but not DI-friendly.
- ✗ No request-level logging. Hard to diagnose API issues.
- ✗ Hardcoded `User-Agent`, no API client version.
- ✗ No rate-limit handling — Polymarket data API enforces ~5 rps; we make 80+ rapid calls per ingest pass.
- ✗ No correlation IDs propagated.

### `backend/data/known_exchanges.py`
- ✓ Curated list is reasonable.
- ✗ Hot wallets rotate — no mechanism to refresh from external source.
- ✗ `lookup_address` doesn't lowercase the input (caller must). Inconsistent with other modules.

### `backend/chain/alchemy_client.py`
- ✗ `RuntimeError` for missing API key — should be a custom exception.
- ✗ `RuntimeError` for RPC errors — caller can't distinguish "rate limited" from "method not supported".
- ✗ No metering of compute units consumed.
- ✗ `get_asset_transfers` builds Alchemy params with `hex(max_count)` — fine, but undocumented.
- ✗ Same singleton pattern problem as Polymarket client.

### `backend/chain/funding_tracer.py`
- ✓ Clear BFS algorithm.
- ✗ `FundingHop.amount` is `float`. Monetary values must be `Decimal`.
- ✗ Recursion in async via inner function — works but harder to test than a stack-based loop.
- ✗ Silently catches `Exception` on `get_asset_transfers` — masks real errors.
- ✗ `_parse_ts` catches `Exception`, returns `None`. Should at least log.

### `backend/scoring/insider_score.py`
- ✗ Returns a dict, not a typed `InsiderScoreResult` dataclass.
- ✗ `WalletProfile` uses `float` for monetary values. Should be `Decimal`.
- ✗ Hardcoded thresholds (`0.90`, `0.85`, `1.0`, `6`, `24`, `72`, `3.0`, `1.5`) — should live in a `ScoringConfig`.
- ✗ Weights live as module constants — should be in `ScoringConfig` so they can be tuned without editing code.
- ✗ No `classification_from_score` uses a string return — should return `Classification` enum.
- ✗ Division-by-zero guards exist for `total_volume` but not for `avg_trade_size` in some branches.

### `backend/scoring/smart_score.py`
- ✗ Same `float` vs `Decimal` issue.
- ✗ Magic numbers everywhere (50, 30, 20, 10, 0.65, 0.55, 90).

### `backend/scoring/statistical_tests.py`
- ✓ Binomial p-value implementation is correct.
- ✗ No type for `volume_anomaly_z_score` arguments.
- ✗ `volume_anomaly_z_score` is unused — dead code.

### `backend/scoring/anomaly_detection.py`
- ✗ `detect_volume_spike` iterates a `dict` of trades from db rows or API responses — type is unclear.
- ✗ Naive datetime arithmetic in places; `datetime.now().astimezone()` is fine but inconsistent with other modules using `datetime.now(tz=timezone.utc)`.
- ✗ Not wired into the scoring engine — no alerts ever fire from this.

### `backend/services/trade_ingester.py`
- ✗ Iterates markets sequentially with one DB session per market — should batch.
- ✗ Limits to first 80 markets — silently drops the long tail.
- ✗ No tracking of "last ingested timestamp" — re-fetches everything every cycle (relies on UPSERT to dedupe).
- ✗ No retry on Polymarket failure — drops the whole cycle if one market 500s.
- ✗ `_to_dt` duplicated across services — DRY violation.
- ✗ Doesn't trigger chain trace for large trades from new wallets (spec requires it).
- ✗ Doesn't fire `FRESH_WHALE` alert for new wallet + large trade.

### `backend/services/scoring_engine.py`
- ✗ N+1 queries: builds one `WalletProfile` per address with separate queries.
- ✗ Recently-active window is 10 minutes but scoring runs every 5 minutes — overlap waste.
- ✗ Doesn't compute `markets_in_common` aggregations efficiently.
- ✗ Missing alert types: `FRESH_WHALE`, `RESOLUTION_SNIPE`, `SENSITIVE_MARKET_SURGE`.
- ✗ `_is_political` linear scan of categories on every trade — should be a precomputed set.

### `backend/services/cluster_detector.py`
- ✗ Loads ALL `FundingChain` rows into memory on every cycle. Won't scale past ~1M rows.
- ✗ `detect_temporal_clusters` loads 7 days of trades; should be a SQL window query.
- ✗ Behavioral cosine similarity is O(n²). With 10k flagged wallets that's 100M comparisons.
- ✗ Cluster ID generation collides on the same member set across types — uses `prefix` to differentiate but only first 3 chars.
- ✗ Alert title for funding cluster uses raw count, doesn't tell you the exchange.

### `backend/services/chain_tracer.py`
- ✗ `_seen_recent` set unbounded growth controlled by length check; better: TTL cache.
- ✗ No worker pool — one trace at a time. Slow when score spikes batch many wallets.
- ✗ Doesn't trigger cluster re-analysis after new funding data (spec requires it).

### `backend/services/resolution_tracker.py`
- ✗ Naive PnL calculation: `size * (1 - price) / price` ignores Polymarket's fee structure (~2% per round trip).
- ✗ Doesn't recompute wallet aggregates after attributing trades — depends on scoring engine to catch up.
- ✗ Doesn't fire alerts for "too good" resolutions.

### `backend/services/scheduler.py`
- ✓ Simple, works.
- ✗ No backpressure or service health introspection — can't tell which service is wedged.
- ✗ No exposing of service status to the dashboard.

### `backend/api/routes_*.py`
- ✗ Business logic mixed with route handlers (e.g., `_serialize_wallet` in `routes_wallets.py`).
- ✗ Response models are untyped dicts. Should be Pydantic response models for OpenAPI schema generation.
- ✗ `PATCH /wallets/{address}` accepts a raw `dict` with no validation — XSS-risk for `notes` field.
- ✗ `GET /alerts` has no pagination — could return arbitrarily large response.
- ✗ Missing endpoints: CSV/JSON export, scheduler status, known-address management.

### `backend/api/websocket.py`
- ✓ Clean implementation.
- ✗ No backpressure if a client falls behind.
- ✗ No auth / origin check on connections.
- ✗ No ping/pong heartbeat — connections die silently.

### `backend/main.py`
- ✗ CORS allows three hardcoded localhost origins — should come from config.
- ✗ No `/metrics` endpoint for observability.
- ✗ No request-correlation middleware.

### Cross-cutting issues
- ✗ No `pyproject.toml` with `ruff`/`mypy`/`pytest` config.
- ✗ No tests at all in the `tests/` directory.
- ✗ Logging is unstructured (`logging` module). Should be `structlog`.
- ✗ No exception hierarchy — every module raises generic `RuntimeError` or returns `None`.
- ✗ No correlation IDs across async tasks.
- ✗ `_to_dt` helper duplicated in three files.

## Frontend Audit

### `src/App.jsx`
- ✓ Clean routing.
- ✗ Inline `SearchBar` component should be extracted to `components/SearchBar.jsx`.
- ✗ `useEffect` for search debounce manages timer inline — should be a `useDebouncedValue` hook.

### `src/services/api.js`
- ✗ Plain object of functions — no error class, no retry, no abort signal support.
- ✗ Errors thrown are plain `Error` with stringified body — caller can't programmatically handle 429 vs 404.

### `src/services/websocket.js`
- ✓ Reconnect-with-backoff is solid.
- ✗ No heartbeat ping.
- ✗ No connection-status surfacing to UI (e.g., "connection lost, reconnecting…").

### `src/pages/*.jsx`
- ✗ Every page does its own `useEffect`/`useState` data fetching. Should be custom hooks (`useWallet`, `useAlerts`, `useStats`).
- ✗ No skeleton loaders — pages show "Loading…" or blank.
- ✗ Error states are minimal — show raw error.message only.
- ✗ `LiveFeed.jsx` polls every 15s AND has WebSocket — redundant; should use WS as source of truth and only poll on reconnect.
- ✗ `WalletExplorer.jsx`: trace button waits 4 seconds then refreshes — should subscribe to trace-complete event.
- ✗ Concentration "Top market %" computed in the component — should come from API.

### `src/components/*.jsx`
- ✗ `BubbleGraph.jsx` re-creates the entire simulation on every prop change instead of incrementally updating.
- ✗ `BubbleGraph.jsx` not responsive — fixed width/height.
- ✗ No `MarketDistribution.jsx` component (spec calls for one).
- ✗ No `TradeTimeline.jsx` component (spec calls for one).
- ✗ `WalletScoreBreakdown.jsx` has signal labels hardcoded — should map from a shared config.

### Cross-cutting frontend issues
- ✗ No design-token CSS variables exposed beyond Tailwind — d3 code uses string literals for colors.
- ✗ No `prop-types` validation, no JSDoc on most components.
- ✗ No error boundary at the router level.
- ✗ No favicon, no loading sequence for initial render.

## Tests Audit
- ✗ No tests exist. Coverage is 0%.
- ✗ Scoring functions have NO regression protection — any change risks silent breakage.
- ✗ Cluster detection algorithms (the most complex code in the system) are untested.

## Prioritized Hardening Plan

**P0 — must ship now (correctness, safety):**
1. Custom exception hierarchy (`backend/exceptions.py`).
2. Enums for categorical values (Classification, AlertType, Severity, Side, Outcome, ClusterType).
3. `Decimal` propagation through `WalletProfile`, `FundingHop`, scoring result types.
4. Typed `InsiderScoreResult` / `SmartScoreResult` dataclasses (no more raw dicts from scoring).
5. Missing alert generators: `FRESH_WHALE`, `RESOLUTION_SNIPE`, `SENSITIVE_MARKET_SURGE`.
6. Pagination on `/alerts` and `/wallets`.
7. Pydantic response models on all routes.
8. Repository pattern for wallet / alert / cluster data access.
9. Structured logging via `structlog`.
10. Rate-limit aware Polymarket client with explicit retry-after handling.

**P1 — strongly recommended (quality, scale):**
11. Pytest suite covering scoring (>90%), cluster detection, statistical tests.
12. `pyproject.toml` with ruff + mypy + pytest config; pre-commit hooks.
13. Custom React data-fetching hooks (`useWallet`, `useAlerts`, `useStats`, `useClusterGraph`).
14. `APIClient` class with typed errors, retry, abort-signal support.
15. `LoadingSkeleton`, `ErrorState`, `EmptyState` shared components.
16. `MarketDistribution`, `TradeTimeline` components (filling spec gaps).
17. Responsive `BubbleGraph` with `ResizeObserver` and incremental simulation updates.
18. CSV / JSON export endpoints.

**P2 — nice to have (operational polish):**
19. WebSocket heartbeat + UI connection-status indicator.
20. Scheduler status endpoint surfaced in Settings page.
21. `/metrics` endpoint for Prometheus scraping.
22. Connection origin allowlist on WebSocket.
23. ENS / SNS resolution for funding-chain addresses.

The Phase 2/3/4/5 work below addresses every P0 and the high-leverage P1 items.
