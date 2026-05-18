-- Polymarket Forensics Dashboard schema

CREATE TABLE IF NOT EXISTS wallets (
    address TEXT PRIMARY KEY,
    first_seen TIMESTAMPTZ,
    last_active TIMESTAMPTZ,
    total_trades INTEGER DEFAULT 0,
    total_volume NUMERIC DEFAULT 0,
    total_pnl NUMERIC DEFAULT 0,
    win_rate NUMERIC DEFAULT 0,
    win_rate_p_value NUMERIC,
    total_resolved INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    markets_traded INTEGER DEFAULT 0,
    avg_entry_price NUMERIC,
    avg_trade_size NUMERIC,
    avg_hours_before_resolution NUMERIC,
    top_market_volume NUMERIC DEFAULT 0,
    top_category_volume NUMERIC DEFAULT 0,
    political_military_volume NUMERIC DEFAULT 0,
    unique_protocols INTEGER DEFAULT 0,
    total_tx_count INTEGER DEFAULT 0,
    smart_score NUMERIC DEFAULT 0,
    insider_score NUMERIC DEFAULT 0,
    score_breakdown JSONB,
    cluster_id TEXT,
    classification TEXT DEFAULT 'unknown',
    funding_source TEXT,
    funding_exchange TEXT,
    ens_name TEXT,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS trades (
    id BIGSERIAL PRIMARY KEY,
    wallet_address TEXT REFERENCES wallets(address) ON DELETE CASCADE,
    market_id TEXT NOT NULL,
    market_question TEXT,
    market_category TEXT,
    side TEXT NOT NULL,
    outcome TEXT NOT NULL,
    size NUMERIC NOT NULL,
    price NUMERIC NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    tx_hash TEXT,
    is_large BOOLEAN DEFAULT FALSE,
    hours_before_resolution NUMERIC,
    resolution_outcome TEXT,
    trade_won BOOLEAN,
    pnl NUMERIC,
    UNIQUE (wallet_address, market_id, tx_hash, timestamp)
);

CREATE TABLE IF NOT EXISTS funding_chains (
    id BIGSERIAL PRIMARY KEY,
    wallet_address TEXT REFERENCES wallets(address) ON DELETE CASCADE,
    source_address TEXT NOT NULL,
    source_type TEXT,
    source_exchange TEXT,
    amount NUMERIC,
    asset TEXT,
    timestamp TIMESTAMPTZ,
    tx_hash TEXT,
    depth INTEGER DEFAULT 0,
    traced_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS clusters (
    id TEXT PRIMARY KEY,
    wallets TEXT[] NOT NULL,
    cluster_type TEXT,
    evidence TEXT,
    total_pnl NUMERIC,
    combined_win_rate NUMERIC,
    markets_in_common TEXT[],
    insider_probability NUMERIC,
    first_detected TIMESTAMPTZ DEFAULT NOW(),
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    status TEXT DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS alerts (
    id BIGSERIAL PRIMARY KEY,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    wallet_address TEXT,
    cluster_id TEXT,
    market_id TEXT,
    data JSONB,
    dismissed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS markets (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    category TEXT,
    status TEXT,
    resolution_outcome TEXT,
    resolved_at TIMESTAMPTZ,
    current_price NUMERIC,
    volume_total NUMERIC,
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS known_addresses (
    address TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    category TEXT NOT NULL,
    chain TEXT DEFAULT 'polygon',
    added_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_wallet ON trades(wallet_address);
CREATE INDEX IF NOT EXISTS idx_trades_market ON trades(market_id);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_large ON trades(is_large) WHERE is_large = TRUE;
CREATE INDEX IF NOT EXISTS idx_funding_wallet ON funding_chains(wallet_address);
CREATE INDEX IF NOT EXISTS idx_funding_source ON funding_chains(source_address);
CREATE INDEX IF NOT EXISTS idx_wallets_insider ON wallets(insider_score DESC);
CREATE INDEX IF NOT EXISTS idx_wallets_cluster ON wallets(cluster_id);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity, dismissed);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_known_category ON known_addresses(category);
