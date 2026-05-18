import { useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import FundingChainViz from '../components/FundingChainViz';
import InsiderScoreBar from '../components/InsiderScoreBar';
import MarketDistribution from '../components/MarketDistribution';
import TradeTimeline from '../components/TradeTimeline';
import WalletScoreBreakdown from '../components/WalletScoreBreakdown';
import { ErrorState } from '../components/ErrorState';
import { LoadingSkeleton } from '../components/LoadingSkeleton';
import { useWallet, useWalletFunding, useWalletTrades } from '../hooks/useWallet';
import { api } from '../services/api';
import { classificationPill, scoreClass } from '../utils/colors';
import { fmtPct, fmtUsd, timeAgo } from '../utils/formatting';

export default function WalletExplorer() {
  const { address } = useParams();
  const wallet = useWallet(address);
  const trades = useWalletTrades(address, 50);
  const funding = useWalletFunding(address);
  const [busy, setBusy] = useState(false);

  const distribution = useMemo(() => {
    if (!trades.data) return [];
    const map = new Map();
    for (const t of trades.data) {
      const c = t.market_category || 'uncategorized';
      map.set(c, (map.get(c) || 0) + Number(t.size || 0));
    }
    return Array.from(map.entries(), ([category, volume]) => ({ category, volume }));
  }, [trades.data]);

  async function trace() {
    setBusy(true);
    try {
      await api.traceWallet(address);
      setTimeout(() => { funding.retry(); wallet.retry(); }, 4000);
    } catch (e) {
      // surfaced via funding.error on next refresh
      console.error(e);
    } finally {
      setBusy(false);
    }
  }

  async function rescore() {
    setBusy(true);
    try {
      await api.rescoreWallet(address);
      wallet.retry();
    } finally {
      setBusy(false);
    }
  }

  if (wallet.error) {
    return <ErrorState error={wallet.error} onRetry={wallet.retry} />;
  }
  if (wallet.loading || !wallet.data) {
    return <LoadingSkeleton lines={6} />;
  }

  const w = wallet.data;
  const topMarketShare = w.total_volume > 0 ? w.top_market_volume / w.total_volume : 0;
  const categoryShare = w.total_volume > 0 ? w.political_military_volume / w.total_volume : 0;

  return (
    <div className="space-y-6">
      <section className="card p-4">
        <div className="flex items-start gap-4 flex-wrap">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-2">
              <span className={classificationPill(w.classification)}>
                {(w.classification || '').replace(/_/g, ' ')}
              </span>
              {w.cluster_id && (
                <Link to={`/clusters?focus=${w.cluster_id}`} className="pill-purple">
                  Cluster {w.cluster_id}
                </Link>
              )}
              {w.funding_exchange && (
                <span className="pill-cyan">via {w.funding_exchange}</span>
              )}
            </div>
            <h1 className="text-xs text-text-secondary break-all">
              {w.ens_name && (
                <span className="text-neon-orange font-bold mr-2">{w.ens_name}</span>
              )}
              {w.address}
            </h1>
            <div className="text-[10px] text-text-dim mt-1">
              First seen {timeAgo(w.first_seen)} · Last active {timeAgo(w.last_active)}
            </div>
          </div>
          <div className="flex flex-col items-end gap-1">
            <div className="text-[10px] text-text-dim uppercase">Insider score</div>
            <div className={`text-3xl font-bold ${scoreClass(w.insider_score)}`}>
              {Number(w.insider_score).toFixed(2)}
            </div>
            <InsiderScoreBar score={w.insider_score} width="w-32" />
          </div>
          <div className="flex flex-col gap-2">
            <button onClick={trace} disabled={busy} className="btn-primary disabled:opacity-50">
              Trace funding
            </button>
            <button onClick={rescore} disabled={busy} className="btn disabled:opacity-50">
              Rescore
            </button>
          </div>
        </div>
      </section>

      <section className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
        <Stat label="Total trades" value={w.total_trades} />
        <Stat label="Volume" value={fmtUsd(w.total_volume)} />
        <Stat label="PnL" value={fmtUsd(w.total_pnl)}
              accent={Number(w.total_pnl) >= 0 ? 'green' : 'red'} />
        <Stat label="Win rate" value={fmtPct(w.win_rate)}
              sublabel={`${w.wins}/${w.total_resolved}`} />
        <Stat label="Markets" value={w.markets_traded} />
        <Stat label="P-value"
              value={w.win_rate_p_value ? Number(w.win_rate_p_value).toExponential(1) : '—'}
              accent={Number(w.win_rate_p_value) < 0.001 ? 'red' : 'cyan'} />
      </section>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <section>
          <h2 className="text-sm uppercase tracking-wider text-text-secondary mb-3">
            Score breakdown
          </h2>
          <div className="card p-4">
            <WalletScoreBreakdown breakdown={w.score_breakdown} />
          </div>
        </section>

        <section>
          <h2 className="text-sm uppercase tracking-wider text-text-secondary mb-3">
            Funding chain
          </h2>
          <div className="card p-4">
            {funding.error && (
              <ErrorState error={funding.error} onRetry={funding.retry} compact />
            )}
            {funding.data && (
              <FundingChainViz chain={funding.data} targetAddress={w.address} />
            )}
          </div>
        </section>
      </div>

      <section>
        <h2 className="text-sm uppercase tracking-wider text-text-secondary mb-3">
          Market category distribution
        </h2>
        <div className="card p-4">
          <MarketDistribution distribution={distribution} />
        </div>
      </section>

      <section>
        <h2 className="text-sm uppercase tracking-wider text-text-secondary mb-3">
          Trade timeline
        </h2>
        <div className="card p-4">
          <TradeTimeline trades={trades.data || []} />
        </div>
      </section>

      <section>
        <h2 className="text-sm uppercase tracking-wider text-text-secondary mb-3">
          Activity profile
        </h2>
        <div className="card p-4 space-y-3 text-xs">
          <Row label="Avg entry price"
               value={w.avg_entry_price ? Number(w.avg_entry_price).toFixed(3) : '—'} />
          <Row label="Avg trade size" value={fmtUsd(w.avg_trade_size)} />
          <Row label="Avg hours before resolution"
               value={w.avg_hours_before_resolution
                 ? `${Number(w.avg_hours_before_resolution).toFixed(1)}h`
                 : '—'} />
          <Row label="Top market concentration" value={fmtPct(topMarketShare)} />
          <Row label="Political/military share" value={fmtPct(categoryShare)} />
        </div>
      </section>

      <section>
        <h2 className="text-sm uppercase tracking-wider text-text-secondary mb-3">
          Trade history
        </h2>
        <div className="card overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="text-[10px] uppercase text-text-dim border-b border-border">
              <tr>
                <th className="text-left p-2">Time</th>
                <th className="text-left p-2">Side</th>
                <th className="text-right p-2">Size</th>
                <th className="text-right p-2">Price</th>
                <th className="text-left p-2">Market</th>
                <th className="text-right p-2">PnL</th>
              </tr>
            </thead>
            <tbody>
              {(trades.data || []).map((t) => (
                <tr key={t.id} className="border-b border-border-subtle hover:bg-bg-hover">
                  <td className="p-2 text-text-secondary whitespace-nowrap">
                    {timeAgo(t.timestamp)}
                  </td>
                  <td className="p-2">
                    <span className={t.side === 'BUY' ? 'text-neon-green' : 'text-neon-red'}>
                      {t.side} {t.outcome}
                    </span>
                  </td>
                  <td className="p-2 text-right">{fmtUsd(t.size)}</td>
                  <td className="p-2 text-right">@{Number(t.price).toFixed(3)}</td>
                  <td className="p-2 truncate max-w-md">
                    <Link to={`/market/${t.market_id}`} className="hover:text-neon-orange">
                      {t.market_question || t.market_id}
                    </Link>
                  </td>
                  <td className={`p-2 text-right ${
                    t.trade_won === true ? 'text-neon-green'
                    : t.trade_won === false ? 'text-neon-red' : 'text-text-dim'
                  }`}>
                    {t.pnl ? fmtUsd(t.pnl) : t.resolution_outcome ? '—' : 'open'}
                  </td>
                </tr>
              ))}
              {(trades.data || []).length === 0 && (
                <tr><td colSpan={6} className="p-6 text-center text-text-dim">
                  No trades recorded yet.
                </td></tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function Stat({ label, value, sublabel, accent }) {
  const accentClass = {
    red: 'text-neon-red', green: 'text-neon-green', orange: 'text-neon-orange',
    cyan: 'text-neon-cyan', yellow: 'text-neon-yellow',
  }[accent] || 'text-text-primary';
  return (
    <div className="card p-3">
      <div className="text-[10px] text-text-dim uppercase tracking-wider mb-1">{label}</div>
      <div className={`text-lg font-semibold ${accentClass}`}>{value}</div>
      {sublabel && <div className="text-[10px] text-text-secondary mt-1">{sublabel}</div>}
    </div>
  );
}

function Row({ label, value }) {
  return (
    <div className="flex justify-between border-b border-border-subtle pb-2 last:border-0 last:pb-0">
      <span className="text-text-secondary">{label}</span>
      <span className="text-text-primary tabular-nums">{value}</span>
    </div>
  );
}
