import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { api } from '../services/api';
import InsiderScoreBar from '../components/InsiderScoreBar';
import WalletScoreBreakdown from '../components/WalletScoreBreakdown';
import FundingChainViz from '../components/FundingChainViz';
import { fmtPct, fmtUsd, shortAddress, timeAgo } from '../utils/formatting';
import { classificationPill, scoreClass } from '../utils/colors';

export default function WalletExplorer() {
  const { address } = useParams();
  const [wallet, setWallet] = useState(null);
  const [trades, setTrades] = useState([]);
  const [funding, setFunding] = useState([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  async function load() {
    try {
      setErr(null);
      const [w, t, f] = await Promise.all([
        api.wallet(address),
        api.walletTrades(address, 50),
        api.walletFunding(address),
      ]);
      setWallet(w);
      setTrades(t);
      setFunding(f);
    } catch (e) {
      setErr(e.message);
    }
  }

  useEffect(() => { load(); }, [address]);

  async function trace() {
    setBusy(true);
    try {
      await api.traceWallet(address);
      setTimeout(load, 4000);
    } finally { setBusy(false); }
  }

  async function rescore() {
    setBusy(true);
    try {
      await api.rescoreWallet(address);
      load();
    } finally { setBusy(false); }
  }

  if (err) {
    return <div className="card p-6 text-neon-red text-xs">{err}</div>;
  }
  if (!wallet) {
    return <div className="text-text-dim text-xs">Loading wallet…</div>;
  }

  const categoryShare = wallet.total_volume
    ? wallet.political_military_volume / wallet.total_volume
    : 0;
  const topMarketShare = wallet.total_volume
    ? wallet.top_market_volume / wallet.total_volume
    : 0;

  return (
    <div className="space-y-6">
      <section className="card p-4">
        <div className="flex items-start gap-4 flex-wrap">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-2">
              <span className={classificationPill(wallet.classification)}>
                {wallet.classification?.replace(/_/g, ' ')}
              </span>
              {wallet.cluster_id && (
                <Link to={`/clusters?focus=${wallet.cluster_id}`}
                      className="pill-purple">
                  Cluster {wallet.cluster_id}
                </Link>
              )}
              {wallet.funding_exchange && (
                <span className="pill-cyan">
                  via {wallet.funding_exchange}
                </span>
              )}
            </div>
            <h1 className="text-xs text-text-secondary break-all">
              {wallet.ens_name && (
                <span className="text-neon-orange font-bold mr-2">
                  {wallet.ens_name}
                </span>
              )}
              {wallet.address}
            </h1>
            <div className="text-[10px] text-text-dim mt-1">
              First seen {timeAgo(wallet.first_seen)} · Last active {timeAgo(wallet.last_active)}
            </div>
          </div>
          <div className="flex flex-col items-end gap-1">
            <div className="text-[10px] text-text-dim uppercase">Insider score</div>
            <div className={`text-3xl font-bold ${scoreClass(wallet.insider_score)}`}>
              {wallet.insider_score?.toFixed(2)}
            </div>
            <InsiderScoreBar score={wallet.insider_score} width="w-32" />
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
        <Stat label="Total trades" value={wallet.total_trades} />
        <Stat label="Volume" value={fmtUsd(wallet.total_volume)} />
        <Stat label="PnL" value={fmtUsd(wallet.total_pnl)}
              accent={wallet.total_pnl >= 0 ? 'green' : 'red'} />
        <Stat label="Win rate" value={fmtPct(wallet.win_rate)}
              sublabel={`${wallet.wins}/${wallet.total_resolved}`} />
        <Stat label="Markets" value={wallet.markets_traded} />
        <Stat label="P-value" value={wallet.win_rate_p_value?.toExponential(1) || '—'}
              accent={wallet.win_rate_p_value < 0.001 ? 'red' : 'cyan'} />
      </section>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <section>
          <h2 className="text-sm uppercase tracking-wider text-text-secondary mb-3">
            Score breakdown
          </h2>
          <div className="card p-4">
            <WalletScoreBreakdown breakdown={wallet.score_breakdown} />
          </div>
        </section>

        <section>
          <h2 className="text-sm uppercase tracking-wider text-text-secondary mb-3">
            Funding chain
          </h2>
          <div className="card p-4">
            <FundingChainViz chain={funding} targetAddress={wallet.address} />
          </div>
        </section>
      </div>

      <section>
        <h2 className="text-sm uppercase tracking-wider text-text-secondary mb-3">
          Activity profile
        </h2>
        <div className="card p-4 space-y-3 text-xs">
          <Row label="Avg entry price"
               value={wallet.avg_entry_price ? Number(wallet.avg_entry_price).toFixed(3) : '—'} />
          <Row label="Avg trade size" value={fmtUsd(wallet.avg_trade_size)} />
          <Row label="Avg hours before resolution"
               value={wallet.avg_hours_before_resolution
                 ? `${Number(wallet.avg_hours_before_resolution).toFixed(1)}h`
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
              {trades.map(t => (
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
                    <Link to={`/market/${t.market_id}`}
                          className="hover:text-neon-orange">
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
              {trades.length === 0 && (
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
