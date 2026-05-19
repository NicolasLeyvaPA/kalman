import { useCallback, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import InsiderScoreBar from '../components/InsiderScoreBar';
import { ErrorState } from '../components/ErrorState';
import { LoadingSkeleton, TableSkeleton } from '../components/LoadingSkeleton';
import { useAsync } from '../hooks/useAsync';
import { api } from '../services/api';
import { scoreClass } from '../utils/colors';
import { fmtUsd, shortAddress } from '../utils/formatting';

function MarketDetail({ id }) {
  const fetcher = useCallback((s) => api.marketForensics(id, s), [id]);
  const { data, loading, error, retry } = useAsync(fetcher, [id]);

  if (error) return <ErrorState error={error} onRetry={retry} />;
  if (loading || !data) return <LoadingSkeleton lines={6} />;

  const dirtyShare = (data.dirty_volume + data.clean_volume) > 0
    ? data.dirty_volume / (data.dirty_volume + data.clean_volume)
    : 0;

  return (
    <div className="space-y-6">
      <Link to="/markets" className="text-xs text-text-secondary hover:text-neon-orange">
        ← All markets
      </Link>
      <section className="card p-4">
        <h1 className="text-lg text-text-primary mb-1">{data.market.question}</h1>
        <div className="flex flex-wrap gap-2 text-[10px] text-text-secondary">
          <span className="pill-cyan">{data.market.category}</span>
          <span className={data.market.status === 'resolved' ? 'pill-purple' : 'pill-green'}>
            {data.market.status}
          </span>
          {data.market.resolution_outcome && (
            <span className="pill-yellow">Resolved {data.market.resolution_outcome}</span>
          )}
        </div>
      </section>

      <section className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat label="Total volume" value={fmtUsd(data.market.volume_total)} />
        <Stat label="Dirty volume" value={fmtUsd(data.dirty_volume)} accent="red" />
        <Stat label="Clean volume" value={fmtUsd(data.clean_volume)} accent="green" />
        <Stat label="Dirty share" value={`${(dirtyShare * 100).toFixed(1)}%`}
              accent={dirtyShare > 0.3 ? 'red' : dirtyShare > 0.1 ? 'orange' : 'green'} />
      </section>

      <section>
        <h2 className="text-sm uppercase tracking-wider text-text-secondary mb-3">
          Active wallets ({data.wallets.length})
        </h2>
        <div className="card overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="text-[10px] uppercase text-text-dim border-b border-border">
              <tr>
                <th className="text-left p-2">Wallet</th>
                <th className="text-right p-2">Insider</th>
                <th className="text-left p-2">Class</th>
                <th className="text-left p-2">Funded via</th>
                <th className="text-right p-2">Volume</th>
                <th className="text-right p-2">Trades</th>
              </tr>
            </thead>
            <tbody>
              {data.wallets.map((w) => (
                <tr key={w.address} className="border-b border-border-subtle hover:bg-bg-hover">
                  <td className="p-2">
                    <Link to={`/wallet/${w.address}`} className="hover:text-neon-orange">
                      {shortAddress(w.address)}
                    </Link>
                  </td>
                  <td className="p-2 text-right">
                    <div className="flex items-center gap-2 justify-end">
                      <InsiderScoreBar score={w.insider_score} width="w-16" />
                      <span className={`tabular-nums ${scoreClass(w.insider_score)}`}>
                        {Number(w.insider_score).toFixed(2)}
                      </span>
                    </div>
                  </td>
                  <td className="p-2 text-text-secondary">{w.classification}</td>
                  <td className="p-2 text-text-secondary">{w.funding_exchange || '—'}</td>
                  <td className="p-2 text-right tabular-nums">{fmtUsd(w.volume)}</td>
                  <td className="p-2 text-right tabular-nums">{w.trades}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function MarketList() {
  const [category, setCategory] = useState('');
  const fetcher = useCallback(
    (s) => api.markets({ category: category || undefined, limit: 80 }, s),
    [category],
  );
  const { data, loading, error, retry } = useAsync(fetcher, [category]);

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between gap-4 flex-wrap">
        <h1 className="text-sm uppercase tracking-wider text-text-secondary">
          Markets
        </h1>
        <div className="flex items-center gap-2 text-xs">
          <label className="text-text-secondary">Category</label>
          <input
            type="text" placeholder="e.g. politics, military"
            value={category}
            onChange={(e) => setCategory(e.target.value)}
            className="px-2 py-1 bg-bg-card border border-border rounded text-xs"
          />
        </div>
      </div>

      {error && <ErrorState error={error} onRetry={retry} />}
      {loading && !data && <TableSkeleton rows={10} />}
      {data && (
        <div className="card overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="text-[10px] uppercase text-text-dim border-b border-border">
              <tr>
                <th className="text-left p-2">Question</th>
                <th className="text-left p-2">Category</th>
                <th className="text-left p-2">Status</th>
                <th className="text-right p-2">Price</th>
                <th className="text-right p-2">Volume</th>
              </tr>
            </thead>
            <tbody>
              {data.map((m) => (
                <tr key={m.id} className="border-b border-border-subtle hover:bg-bg-hover">
                  <td className="p-2 max-w-xl">
                    <Link to={`/market/${m.id}`}
                          className="hover:text-neon-orange truncate block">
                      {m.question}
                    </Link>
                  </td>
                  <td className="p-2 text-text-secondary">{m.category || '—'}</td>
                  <td className="p-2 text-text-secondary">{m.status || '—'}</td>
                  <td className="p-2 text-right">
                    {m.current_price ? Number(m.current_price).toFixed(3) : '—'}
                  </td>
                  <td className="p-2 text-right tabular-nums">{fmtUsd(m.volume_total)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default function MarketIntel() {
  const { id } = useParams();
  return id ? <MarketDetail id={id} /> : <MarketList />;
}

function Stat({ label, value, accent }) {
  const accentClass = {
    red: 'text-neon-red', green: 'text-neon-green',
    orange: 'text-neon-orange', cyan: 'text-neon-cyan',
  }[accent] || 'text-text-primary';
  return (
    <div className="card p-3">
      <div className="text-[10px] text-text-dim uppercase tracking-wider mb-1">{label}</div>
      <div className={`text-lg font-semibold ${accentClass}`}>{value}</div>
    </div>
  );
}
