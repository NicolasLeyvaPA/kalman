import { useCallback, useState } from 'react';
import { Link } from 'react-router-dom';
import InsiderScoreBar from '../components/InsiderScoreBar';
import { ErrorState } from '../components/ErrorState';
import { TableSkeleton } from '../components/LoadingSkeleton';
import { useAsync } from '../hooks/useAsync';
import { api } from '../services/api';
import { classificationPill, scoreClass } from '../utils/colors';
import { fmtPct, fmtUsd, shortAddress, timeAgo } from '../utils/formatting';

const ORDERS = [
  ['insider_score', 'insider score'],
  ['smart_score', 'smart score'],
  ['total_pnl', 'PnL'],
  ['total_volume', 'volume'],
  ['win_rate', 'win rate'],
  ['last_active', 'recency'],
];

const CLASSIFICATIONS = [
  ['', 'all'],
  ['insider_suspect', 'insider suspect'],
  ['suspicious', 'suspicious'],
  ['watch', 'watch'],
  ['normal', 'normal'],
  ['unknown', 'unknown'],
];

export default function Leaderboard() {
  const [minScore, setMinScore] = useState(0.3);
  const [classification, setClassification] = useState('');
  const [order, setOrder] = useState('insider_score');

  const fetcher = useCallback(
    (signal) => api.wallets({
      minScore,
      classification: classification || undefined,
      order,
      limit: 200,
    }, signal),
    [minScore, classification, order],
  );
  const { data, loading, error, retry } = useAsync(
    fetcher, [minScore, classification, order],
  );

  const exportUrl = api.exportUrl(
    `/export/wallets.csv?min_score=${minScore}&limit=10000`,
  );

  return (
    <div className="space-y-4">
      <div className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-sm uppercase tracking-wider text-text-secondary">
            Insider leaderboard
          </h1>
          <p className="text-[10px] text-text-dim mt-1">
            {(data?.total ?? 0).toLocaleString()} wallets match.
            Click any row for the full forensic profile.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3 text-xs">
          <label className="flex items-center gap-2 text-text-secondary">
            Min score
            <input type="range" min="0" max="1" step="0.05"
              value={minScore}
              onChange={(e) => setMinScore(parseFloat(e.target.value))}
              className="w-24" />
            <span className="text-text-primary tabular-nums">{minScore.toFixed(2)}</span>
          </label>
          <select value={classification}
                  onChange={(e) => setClassification(e.target.value)}
                  className="bg-bg-card border border-border rounded px-2 py-1 text-xs">
            {CLASSIFICATIONS.map(([v, label]) => (
              <option key={v} value={v}>{`Class: ${label}`}</option>
            ))}
          </select>
          <select value={order}
                  onChange={(e) => setOrder(e.target.value)}
                  className="bg-bg-card border border-border rounded px-2 py-1 text-xs">
            {ORDERS.map(([v, label]) => (
              <option key={v} value={v}>{`Sort: ${label}`}</option>
            ))}
          </select>
          <a href={exportUrl} className="btn">CSV ↓</a>
        </div>
      </div>

      {error && <ErrorState error={error} onRetry={retry} />}
      {loading && !data && <TableSkeleton rows={12} />}
      {data && (
        <div className="card overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="text-[10px] uppercase text-text-dim border-b border-border">
              <tr>
                <th className="p-2 text-left">#</th>
                <th className="p-2 text-left">Wallet</th>
                <th className="p-2 text-left">Score</th>
                <th className="p-2 text-left">Class</th>
                <th className="p-2 text-right">Win rate</th>
                <th className="p-2 text-right">PnL</th>
                <th className="p-2 text-right">Volume</th>
                <th className="p-2 text-right">Markets</th>
                <th className="p-2 text-left">Funded via</th>
                <th className="p-2 text-left">Cluster</th>
                <th className="p-2 text-right">Active</th>
              </tr>
            </thead>
            <tbody>
              {data.wallets.map((w, i) => (
                <tr key={w.address}
                    className="border-b border-border-subtle hover:bg-bg-hover">
                  <td className="p-2 text-text-dim">{i + 1}</td>
                  <td className="p-2">
                    <Link to={`/wallet/${w.address}`}
                          className="hover:text-neon-orange">
                      {w.ens_name || shortAddress(w.address)}
                    </Link>
                  </td>
                  <td className="p-2">
                    <div className="flex items-center gap-2">
                      <InsiderScoreBar score={w.insider_score} width="w-16" />
                      <span className={`tabular-nums ${scoreClass(w.insider_score)}`}>
                        {Number(w.insider_score).toFixed(2)}
                      </span>
                    </div>
                  </td>
                  <td className="p-2">
                    <span className={classificationPill(w.classification)}>
                      {(w.classification || '').replace(/_/g, ' ') || '—'}
                    </span>
                  </td>
                  <td className="p-2 text-right tabular-nums">
                    {fmtPct(w.win_rate)} ({w.wins}/{w.total_resolved})
                  </td>
                  <td className={`p-2 text-right tabular-nums ${
                    Number(w.total_pnl) >= 0 ? 'text-neon-green' : 'text-neon-red'
                  }`}>
                    {fmtUsd(w.total_pnl)}
                  </td>
                  <td className="p-2 text-right tabular-nums">{fmtUsd(w.total_volume)}</td>
                  <td className="p-2 text-right tabular-nums">{w.markets_traded}</td>
                  <td className="p-2 text-text-secondary">{w.funding_exchange || '—'}</td>
                  <td className="p-2 text-text-secondary">{w.cluster_id || '—'}</td>
                  <td className="p-2 text-right text-text-dim">{timeAgo(w.last_active)}</td>
                </tr>
              ))}
              {data.wallets.length === 0 && (
                <tr><td colSpan={11} className="p-12 text-center text-text-dim">
                  No wallets match the current filters.
                </td></tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
