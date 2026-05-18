import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../services/api';
import InsiderScoreBar from '../components/InsiderScoreBar';
import { fmtPct, fmtUsd, shortAddress, timeAgo } from '../utils/formatting';
import { classificationPill, scoreClass } from '../utils/colors';

export default function Leaderboard() {
  const [wallets, setWallets] = useState([]);
  const [minScore, setMinScore] = useState(0.3);
  const [classification, setClassification] = useState('');
  const [order, setOrder] = useState('insider_score');
  const [total, setTotal] = useState(0);

  useEffect(() => {
    async function load() {
      const res = await api.wallets({
        minScore,
        classification: classification || undefined,
        order,
        limit: 200,
      });
      setWallets(res.wallets);
      setTotal(res.total);
    }
    load();
  }, [minScore, classification, order]);

  return (
    <div className="space-y-4">
      <div className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-sm uppercase tracking-wider text-text-secondary">
            Insider leaderboard
          </h1>
          <p className="text-[10px] text-text-dim mt-1">
            {total.toLocaleString()} wallets match. Click any row for full forensic profile.
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
            <option value="">All classifications</option>
            <option value="insider_suspect">Insider suspect</option>
            <option value="suspicious">Suspicious</option>
            <option value="watch">Watch</option>
            <option value="normal">Normal</option>
            <option value="unknown">Unknown</option>
          </select>
          <select value={order}
                  onChange={(e) => setOrder(e.target.value)}
                  className="bg-bg-card border border-border rounded px-2 py-1 text-xs">
            <option value="insider_score">Sort: insider score</option>
            <option value="smart_score">Sort: smart score</option>
            <option value="total_pnl">Sort: PnL</option>
            <option value="total_volume">Sort: volume</option>
            <option value="win_rate">Sort: win rate</option>
            <option value="last_active">Sort: recency</option>
          </select>
        </div>
      </div>

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
            {wallets.map((w, i) => (
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
                    {w.classification?.replace(/_/g, ' ') || '—'}
                  </span>
                </td>
                <td className="p-2 text-right tabular-nums">
                  {fmtPct(w.win_rate)} ({w.wins}/{w.total_resolved})
                </td>
                <td className={`p-2 text-right tabular-nums ${
                  w.total_pnl >= 0 ? 'text-neon-green' : 'text-neon-red'
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
            {wallets.length === 0 && (
              <tr><td colSpan={11} className="p-12 text-center text-text-dim">
                No wallets match the current filters.
              </td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
