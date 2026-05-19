import { useState } from 'react';
import AlertCard from '../components/AlertCard';
import { ErrorState } from '../components/ErrorState';
import { LoadingSkeleton, TableSkeleton } from '../components/LoadingSkeleton';
import StatCard from '../components/StatCard';
import TradeRow from '../components/TradeRow';
import { useAlerts } from '../hooks/useAlerts';
import { usePolledAsync } from '../hooks/useAsync';
import { api } from '../services/api';
import { fmtUsd } from '../utils/formatting';

const SEVERITY_FILTERS = ['all', 'critical', 'high', 'medium', 'low'];

export default function LiveFeed() {
  const [filter, setFilter] = useState('all');

  const stats = usePolledAsync((s) => api.stats(s), [], { intervalMs: 20000 });
  const trades = usePolledAsync(
    (s) => api.recentTrades(40, s), [], { intervalMs: 20000 },
  );
  const { alerts, dismiss, error: alertsError, refresh } = useAlerts({ max: 100 });

  const filtered = filter === 'all'
    ? alerts
    : alerts.filter((a) => a.severity === filter);

  return (
    <div className="space-y-6">
      {stats.error && <ErrorState error={stats.error} onRetry={stats.retry} compact />}
      {stats.data && !stats.error && (
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
          <StatCard label="Wallets tracked" value={stats.data.wallets_total.toLocaleString()} />
          <StatCard label="Trades observed" value={stats.data.trades_total.toLocaleString()} />
          <StatCard label="24h volume" value={fmtUsd(stats.data.volume_24h)} accent="cyan" />
          <StatCard label="Large trades 24h"
                    value={stats.data.large_trades_24h.toLocaleString()} accent="orange" />
          <StatCard label="Insider suspects"
                    value={stats.data.insider_suspects.toLocaleString()} accent="red" />
          <StatCard label="Critical alerts"
                    value={stats.data.critical_alerts.toLocaleString()}
                    accent={stats.data.critical_alerts > 0 ? 'red' : 'green'} />
        </div>
      )}
      {stats.loading && !stats.data && <LoadingSkeleton lines={3} />}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <section className="lg:col-span-2">
          <div className="flex items-center gap-2 mb-3">
            <h2 className="text-sm uppercase tracking-wider text-text-secondary">
              Alert feed
            </h2>
            <div className="flex gap-1 ml-auto text-[10px]">
              {SEVERITY_FILTERS.map((f) => (
                <button key={f} onClick={() => setFilter(f)}
                  className={`px-2 py-0.5 rounded uppercase tracking-wider ${
                    filter === f
                      ? 'bg-neon-orange/10 text-neon-orange border border-neon-orange/40'
                      : 'text-text-secondary border border-border hover:bg-bg-hover'
                  }`}>
                  {f}
                </button>
              ))}
            </div>
          </div>
          {alertsError && <ErrorState error={alertsError} onRetry={refresh} compact />}
          {!alertsError && filtered.length === 0 && (
            <div className="card p-6 text-center text-text-dim text-xs">
              No alerts. The system is quiet — or it's still warming up.
            </div>
          )}
          {filtered.map((a) => (
            <AlertCard key={a.id} alert={a} onDismiss={dismiss} />
          ))}
        </section>

        <section>
          <h2 className="text-sm uppercase tracking-wider text-text-secondary mb-3">
            Recent large trades
          </h2>
          {trades.error && (
            <ErrorState error={trades.error} onRetry={trades.retry} compact />
          )}
          {trades.loading && !trades.data && <TableSkeleton rows={8} />}
          {trades.data && (
            <div className="card p-2">
              <div className="grid grid-cols-12 gap-2 text-[10px] text-text-dim uppercase
                              border-b border-border pb-1 px-2">
                <span className="col-span-2">Wallet</span>
                <span className="col-span-1">Score</span>
                <span className="col-span-1">Side</span>
                <span className="col-span-1">Size</span>
                <span className="col-span-1">Price</span>
                <span className="col-span-5">Market</span>
                <span className="col-span-1 text-right">Age</span>
              </div>
              {trades.data.length === 0 && (
                <div className="text-text-dim text-xs p-4 text-center">
                  Waiting for trades…
                </div>
              )}
              {trades.data.map((t) => <TradeRow key={t.id} trade={t} />)}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
