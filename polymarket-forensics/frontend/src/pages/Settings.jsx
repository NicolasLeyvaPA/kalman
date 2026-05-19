import { useCallback } from 'react';
import { useAsync } from '../hooks/useAsync';
import { api } from '../services/api';
import { ErrorState } from '../components/ErrorState';
import { LoadingSkeleton } from '../components/LoadingSkeleton';

const ENV_VARS = [
  ['ALCHEMY_API_KEY', 'Polygon RPC key for funding-chain tracing'],
  ['DATABASE_URL', 'Postgres async connection string'],
  ['TRADE_POLL_INTERVAL', 'Seconds between trade-ingest passes (default 60)'],
  ['SCORING_INTERVAL', 'Seconds between scoring passes (default 300)'],
  ['CLUSTER_INTERVAL', 'Seconds between cluster detection (default 900)'],
  ['RESOLUTION_INTERVAL', 'Seconds between resolution attribution (default 3600)'],
  ['LARGE_TRADE_USD', 'USD threshold above which a trade is flagged "large"'],
  ['INSIDER_TRACE_THRESHOLD', 'Insider score that auto-enqueues a chain trace'],
  ['FRESH_WHALE_MIN_SIZE_USD', 'Minimum trade size for the FRESH_WHALE alert'],
  ['LOG_FORMAT', 'console (dev) or json (prod aggregators)'],
];

const WEIGHTS = [
  ['win_anomaly', 0.20],
  ['timing', 0.20],
  ['freshness', 0.15],
  ['concentration', 0.15],
  ['size_odds', 0.10],
  ['single_purpose', 0.08],
  ['sensitive_markets', 0.07],
  ['cluster', 0.05],
];

function SchedulerStatus() {
  const fetcher = useCallback((s) => api.schedulerStatus(s), []);
  const { data, loading, error, retry } = useAsync(fetcher, []);
  return (
    <div className="card p-4">
      {error && <ErrorState error={error} onRetry={retry} compact />}
      {loading && !data && <LoadingSkeleton lines={5} />}
      {data && (
        <table className="w-full text-xs">
          <thead className="text-[10px] uppercase text-text-dim">
            <tr>
              <th className="text-left p-1">Service</th>
              <th className="text-left p-1">Status</th>
              <th className="text-left p-1">Last started</th>
            </tr>
          </thead>
          <tbody>
            {data.services.map((s) => (
              <tr key={s.name} className="border-t border-border-subtle">
                <td className="p-1 text-text-primary">{s.name}</td>
                <td className="p-1">
                  <span className={s.running ? 'pill-green' : 'pill-red'}>
                    {s.running ? 'running' : 'stopped'}
                  </span>
                </td>
                <td className="p-1 text-text-dim">{s.last_started || '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default function Settings() {
  return (
    <div className="space-y-6 max-w-3xl">
      <section>
        <h1 className="text-sm uppercase tracking-wider text-text-secondary mb-3">
          Scheduler
        </h1>
        <SchedulerStatus />
      </section>

      <section>
        <h2 className="text-sm uppercase tracking-wider text-text-secondary mb-3">
          Backend configuration
        </h2>
        <div className="card p-4 text-xs space-y-3 text-text-secondary">
          <p>
            All runtime configuration lives in environment variables read by the
            FastAPI backend. To change anything below, edit{' '}
            <code className="text-neon-orange">.env</code> at the project root
            and restart the backend.
          </p>
          <table className="w-full text-xs">
            <tbody>
              {ENV_VARS.map(([k, v]) => (
                <tr key={k} className="border-b border-border-subtle">
                  <td className="p-2 text-neon-orange whitespace-nowrap">{k}</td>
                  <td className="p-2">{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2 className="text-sm uppercase tracking-wider text-text-secondary mb-3">
          Scoring weights
        </h2>
        <div className="card p-4 text-xs text-text-secondary space-y-2">
          <p>
            The insider score is a weighted combination of 8 sub-signals. Weights
            live in <code className="text-neon-orange">backend/scoring/types.py</code>{' '}
            (<code>ScoringConfig</code>).
          </p>
          <table className="w-full">
            <thead>
              <tr className="text-[10px] uppercase text-text-dim border-b border-border">
                <th className="p-1 text-left">Signal</th>
                <th className="p-1 text-right">Weight</th>
              </tr>
            </thead>
            <tbody>
              {WEIGHTS.map(([k, v]) => (
                <tr key={k} className="border-b border-border-subtle">
                  <td className="p-1">{k}</td>
                  <td className="p-1 text-right text-neon-orange">{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2 className="text-sm uppercase tracking-wider text-text-secondary mb-3">
          Export
        </h2>
        <div className="card p-4 space-y-2 text-xs">
          <a href={api.exportUrl('/export/wallets.csv?limit=10000')} className="btn block w-fit">
            Wallets CSV ↓
          </a>
          <a href={api.exportUrl('/export/trades.csv?limit=50000')} className="btn block w-fit">
            Trades CSV ↓
          </a>
          <a href={api.exportUrl('/export/alerts.json?limit=1000')} className="btn block w-fit">
            Alerts JSON ↓
          </a>
        </div>
      </section>

      <section>
        <h2 className="text-sm uppercase tracking-wider text-text-secondary mb-3">
          Disclaimer
        </h2>
        <div className="card p-4 text-xs text-text-secondary leading-relaxed">
          <p>
            This is a <strong className="text-neon-orange">personal analysis tool</strong>.
            The insider score is a heuristic — it is not proof. Do not publish
            attributions tying real names to addresses without legal review.
          </p>
          <p className="mt-2">
            Trading on insider information you detect from this dashboard may be
            illegal in your jurisdiction. Use this tool for research, monitoring,
            and avoidance — not for front-running.
          </p>
        </div>
      </section>
    </div>
  );
}
