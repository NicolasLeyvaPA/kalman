export default function Settings() {
  return (
    <div className="space-y-6 max-w-3xl">
      <section>
        <h1 className="text-sm uppercase tracking-wider text-text-secondary mb-3">
          Backend configuration
        </h1>
        <div className="card p-4 text-xs space-y-3 text-text-secondary">
          <p>
            All runtime configuration lives in environment variables read by the
            FastAPI backend. To change anything below, edit{' '}
            <code className="text-neon-orange">.env</code> at the project root
            and restart the backend.
          </p>
          <table className="w-full text-xs">
            <tbody>
              {[
                ['ALCHEMY_API_KEY', 'Polygon RPC key for funding-chain tracing'],
                ['DATABASE_URL', 'Postgres async connection string'],
                ['TRADE_POLL_INTERVAL', 'Seconds between trade-ingest passes (default 60)'],
                ['SCORING_INTERVAL', 'Seconds between scoring passes (default 300)'],
                ['CLUSTER_INTERVAL', 'Seconds between cluster detection (default 900)'],
                ['RESOLUTION_INTERVAL', 'Seconds between resolution attribution (default 3600)'],
                ['LARGE_TRADE_USD', 'USD threshold above which a trade is flagged "large"'],
                ['INSIDER_TRACE_THRESHOLD', 'Insider score that auto-enqueues a chain trace'],
              ].map(([k, v]) => (
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
            live in <code className="text-neon-orange">backend/scoring/insider_score.py</code>.
          </p>
          <table className="w-full">
            <thead>
              <tr className="text-[10px] uppercase text-text-dim border-b border-border">
                <th className="p-1 text-left">Signal</th>
                <th className="p-1 text-right">Weight</th>
              </tr>
            </thead>
            <tbody>
              {[
                ['win_anomaly', 0.20],
                ['timing', 0.20],
                ['freshness', 0.15],
                ['concentration', 0.15],
                ['size_odds', 0.10],
                ['single_purpose', 0.08],
                ['sensitive_markets', 0.07],
                ['cluster', 0.05],
              ].map(([k, v]) => (
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
