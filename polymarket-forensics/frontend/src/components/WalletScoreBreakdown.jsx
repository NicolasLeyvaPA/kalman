import InsiderScoreBar from './InsiderScoreBar';

const SIGNALS = [
  ['freshness',         'Freshness',         'New wallet, first big trade soon after creation'],
  ['concentration',     'Concentration',     'Volume concentrated in one market or category'],
  ['win_anomaly',       'Win anomaly',       'Binomial p-value vs. avg implied odds'],
  ['timing',            'Timing',            'Avg hours before market resolution'],
  ['size_odds',         'Size vs odds',      'Aggressive sizing at long odds'],
  ['single_purpose',    'Single purpose',    'Wallet only touches CEX + Polymarket'],
  ['sensitive_markets', 'Sensitive markets', '% volume in political / military'],
  ['cluster',           'Cluster',           'Linked to other flagged wallets'],
];

export default function WalletScoreBreakdown({ breakdown }) {
  if (!breakdown) {
    return (
      <div className="text-xs text-text-dim">No score breakdown available.</div>
    );
  }
  return (
    <div className="space-y-2">
      {SIGNALS.map(([key, label, hint]) => {
        const v = breakdown[key] ?? 0;
        return (
          <div key={key} className="flex items-center gap-3" title={hint}>
            <span className="w-32 text-xs text-text-secondary">{label}</span>
            <InsiderScoreBar score={v} width="w-48" />
            <span className="text-xs text-text-primary tabular-nums">
              {Number(v).toFixed(2)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
