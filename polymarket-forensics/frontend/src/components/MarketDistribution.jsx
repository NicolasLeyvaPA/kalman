/**
 * Bar chart of a wallet's volume distribution across market categories.
 *
 * Props:
 *   distribution: [{ category: string, volume: number }, ...]
 */
import { fmtPct, fmtUsd } from '../utils/formatting';

export default function MarketDistribution({ distribution }) {
  if (!distribution || distribution.length === 0) {
    return (
      <div className="text-xs text-text-dim italic">
        No category data available.
      </div>
    );
  }
  const total = distribution.reduce((s, d) => s + Number(d.volume || 0), 0) || 1;
  const sorted = [...distribution].sort((a, b) => b.volume - a.volume).slice(0, 10);
  const max = Math.max(...sorted.map((d) => d.volume), 1);

  return (
    <div className="space-y-1.5">
      {sorted.map((d) => {
        const w = (d.volume / max) * 100;
        const share = d.volume / total;
        return (
          <div key={d.category} className="flex items-center gap-3 text-xs">
            <span className="w-28 truncate text-text-secondary">{d.category}</span>
            <div className="flex-1 h-3 bg-bg-hover rounded overflow-hidden">
              <div className="h-full bg-neon-orange/60" style={{ width: `${w}%` }} />
            </div>
            <span className="w-20 text-right text-text-primary tabular-nums">
              {fmtUsd(d.volume)}
            </span>
            <span className="w-12 text-right text-text-dim tabular-nums">
              {fmtPct(share)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
