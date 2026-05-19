/**
 * Compact horizontal timeline of trades.
 *
 * Each trade is a vertical tick on the x-axis (time), colored by side and
 * sized by trade size. Hover for details.
 *
 * Props:
 *   trades: [{ id, timestamp, size, price, side, outcome, is_large }, ...]
 *   height: optional; defaults to 80
 */
import { useMemo } from 'react';
import { fmtUsd, timeAgo } from '../utils/formatting';

export default function TradeTimeline({ trades, height = 80 }) {
  const points = useMemo(() => {
    if (!trades || trades.length === 0) return [];
    const ts = trades.map((t) => new Date(t.timestamp).getTime()).filter(Boolean);
    if (ts.length === 0) return [];
    const minT = Math.min(...ts);
    const maxT = Math.max(...ts);
    const span = Math.max(1, maxT - minT);
    const maxSize = Math.max(...trades.map((t) => Number(t.size || 0)), 1);
    return trades.map((t) => {
      const x = ((new Date(t.timestamp).getTime() - minT) / span) * 100;
      const intensity = Math.min(1, Number(t.size || 0) / maxSize);
      const size = 2 + intensity * 6;
      const color = t.side === 'BUY'
        ? t.outcome === 'YES' ? '#00ff41' : '#ff4444'
        : '#888';
      return { ...t, x, size, color };
    });
  }, [trades]);

  if (points.length === 0) {
    return (
      <div className="text-xs text-text-dim italic h-20 flex items-center justify-center">
        No trades to plot.
      </div>
    );
  }

  return (
    <div className="relative w-full" style={{ height }}>
      <div className="absolute inset-x-0 bottom-1/2 h-px bg-border" />
      {points.map((p) => (
        <div
          key={p.id}
          className="absolute -translate-x-1/2 -translate-y-1/2 rounded-full cursor-pointer"
          style={{
            left: `${p.x}%`,
            top: '50%',
            width: p.size,
            height: p.size,
            background: p.color,
            boxShadow: p.is_large ? `0 0 6px ${p.color}` : undefined,
          }}
          title={
            `${timeAgo(p.timestamp)} · ${p.side} ${p.outcome} ${fmtUsd(p.size)} @ `
            + `${Number(p.price).toFixed(3)}`
          }
        />
      ))}
      <div className="absolute left-0 bottom-0 text-[9px] text-text-dim">
        {points.length > 0 ? timeAgo(trades[trades.length - 1].timestamp) : ''}
      </div>
      <div className="absolute right-0 bottom-0 text-[9px] text-text-dim">
        now
      </div>
    </div>
  );
}
