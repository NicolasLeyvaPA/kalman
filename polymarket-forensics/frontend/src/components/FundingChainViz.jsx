import { shortAddress, fmtUsd } from '../utils/formatting';

function sourceColor(type) {
  switch (type) {
    case 'exchange': return 'text-neon-yellow border-neon-yellow/40';
    case 'bridge':   return 'text-neon-cyan border-neon-cyan/40';
    case 'contract': return 'text-neon-purple border-neon-purple/40';
    case 'wallet':   return 'text-text-secondary border-border';
    default:         return 'text-text-dim border-border';
  }
}

export default function FundingChainViz({ chain, targetAddress }) {
  if (!chain || chain.length === 0) {
    return (
      <div className="text-xs text-text-dim italic">
        No funding data yet. Trigger a trace from the wallet header.
      </div>
    );
  }

  const byDepth = {};
  for (const hop of chain) {
    if (!byDepth[hop.depth]) byDepth[hop.depth] = [];
    byDepth[hop.depth].push(hop);
  }

  const depths = Object.keys(byDepth).map(Number).sort((a, b) => a - b);

  return (
    <div className="space-y-3">
      <div className="text-[10px] text-text-dim uppercase tracking-wider">
        Wallet ← Funding sources
      </div>
      <div className="card p-3 inline-block">
        <span className="text-xs text-neon-orange">
          {shortAddress(targetAddress)}
        </span>
      </div>
      {depths.map(d => (
        <div key={d} className="ml-6 border-l border-border pl-4 space-y-1">
          <div className="text-[10px] text-text-dim">
            Depth {d}
          </div>
          {byDepth[d].slice(0, 12).map((h, i) => (
            <div key={i} className={`card p-2 inline-block mr-2 mb-1 ${sourceColor(h.source_type)}`}
                 style={{ borderWidth: 1 }}>
              <div className="text-xs">
                {h.source_exchange
                  ? `[${h.source_exchange.toUpperCase()}]`
                  : `[${h.source_type || 'unknown'}]`}{' '}
                {shortAddress(h.source_address)}
              </div>
              <div className="text-[10px] text-text-secondary">
                {fmtUsd(h.amount)} {h.asset}
              </div>
            </div>
          ))}
          {byDepth[d].length > 12 && (
            <div className="text-[10px] text-text-dim">
              +{byDepth[d].length - 12} more sources at this depth
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
