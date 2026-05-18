import { Link } from 'react-router-dom';
import { scoreClass } from '../utils/colors';
import { fmtUsd, shortAddress, timeAgo } from '../utils/formatting';

export default function TradeRow({ trade }) {
  const side = trade.side?.toUpperCase();
  const outcome = trade.outcome?.toUpperCase();
  const sideColor = side === 'BUY'
    ? (outcome === 'YES' ? 'text-neon-green' : 'text-neon-red')
    : 'text-text-secondary';

  return (
    <div className="grid grid-cols-12 gap-2 py-1.5 px-2 text-xs hover:bg-bg-hover
                    border-b border-border-subtle">
      <Link to={`/wallet/${trade.wallet_address}`}
            className="col-span-2 hover:text-neon-orange truncate">
        {trade.wallet_ens || shortAddress(trade.wallet_address)}
      </Link>
      <span className={`col-span-1 ${scoreClass(trade.wallet_insider_score)}`}>
        {(trade.wallet_insider_score || 0).toFixed(2)}
      </span>
      <span className={`col-span-1 ${sideColor}`}>
        {side} {outcome}
      </span>
      <span className="col-span-1 text-text-primary">
        {fmtUsd(trade.size)}
      </span>
      <span className="col-span-1 text-text-secondary">
        @{Number(trade.price).toFixed(3)}
      </span>
      <Link to={`/market/${trade.market_id}`}
            className="col-span-5 text-text-secondary hover:text-text-primary truncate">
        {trade.market_question || trade.market_id}
      </Link>
      <span className="col-span-1 text-text-dim text-right">
        {timeAgo(trade.timestamp)}
      </span>
    </div>
  );
}
