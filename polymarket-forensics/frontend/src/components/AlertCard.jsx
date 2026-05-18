import { Link } from 'react-router-dom';
import { severityClass } from '../utils/colors';
import { timeAgo, shortAddress } from '../utils/formatting';

export default function AlertCard({ alert, onDismiss }) {
  const link = alert.cluster_id
    ? `/clusters?focus=${alert.cluster_id}`
    : alert.wallet_address
    ? `/wallet/${alert.wallet_address}`
    : alert.market_id
    ? `/market/${alert.market_id}`
    : null;

  return (
    <div className="card p-3 mb-2">
      <div className="flex items-start gap-3">
        <span className={severityClass(alert.severity)}>
          {alert.severity}
        </span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs text-text-secondary uppercase tracking-wide">
              {alert.alert_type.replace(/_/g, ' ')}
            </span>
            <span className="text-[10px] text-text-dim">
              {timeAgo(alert.created_at)}
            </span>
          </div>
          <h3 className="text-sm text-text-primary mb-1">{alert.title}</h3>
          <p className="text-xs text-text-secondary leading-relaxed">{alert.description}</p>
          {alert.wallet_address && (
            <div className="mt-1 text-[10px] text-text-dim">
              {shortAddress(alert.wallet_address)}
            </div>
          )}
        </div>
        <div className="flex gap-2">
          {link && (
            <Link to={link} className="btn">View</Link>
          )}
          {onDismiss && (
            <button onClick={() => onDismiss(alert.id)} className="btn text-text-dim">
              ×
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
