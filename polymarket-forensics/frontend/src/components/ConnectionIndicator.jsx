import { useEffect, useState } from 'react';
import { alertSocket } from '../services/websocket';

export default function ConnectionIndicator() {
  const [status, setStatus] = useState('idle');
  useEffect(() => alertSocket.subscribeStatus(setStatus), []);

  const dotClass = status === 'open'
    ? 'bg-neon-green shadow-neon-green'
    : status === 'connecting'
    ? 'bg-neon-yellow shadow-neon-orange animate-pulse'
    : 'bg-neon-red shadow-neon-red';

  const label = status === 'open' ? 'LIVE'
              : status === 'connecting' ? 'CONNECTING'
              : status === 'closed' ? 'OFFLINE'
              : 'IDLE';

  return (
    <span className="flex items-center gap-2 text-[10px] uppercase tracking-wider">
      <span className={`inline-block w-2 h-2 rounded-full ${dotClass}`}
            style={{ boxShadow: status === 'open' ? '0 0 8px currentColor' : 'none' }} />
      <span className="text-text-secondary">{label}</span>
    </span>
  );
}
