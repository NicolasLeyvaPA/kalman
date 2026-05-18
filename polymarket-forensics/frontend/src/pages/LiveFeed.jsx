import { useEffect, useState } from 'react';
import { api } from '../services/api';
import { alertSocket } from '../services/websocket';
import AlertCard from '../components/AlertCard';
import TradeRow from '../components/TradeRow';
import StatCard from '../components/StatCard';
import { fmtUsd } from '../utils/formatting';

export default function LiveFeed() {
  const [stats, setStats] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [trades, setTrades] = useState([]);
  const [filter, setFilter] = useState('all');

  async function refresh() {
    try {
      const [s, a, t] = await Promise.all([
        api.stats(),
        api.alerts({ dismissed: false, limit: 50 }),
        api.recentTrades(40),
      ]);
      setStats(s);
      setAlerts(a);
      setTrades(t);
    } catch (e) {
      console.error(e);
    }
  }

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 15000);
    const unsub = alertSocket.subscribe((msg) => {
      if (msg.type === 'alert') {
        setAlerts(prev => [{
          id: msg.id,
          alert_type: msg.alert_type,
          severity: msg.severity,
          title: msg.title,
          description: msg.description,
          wallet_address: msg.wallet_address,
          cluster_id: msg.cluster_id,
          market_id: msg.market_id,
          data: msg.data,
          created_at: new Date().toISOString(),
        }, ...prev].slice(0, 100));
      }
    });
    return () => { clearInterval(id); unsub(); };
  }, []);

  async function dismiss(id) {
    try {
      await api.dismissAlert(id);
      setAlerts(prev => prev.filter(a => a.id !== id));
    } catch (e) { console.error(e); }
  }

  const filteredAlerts = filter === 'all'
    ? alerts
    : alerts.filter(a => a.severity === filter);

  return (
    <div className="space-y-6">
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
          <StatCard label="Wallets tracked" value={stats.wallets_total.toLocaleString()} />
          <StatCard label="Trades observed" value={stats.trades_total.toLocaleString()} />
          <StatCard label="24h volume" value={fmtUsd(stats.volume_24h)} accent="cyan" />
          <StatCard label="Large trades 24h" value={stats.large_trades_24h.toLocaleString()} accent="orange" />
          <StatCard label="Insider suspects" value={stats.insider_suspects.toLocaleString()} accent="red" />
          <StatCard label="Critical alerts" value={stats.critical_alerts.toLocaleString()}
                    accent={stats.critical_alerts > 0 ? 'red' : 'green'} />
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <section className="lg:col-span-2">
          <div className="flex items-center gap-2 mb-3">
            <h2 className="text-sm uppercase tracking-wider text-text-secondary">
              Alert feed
            </h2>
            <div className="flex gap-1 ml-auto text-[10px]">
              {['all', 'critical', 'high', 'medium', 'low'].map(f => (
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
          <div>
            {filteredAlerts.length === 0 && (
              <div className="card p-6 text-center text-text-dim text-xs">
                No alerts. The system is quiet — or it's still warming up.
              </div>
            )}
            {filteredAlerts.map(a => (
              <AlertCard key={a.id} alert={a} onDismiss={dismiss} />
            ))}
          </div>
        </section>

        <section>
          <h2 className="text-sm uppercase tracking-wider text-text-secondary mb-3">
            Recent large trades
          </h2>
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
            {trades.length === 0 && (
              <div className="text-text-dim text-xs p-4 text-center">
                Waiting for trades…
              </div>
            )}
            {trades.map(t => <TradeRow key={t.id} trade={t} />)}
          </div>
        </section>
      </div>
    </div>
  );
}
