import { useCallback, useEffect, useState } from 'react';
import { api } from '../services/api';
import { alertSocket } from '../services/websocket';

/**
 * Stream of open alerts. WebSocket is the source of truth; we backfill
 * with one REST call on mount and on reconnect.
 */
export function useAlerts({ severity = null, max = 100 } = {}) {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState('idle');

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.alerts({ dismissed: false, limit: max, severity });
      setAlerts(res.alerts || []);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }, [max, severity]);

  useEffect(() => { refresh(); }, [refresh]);

  useEffect(() => {
    const unsub = alertSocket.subscribe((msg) => {
      if (msg?.type !== 'alert') return;
      setAlerts((prev) => {
        if (severity && msg.severity !== severity) return prev;
        return [{
          id: msg.id,
          alert_type: msg.alert_type,
          severity: msg.severity,
          title: msg.title,
          description: msg.description,
          wallet_address: msg.wallet_address,
          cluster_id: msg.cluster_id,
          market_id: msg.market_id,
          data: msg.data,
          dismissed: false,
          created_at: msg.created_at || new Date().toISOString(),
        }, ...prev].slice(0, max);
      });
    });
    return unsub;
  }, [severity, max]);

  useEffect(() => {
    return alertSocket.subscribeStatus((s) => {
      setStatus(s);
      if (s === 'open') refresh();
    });
  }, [refresh]);

  const dismiss = useCallback(async (id) => {
    setAlerts((prev) => prev.filter((a) => a.id !== id));
    try {
      await api.dismissAlert(id);
    } catch {
      refresh();
    }
  }, [refresh]);

  return { alerts, loading, error, status, dismiss, refresh };
}
