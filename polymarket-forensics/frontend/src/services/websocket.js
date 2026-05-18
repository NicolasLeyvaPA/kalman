/**
 * WebSocket alert feed with reconnect, heartbeat, and connection-status events.
 */

const WS_URL = (import.meta.env.VITE_WS_URL || 'ws://localhost:8000') + '/ws';
const HEARTBEAT_MS = 30000;
const MAX_BACKOFF_MS = 30000;

class AlertSocket {
  constructor() {
    this.handlers = new Set();
    this.statusHandlers = new Set();
    this.ws = null;
    this.retry = 0;
    this.lastSeen = 0;
    this.heartbeatTimer = null;
    this.status = 'idle';
  }

  /** Subscribe to incoming alert messages. Returns an unsubscribe fn. */
  subscribe(handler) {
    this.handlers.add(handler);
    if (!this.ws) this._connect();
    return () => this.handlers.delete(handler);
  }

  /** Subscribe to connection status changes ('connecting' | 'open' | 'closed'). */
  subscribeStatus(handler) {
    this.statusHandlers.add(handler);
    handler(this.status);
    return () => this.statusHandlers.delete(handler);
  }

  _setStatus(s) {
    if (this.status === s) return;
    this.status = s;
    for (const h of this.statusHandlers) h(s);
  }

  _connect() {
    this._setStatus('connecting');
    try {
      this.ws = new WebSocket(WS_URL);
    } catch {
      this._scheduleReconnect();
      return;
    }
    this.ws.onopen = () => {
      this.retry = 0;
      this.lastSeen = Date.now();
      this._setStatus('open');
      this._startHeartbeat();
    };
    this.ws.onmessage = (ev) => {
      this.lastSeen = Date.now();
      let payload;
      try { payload = JSON.parse(ev.data); } catch { return; }
      if (payload?.type === 'ping') return;
      for (const h of this.handlers) h(payload);
    };
    this.ws.onclose = () => {
      this._stopHeartbeat();
      this.ws = null;
      this._setStatus('closed');
      this._scheduleReconnect();
    };
    this.ws.onerror = () => {
      try { this.ws?.close(); } catch { /* noop */ }
    };
  }

  _startHeartbeat() {
    this._stopHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      if (Date.now() - this.lastSeen > HEARTBEAT_MS * 2) {
        try { this.ws?.close(); } catch { /* noop */ }
      }
    }, HEARTBEAT_MS);
  }

  _stopHeartbeat() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  _scheduleReconnect() {
    const delay = Math.min(MAX_BACKOFF_MS, 1000 * Math.pow(2, this.retry++));
    setTimeout(() => this._connect(), delay);
  }
}

export const alertSocket = new AlertSocket();
