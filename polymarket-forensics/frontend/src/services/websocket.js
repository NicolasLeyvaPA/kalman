const WS_URL = (import.meta.env.VITE_WS_URL || 'ws://localhost:8000') + '/ws';

class AlertSocket {
  constructor() {
    this.handlers = new Set();
    this.ws = null;
    this.retry = 0;
  }

  subscribe(handler) {
    this.handlers.add(handler);
    if (!this.ws) this._connect();
    return () => this.handlers.delete(handler);
  }

  _connect() {
    try {
      this.ws = new WebSocket(WS_URL);
    } catch (e) {
      this._scheduleReconnect();
      return;
    }
    this.ws.onmessage = (ev) => {
      try {
        const payload = JSON.parse(ev.data);
        for (const h of this.handlers) h(payload);
      } catch {}
    };
    this.ws.onopen = () => { this.retry = 0; };
    this.ws.onclose = () => { this.ws = null; this._scheduleReconnect(); };
    this.ws.onerror = () => { try { this.ws.close(); } catch {} };
  }

  _scheduleReconnect() {
    const delay = Math.min(30000, 1000 * Math.pow(2, this.retry++));
    setTimeout(() => this._connect(), delay);
  }
}

export const alertSocket = new AlertSocket();
