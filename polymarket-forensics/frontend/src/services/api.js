/**
 * Backend API client.
 *
 * All HTTP calls go through ApiClient. Errors are typed (APIError /
 * RateLimitError / NetworkError) so callers can react meaningfully.
 * Every request supports an AbortSignal so callers can cancel on unmount.
 */

const DEFAULT_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const MAX_RETRIES = 2;

export class APIError extends Error {
  constructor(endpoint, status, body) {
    super(`${endpoint} → ${status}`);
    this.name = 'APIError';
    this.endpoint = endpoint;
    this.status = status;
    this.body = body;
  }
}

export class RateLimitError extends APIError {
  constructor(endpoint, retryAfterSec, body) {
    super(endpoint, 429, body);
    this.name = 'RateLimitError';
    this.retryAfter = retryAfterSec;
  }
}

export class NetworkError extends Error {
  constructor(endpoint, cause) {
    super(`${endpoint} → network error: ${cause?.message || cause}`);
    this.name = 'NetworkError';
    this.endpoint = endpoint;
    this.cause = cause;
  }
}

class ApiClient {
  constructor(baseUrl = DEFAULT_BASE) {
    this.baseUrl = baseUrl;
  }

  async _request(endpoint, { method = 'GET', body, signal, retries = MAX_RETRIES } = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    let attempt = 0;
    let lastError = null;

    while (attempt <= retries) {
      let response;
      try {
        response = await fetch(url, {
          method,
          signal,
          headers: { 'Content-Type': 'application/json' },
          body: body ? JSON.stringify(body) : undefined,
        });
      } catch (e) {
        if (e.name === 'AbortError') throw e;
        lastError = new NetworkError(endpoint, e);
        if (attempt < retries) {
          await new Promise((r) => setTimeout(r, 500 * Math.pow(2, attempt)));
          attempt += 1;
          continue;
        }
        throw lastError;
      }

      if (response.status === 429) {
        const retryAfter = Number(response.headers.get('Retry-After') || '5');
        throw new RateLimitError(endpoint, retryAfter, await response.text());
      }
      if (response.status >= 500 && attempt < retries) {
        await new Promise((r) => setTimeout(r, 500 * Math.pow(2, attempt)));
        attempt += 1;
        continue;
      }
      if (!response.ok) {
        throw new APIError(endpoint, response.status, await response.text());
      }
      return response.json();
    }

    throw lastError || new APIError(endpoint, 0, 'exhausted retries');
  }

  // ---- stats ----
  stats(signal) { return this._request('/stats/overview', { signal }); }
  recentTrades(limit = 30, signal) {
    return this._request(`/stats/recent-trades?limit=${limit}`, { signal });
  }
  schedulerStatus(signal) { return this._request('/stats/scheduler', { signal }); }

  // ---- wallets ----
  wallets({ minScore = 0, classification, clusterId, order = 'insider_score',
            limit = 100, offset = 0 } = {}, signal) {
    const p = new URLSearchParams({ min_score: minScore, order, limit, offset });
    if (classification) p.set('classification', classification);
    if (clusterId) p.set('cluster_id', clusterId);
    return this._request(`/wallets?${p.toString()}`, { signal });
  }
  wallet(address, signal) { return this._request(`/wallets/${address}`, { signal }); }
  walletTrades(address, limit = 100, signal) {
    return this._request(`/wallets/${address}/trades?limit=${limit}`, { signal });
  }
  walletFunding(address, signal) {
    return this._request(`/wallets/${address}/funding`, { signal });
  }
  traceWallet(address) {
    return this._request(`/wallets/${address}/trace`, { method: 'POST', retries: 0 });
  }
  rescoreWallet(address) {
    return this._request(`/wallets/${address}/rescore`, { method: 'POST', retries: 0 });
  }
  updateWallet(address, patch) {
    return this._request(`/wallets/${address}`, {
      method: 'PATCH', body: patch, retries: 0,
    });
  }

  // ---- clusters ----
  clusters(minProb = 0, signal) {
    return this._request(`/clusters?min_prob=${minProb}`, { signal });
  }
  cluster(id, signal) { return this._request(`/clusters/${id}`, { signal }); }
  clusterGraph(signal) { return this._request('/clusters/graph/edges', { signal }); }

  // ---- alerts ----
  alerts({ severity, dismissed = false, limit = 100, offset = 0 } = {}, signal) {
    const p = new URLSearchParams({ dismissed, limit, offset });
    if (severity) p.set('severity', severity);
    return this._request(`/alerts?${p.toString()}`, { signal });
  }
  dismissAlert(id) {
    return this._request(`/alerts/${id}/dismiss`, { method: 'POST', retries: 0 });
  }

  // ---- markets ----
  markets({ category, status, limit = 100 } = {}, signal) {
    const p = new URLSearchParams({ limit });
    if (category) p.set('category', category);
    if (status) p.set('status', status);
    return this._request(`/markets?${p.toString()}`, { signal });
  }
  marketForensics(id, signal) {
    return this._request(`/markets/${id}/forensics`, { signal });
  }
  marketTrades(id, limit = 200, signal) {
    return this._request(`/markets/${id}/trades?limit=${limit}`, { signal });
  }

  // ---- search ----
  search(q, signal) {
    return this._request(`/search?q=${encodeURIComponent(q)}`, { signal });
  }

  // ---- export ----
  exportUrl(path) { return `${this.baseUrl}${path}`; }
}

export const api = new ApiClient();
