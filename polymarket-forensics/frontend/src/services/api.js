const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function request(path, opts = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) },
    ...opts,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json();
}

export const api = {
  stats: () => request('/stats/overview'),
  recentTrades: (limit = 30) => request(`/stats/recent-trades?limit=${limit}`),

  wallets: ({ minScore = 0, classification, clusterId, order = 'insider_score', limit = 100, offset = 0 } = {}) => {
    const p = new URLSearchParams({ min_score: minScore, order, limit, offset });
    if (classification) p.set('classification', classification);
    if (clusterId) p.set('cluster_id', clusterId);
    return request(`/wallets?${p.toString()}`);
  },
  wallet: (addr) => request(`/wallets/${addr}`),
  walletTrades: (addr, limit = 100) => request(`/wallets/${addr}/trades?limit=${limit}`),
  walletFunding: (addr) => request(`/wallets/${addr}/funding`),
  traceWallet: (addr) => request(`/wallets/${addr}/trace`, { method: 'POST' }),
  rescoreWallet: (addr) => request(`/wallets/${addr}/rescore`, { method: 'POST' }),
  updateWallet: (addr, patch) => request(`/wallets/${addr}`, {
    method: 'PATCH', body: JSON.stringify(patch),
  }),

  clusters: (minProb = 0) => request(`/clusters?min_prob=${minProb}`),
  cluster: (id) => request(`/clusters/${id}`),
  clusterGraph: () => request('/clusters/graph/edges'),

  alerts: ({ severity, dismissed = false, limit = 100 } = {}) => {
    const p = new URLSearchParams({ dismissed, limit });
    if (severity) p.set('severity', severity);
    return request(`/alerts?${p.toString()}`);
  },
  dismissAlert: (id) => request(`/alerts/${id}/dismiss`, { method: 'POST' }),

  markets: ({ category, status, limit = 100 } = {}) => {
    const p = new URLSearchParams({ limit });
    if (category) p.set('category', category);
    if (status) p.set('status', status);
    return request(`/markets?${p.toString()}`);
  },
  marketForensics: (id) => request(`/markets/${id}/forensics`),
  marketTrades: (id, limit = 200) => request(`/markets/${id}/trades?limit=${limit}`),

  search: (q) => request(`/search?q=${encodeURIComponent(q)}`),
};
