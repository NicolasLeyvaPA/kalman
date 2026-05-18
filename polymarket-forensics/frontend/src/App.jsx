import { Link, NavLink, Route, Routes, useNavigate } from 'react-router-dom';
import { useEffect, useState } from 'react';
import LiveFeed from './pages/LiveFeed';
import WalletExplorer from './pages/WalletExplorer';
import ClusterMap from './pages/ClusterMap';
import MarketIntel from './pages/MarketIntel';
import Leaderboard from './pages/Leaderboard';
import Settings from './pages/Settings';
import { api } from './services/api';

function SearchBar() {
  const [q, setQ] = useState('');
  const [results, setResults] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    if (q.length < 2) {
      setResults(null);
      return;
    }
    const t = setTimeout(async () => {
      try {
        setResults(await api.search(q));
      } catch {
        setResults(null);
      }
    }, 200);
    return () => clearTimeout(t);
  }, [q]);

  const go = (path) => {
    setQ('');
    setResults(null);
    navigate(path);
  };

  return (
    <div className="relative w-96">
      <input
        type="text"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="search wallet, market, ENS…"
        className="w-full px-3 py-1.5 bg-bg-card border border-border rounded text-xs
                   focus:outline-none focus:border-neon-orange"
      />
      {results && (results.wallets.length || results.markets.length) > 0 && (
        <div className="absolute top-full mt-1 w-full bg-bg-card border border-border rounded
                       shadow-lg z-50 max-h-96 overflow-auto">
          {results.wallets.map(w => (
            <button
              key={w.address}
              onClick={() => go(`/wallet/${w.address}`)}
              className="block w-full text-left px-3 py-2 hover:bg-bg-hover border-b border-border-subtle"
            >
              <div className="text-xs text-text-primary">{w.ens_name || w.address}</div>
              <div className="text-[10px] text-text-secondary">
                insider {w.insider_score?.toFixed(2)} · {w.classification}
              </div>
            </button>
          ))}
          {results.markets.map(m => (
            <button
              key={m.id}
              onClick={() => go(`/market/${m.id}`)}
              className="block w-full text-left px-3 py-2 hover:bg-bg-hover border-b border-border-subtle"
            >
              <div className="text-xs text-text-primary truncate">{m.question}</div>
              <div className="text-[10px] text-text-secondary">{m.category} · {m.status}</div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function Nav() {
  const link = ({ isActive }) =>
    `px-3 py-1 text-xs uppercase tracking-wider rounded transition-colors ${
      isActive
        ? 'bg-neon-orange/10 text-neon-orange border border-neon-orange/40'
        : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover border border-transparent'
    }`;

  return (
    <nav className="flex items-center gap-1">
      <NavLink to="/" end className={link}>Live</NavLink>
      <NavLink to="/leaderboard" className={link}>Leaderboard</NavLink>
      <NavLink to="/clusters" className={link}>Clusters</NavLink>
      <NavLink to="/markets" className={link}>Markets</NavLink>
      <NavLink to="/settings" className={link}>Settings</NavLink>
    </nav>
  );
}

export default function App() {
  return (
    <div className="min-h-screen bg-bg-primary text-text-primary">
      <header className="sticky top-0 z-40 border-b border-border bg-bg-primary/95 backdrop-blur">
        <div className="max-w-[1600px] mx-auto px-4 py-2 flex items-center gap-6">
          <Link to="/" className="text-sm font-bold tracking-wider text-neon-orange">
            POLYMARKET FORENSICS
          </Link>
          <span className="text-[10px] text-text-secondary flex items-center gap-2">
            <span className="live-dot" />
            LIVE
          </span>
          <Nav />
          <div className="ml-auto">
            <SearchBar />
          </div>
        </div>
      </header>

      <main className="max-w-[1600px] mx-auto px-4 py-6">
        <Routes>
          <Route path="/" element={<LiveFeed />} />
          <Route path="/leaderboard" element={<Leaderboard />} />
          <Route path="/clusters" element={<ClusterMap />} />
          <Route path="/markets" element={<MarketIntel />} />
          <Route path="/market/:id" element={<MarketIntel />} />
          <Route path="/wallet/:address" element={<WalletExplorer />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </main>

      <footer className="border-t border-border mt-12 py-3">
        <div className="max-w-[1600px] mx-auto px-4 text-[10px] text-text-dim text-center">
          Personal analysis tool · Do not publish wallet attributions without legal review
        </div>
      </footer>
    </div>
  );
}
