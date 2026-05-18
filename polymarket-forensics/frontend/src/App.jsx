import { Link, NavLink, Route, Routes } from 'react-router-dom';
import ErrorBoundary from './components/ErrorBoundary';
import ConnectionIndicator from './components/ConnectionIndicator';
import SearchBar from './components/SearchBar';
import LiveFeed from './pages/LiveFeed';
import WalletExplorer from './pages/WalletExplorer';
import ClusterMap from './pages/ClusterMap';
import MarketIntel from './pages/MarketIntel';
import Leaderboard from './pages/Leaderboard';
import Settings from './pages/Settings';

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
          <ConnectionIndicator />
          <Nav />
          <div className="ml-auto">
            <SearchBar />
          </div>
        </div>
      </header>

      <main className="max-w-[1600px] mx-auto px-4 py-6">
        <ErrorBoundary>
          <Routes>
            <Route path="/" element={<LiveFeed />} />
            <Route path="/leaderboard" element={<Leaderboard />} />
            <Route path="/clusters" element={<ClusterMap />} />
            <Route path="/markets" element={<MarketIntel />} />
            <Route path="/market/:id" element={<MarketIntel />} />
            <Route path="/wallet/:address" element={<WalletExplorer />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </ErrorBoundary>
      </main>

      <footer className="border-t border-border mt-12 py-3">
        <div className="max-w-[1600px] mx-auto px-4 text-[10px] text-text-dim text-center">
          Personal analysis tool · Do not publish wallet attributions without legal review
        </div>
      </footer>
    </div>
  );
}
