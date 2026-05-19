import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';
import { useDebouncedValue } from '../hooks/useDebouncedValue';

export default function SearchBar() {
  const [q, setQ] = useState('');
  const debounced = useDebouncedValue(q, 200);
  const [results, setResults] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    if (debounced.length < 2) {
      setResults(null);
      return undefined;
    }
    const ctrl = new AbortController();
    api.search(debounced, ctrl.signal)
      .then(setResults)
      .catch((e) => {
        if (e.name !== 'AbortError') setResults(null);
      });
    return () => ctrl.abort();
  }, [debounced]);

  const go = (path) => {
    setQ('');
    setResults(null);
    navigate(path);
  };

  const hasResults = results && (results.wallets.length || results.markets.length);

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
      {hasResults && (
        <div className="absolute top-full mt-1 w-full bg-bg-card border border-border rounded
                       shadow-lg z-50 max-h-96 overflow-auto">
          {results.wallets.map((w) => (
            <button
              key={w.address}
              onClick={() => go(`/wallet/${w.address}`)}
              className="block w-full text-left px-3 py-2 hover:bg-bg-hover
                         border-b border-border-subtle"
            >
              <div className="text-xs text-text-primary">
                {w.ens_name || w.address}
              </div>
              <div className="text-[10px] text-text-secondary">
                insider {Number(w.insider_score).toFixed(2)} · {w.classification}
              </div>
            </button>
          ))}
          {results.markets.map((m) => (
            <button
              key={m.id}
              onClick={() => go(`/market/${m.id}`)}
              className="block w-full text-left px-3 py-2 hover:bg-bg-hover
                         border-b border-border-subtle"
            >
              <div className="text-xs text-text-primary truncate">{m.question}</div>
              <div className="text-[10px] text-text-secondary">
                {m.category} · {m.status}
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
