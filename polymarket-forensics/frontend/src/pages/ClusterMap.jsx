import { useEffect, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { api } from '../services/api';
import BubbleGraph from '../components/BubbleGraph';
import { fmtPct, fmtUsd, shortAddress } from '../utils/formatting';
import { scoreClass } from '../utils/colors';

export default function ClusterMap() {
  const [graph, setGraph] = useState(null);
  const [clusters, setClusters] = useState([]);
  const [minProb, setMinProb] = useState(0.0);
  const [params] = useSearchParams();
  const focus = params.get('focus');

  useEffect(() => {
    async function load() {
      try {
        const [g, c] = await Promise.all([
          api.clusterGraph(),
          api.clusters(minProb),
        ]);
        setGraph(g);
        setClusters(c);
      } catch (e) { console.error(e); }
    }
    load();
  }, [minProb]);

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-sm uppercase tracking-wider text-text-secondary">
            Cluster map
          </h1>
          <p className="text-[10px] text-text-dim mt-1 max-w-2xl">
            Wallets grouped by shared funding sources, coordinated trading windows,
            or behavioral similarity. Size = total volume. Color = insider score.
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <label className="text-text-secondary">Min cluster probability</label>
          <input
            type="range" min="0" max="1" step="0.05"
            value={minProb}
            onChange={(e) => setMinProb(parseFloat(e.target.value))}
            className="w-32"
          />
          <span className="text-text-primary tabular-nums">{minProb.toFixed(2)}</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          {graph && graph.nodes.length > 0 ? (
            <BubbleGraph nodes={graph.nodes} edges={graph.edges} />
          ) : (
            <div className="card p-12 text-center text-text-dim text-xs">
              No clustered wallets yet. The detector runs every 15 minutes.
            </div>
          )}
        </div>
        <aside className="space-y-2">
          <h2 className="text-xs uppercase tracking-wider text-text-secondary mb-2">
            Top clusters
          </h2>
          {clusters.length === 0 && (
            <div className="card p-4 text-text-dim text-xs">No clusters yet.</div>
          )}
          {clusters.slice(0, 30).map(c => (
            <div key={c.id}
                 className={`card p-3 ${focus === c.id ? 'border-neon-orange' : ''}`}>
              <div className="flex justify-between items-start gap-2 mb-1">
                <span className="text-xs font-bold text-neon-orange">{c.id}</span>
                <span className={`text-xs tabular-nums ${scoreClass(c.insider_probability)}`}>
                  {Number(c.insider_probability).toFixed(2)}
                </span>
              </div>
              <div className="text-[10px] text-text-secondary mb-2">
                {c.cluster_type} · {c.wallets.length} wallets
              </div>
              <div className="text-[10px] text-text-dim italic mb-2">
                {c.evidence}
              </div>
              <div className="flex flex-wrap gap-1">
                {c.wallets.slice(0, 4).map(w => (
                  <Link key={w} to={`/wallet/${w}`}
                        className="text-[10px] text-text-secondary hover:text-neon-orange">
                    {shortAddress(w, 4, 3)}
                  </Link>
                ))}
                {c.wallets.length > 4 && (
                  <span className="text-[10px] text-text-dim">
                    +{c.wallets.length - 4}
                  </span>
                )}
              </div>
            </div>
          ))}
        </aside>
      </div>
    </div>
  );
}
