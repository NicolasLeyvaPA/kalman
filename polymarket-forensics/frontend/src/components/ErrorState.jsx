export function ErrorState({ error, onRetry, compact = false }) {
  const msg = error?.message || String(error || 'Unknown error');
  const status = error?.status;
  return (
    <div className={`card border-neon-red/40 ${compact ? 'p-3' : 'p-6'}`}>
      <div className="text-xs uppercase tracking-wider text-neon-red mb-1">
        Error{status ? ` · ${status}` : ''}
      </div>
      <div className="text-xs text-text-secondary break-all mb-3">{msg}</div>
      {onRetry && (
        <button onClick={onRetry} className="btn">Retry</button>
      )}
    </div>
  );
}

export function EmptyState({ message = 'No results.', children }) {
  return (
    <div className="card p-6 text-center text-xs text-text-dim">
      <div>{message}</div>
      {children && <div className="mt-2">{children}</div>}
    </div>
  );
}
