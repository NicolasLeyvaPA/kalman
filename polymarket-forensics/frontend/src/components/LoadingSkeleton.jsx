export function LoadingSkeleton({ lines = 4, className = '' }) {
  return (
    <div className={`card p-4 ${className}`}>
      <div className="space-y-2">
        {Array.from({ length: lines }).map((_, i) => (
          <div key={i}
               className="h-3 bg-bg-hover rounded animate-pulse"
               style={{ width: `${100 - i * 12}%` }} />
        ))}
      </div>
    </div>
  );
}

export function TableSkeleton({ rows = 8 }) {
  return (
    <div className="card p-2">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i}
             className="h-5 my-1 bg-bg-hover rounded animate-pulse"
             style={{ width: `${85 + (i % 3) * 5}%` }} />
      ))}
    </div>
  );
}
