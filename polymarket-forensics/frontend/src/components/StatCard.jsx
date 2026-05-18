export default function StatCard({ label, value, sublabel, accent }) {
  const accentClass = {
    red: 'text-neon-red',
    orange: 'text-neon-orange',
    yellow: 'text-neon-yellow',
    green: 'text-neon-green',
    cyan: 'text-neon-cyan',
  }[accent] || 'text-text-primary';

  return (
    <div className="card p-3">
      <div className="text-[10px] text-text-dim uppercase tracking-wider mb-1">{label}</div>
      <div className={`text-xl font-semibold ${accentClass}`}>{value}</div>
      {sublabel && (
        <div className="text-[10px] text-text-secondary mt-1">{sublabel}</div>
      )}
    </div>
  );
}
