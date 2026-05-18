import { scoreColor } from '../utils/colors';

export default function InsiderScoreBar({ score, width = 'w-32' }) {
  const pct = Math.min(100, Math.max(0, (Number(score) || 0) * 100));
  const color = scoreColor(score);
  return (
    <div className={`${width} h-2 bg-bg-hover rounded overflow-hidden`}>
      <div
        className="h-full transition-all"
        style={{ width: `${pct}%`, background: color }}
      />
    </div>
  );
}
