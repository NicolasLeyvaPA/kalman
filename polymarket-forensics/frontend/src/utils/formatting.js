export const shortAddress = (a, head = 6, tail = 4) => {
  if (!a) return '';
  if (a.length <= head + tail + 3) return a;
  return `${a.slice(0, head)}…${a.slice(-tail)}`;
};

export const fmtUsd = (v) => {
  if (v == null || isNaN(v)) return '—';
  const n = Number(v);
  if (Math.abs(n) >= 1_000_000) return `$${(n / 1_000_000).toFixed(2)}M`;
  if (Math.abs(n) >= 1_000) return `$${(n / 1_000).toFixed(1)}K`;
  return `$${n.toFixed(2)}`;
};

export const fmtPct = (v, digits = 1) => {
  if (v == null || isNaN(v)) return '—';
  return `${(Number(v) * 100).toFixed(digits)}%`;
};

export const fmtScore = (v) => {
  if (v == null || isNaN(v)) return '—';
  return Number(v).toFixed(2);
};

export const timeAgo = (iso) => {
  if (!iso) return '—';
  const d = new Date(iso);
  const sec = Math.max(1, Math.floor((Date.now() - d.getTime()) / 1000));
  if (sec < 60) return `${sec}s ago`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const day = Math.floor(hr / 24);
  return `${day}d ago`;
};
