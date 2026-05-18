export const scoreColor = (score) => {
  const s = Number(score) || 0;
  if (s >= 0.7) return '#ff4444';
  if (s >= 0.5) return '#ff6b1c';
  if (s >= 0.3) return '#ffaa00';
  return '#00ff41';
};

export const scoreClass = (score) => {
  const s = Number(score) || 0;
  if (s >= 0.7) return 'text-neon-red';
  if (s >= 0.5) return 'text-neon-orange';
  if (s >= 0.3) return 'text-neon-yellow';
  return 'text-neon-green';
};

export const severityClass = (sev) => {
  switch (sev) {
    case 'critical': return 'pill-red';
    case 'high':     return 'pill-orange';
    case 'medium':   return 'pill-yellow';
    default:         return 'pill-cyan';
  }
};

export const classificationPill = (cls) => {
  switch (cls) {
    case 'insider_suspect':    return 'pill-red';
    case 'confirmed_insider':  return 'pill-red';
    case 'suspicious':         return 'pill-orange';
    case 'watch':              return 'pill-yellow';
    case 'smart':              return 'pill-cyan';
    case 'normal':             return 'pill-green';
    default:                   return 'pill-purple';
  }
};
