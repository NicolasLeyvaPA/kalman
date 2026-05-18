/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: {
          primary: '#0a0a0a',
          card: '#111111',
          hover: '#1a1a1a',
        },
        border: {
          DEFAULT: '#222222',
          subtle: '#1a1a1a',
        },
        text: {
          primary: '#e0e0e0',
          secondary: '#888888',
          dim: '#555555',
        },
        neon: {
          green: '#00ff41',
          red: '#ff4444',
          yellow: '#ffaa00',
          cyan: '#22d3ee',
          orange: '#ff6b1c',
          purple: '#bc8cff',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s ease-in-out infinite',
        'flicker': 'flicker 0.15s infinite',
      },
      keyframes: {
        flicker: {
          '0%, 100%': { opacity: 1 },
          '50%': { opacity: 0.98 },
        },
      },
      boxShadow: {
        'neon-green': '0 0 12px rgba(0, 255, 65, 0.3)',
        'neon-red': '0 0 12px rgba(255, 68, 68, 0.3)',
        'neon-orange': '0 0 12px rgba(255, 107, 28, 0.3)',
      },
    },
  },
  plugins: [],
};
