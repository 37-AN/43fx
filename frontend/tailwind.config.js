/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        panel: '#0a0a0a',
        neon: '#39FF14',
        danger: '#FF0000',
        muted: '#6b7280',
      },
      fontFamily: {
        mono: ['IBM Plex Mono', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      boxShadow: {
        neon: '0 0 20px rgba(57,255,20,0.2)',
      },
      backgroundImage: {
        grid: 'linear-gradient(rgba(57,255,20,0.08) 1px, transparent 1px), linear-gradient(90deg, rgba(57,255,20,0.08) 1px, transparent 1px)',
      },
    },
  },
  plugins: [],
};
