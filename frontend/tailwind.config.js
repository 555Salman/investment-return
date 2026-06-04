/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        primary:  '#1e40af',
        accent:   '#0ea5e9',
        surface:  '#1e293b',
        muted:    '#94a3b8',
      },
    },
  },
  plugins: [],
}
