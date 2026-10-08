/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: {
          DEFAULT: "#090D16",
          raised: "#0F172A",
          panel: "#111827",
          line: "#1E293B",
        },
        ink: {
          DEFAULT: "#E2E8F0",
          muted: "#94A3B8",
          faint: "#64748B",
        },
        accent: {
          DEFAULT: "#10B981",
          cyan: "#06B6D4",
          warn: "#F59E0B",
          danger: "#EF4444",
        },
      },
      fontFamily: {
        sans: [
          "IBM Plex Sans",
          "Segoe UI",
          "system-ui",
          "sans-serif",
        ],
        display: [
          "Space Grotesk",
          "IBM Plex Sans",
          "system-ui",
          "sans-serif",
        ],
      },
      boxShadow: {
        card: "0 0 0 1px rgba(30,41,59,0.9), 0 12px 40px rgba(0,0,0,0.35)",
      },
      animation: {
        "fade-in": "fadeIn 0.35s ease-out",
        "slide-up": "slideUp 0.35s ease-out",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%": { opacity: "0", transform: "translateY(12px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
    },
  },
  plugins: [],
};
