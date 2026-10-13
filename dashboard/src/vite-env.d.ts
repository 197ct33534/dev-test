/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** API origin, e.g. https://score.example.com. Empty → same origin / dev proxy. */
  readonly VITE_API_BASE?: string;
  /** "true" → render bundled mock data instead of calling the API. */
  readonly VITE_USE_MOCK?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
