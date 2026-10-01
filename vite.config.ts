import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Permit this Codespace's exact forwarded hostname, while retaining host checks.
const codespaceHost = process.env.CODESPACE_NAME
  ? `${process.env.CODESPACE_NAME}-5173.${process.env.GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN || 'app.github.dev'}`
  : undefined;

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
    allowedHosts: codespaceHost ? [codespaceHost] : [],
  },
});
