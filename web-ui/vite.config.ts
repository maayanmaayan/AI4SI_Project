import { defineConfig } from 'vite'
import type { UserConfigExport } from 'vitest/config'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
}) as UserConfigExport & { test: { environment: string; globals: boolean } }

export const test = {
  environment: 'jsdom',
  globals: true,
}
