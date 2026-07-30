import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Music Trail — front-end design demo. No backend, no real audio.
export default defineConfig({
  plugins: [react()],
  server: {
    port: process.env.PORT ? Number(process.env.PORT) : 5273,
    strictPort: false,
    open: false,
  },
})
