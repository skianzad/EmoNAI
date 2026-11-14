import { defineConfig } from "vite";
import { sketchApi } from "./server/api.js";

export default defineConfig({
  plugins: [sketchApi()],
  server: { port: 5175, host: true },
});
