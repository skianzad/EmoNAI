import { defineConfig } from "vite";
import { reactApi } from "./server/react-api.js";

export default defineConfig({
  plugins: [reactApi()],
  server: { port: 5174, host: true },
});
