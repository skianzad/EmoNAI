import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  output: "standalone",
  outputFileTracingRoot: path.join(__dirname),
  experimental: {
    // Segment layers + empty background are large data URLs on first save.
    serverActions: {
      bodySizeLimit: "32mb",
    },
  },
};

export default nextConfig;
