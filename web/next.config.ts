import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The Playwright fixtures build uses its own output dir so it never clobbers `npm run build`.
  distDir: process.env.NEXT_DIST_DIR ?? ".next",
};

export default nextConfig;
