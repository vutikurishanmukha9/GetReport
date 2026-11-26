import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";


// https://vitejs.dev/config/
export default defineConfig(() => ({
  server: {
    host: "::",
    port: 8080,
    hmr: {
      overlay: false,
    },
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        ws: true,
      },
    },
  },
  plugins: [react()].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("node_modules")) {
            if (id.includes("echarts") || id.includes("zrender")) {
              return "vendor-echarts";
            }
            if (id.includes("@codemirror") || id.includes("@lezer") || id.includes("@uiw")) {
              return "vendor-codemirror";
            }
            if (id.includes("@radix-ui")) {
              return "vendor-radix";
            }
            if (id.includes("framer-motion") || id.includes("\\motion\\") || id.includes("/motion/")) {
              return "vendor-motion";
            }
            if (id.includes("d3")) {
              return "vendor-charts";
            }
            if (id.includes("xlsx") || id.includes("papaparse")) {
              return "vendor-parsers";
            }
            if (id.includes("lucide-react")) {
              return "vendor-icons";
            }
            return "vendor";
          }
        },
      },
    },
  },
}));
