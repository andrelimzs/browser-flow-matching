import { defineConfig } from "vite";
import { resolve } from "node:path";

// GitHub Pages serves this project from https://<user>.github.io/browser-flow-matching/,
// so assets must be referenced under that subpath.
export default defineConfig({
  base: "/browser-flow-matching/",
  build: {
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, "index.html"),
        pusht: resolve(import.meta.dirname, "pusht/index.html"),
        viewer: resolve(import.meta.dirname, "pusht/viewer/index.html"),
      },
    },
  },
});
