import { defineConfig } from "vite";

// GitHub Pages serves this project from https://<user>.github.io/browser-flow-matching/,
// so assets must be referenced under that subpath.
export default defineConfig({
  base: "/browser-flow-matching/",
});
