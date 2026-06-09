// Render an animated SVG (CSS @keyframes or SMIL) to PNG frames using the
// system Chrome, stepping the animation timeline deterministically so the
// captured loop is seamless and frame-accurate.
//
// Usage:
//   node svg2anim.mjs <input.svg> <outFramesDir> [durationMs=9000] [fps=20] [size=1080]
//
// Chrome binary: set CHROME_PATH to override; otherwise common macOS/Linux
// locations are probed. Frames are written as f_0000.png, f_0001.png, ...

import puppeteer from "puppeteer-core";
import { readFileSync, mkdirSync, rmSync, existsSync } from "node:fs";
import { resolve } from "node:path";

function findChrome() {
  if (process.env.CHROME_PATH) return process.env.CHROME_PATH;
  const candidates = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
  ];
  const hit = candidates.find((p) => existsSync(p));
  if (!hit) {
    console.error(
      "No Chrome found. Install Google Chrome or set CHROME_PATH=/path/to/chrome."
    );
    process.exit(1);
  }
  return hit;
}

const [, , svgArg, outArg, durArg, fpsArg, sizeArg] = process.argv;
if (!svgArg || !outArg) {
  console.error("usage: node svg2anim.mjs <input.svg> <outDir> [durationMs] [fps] [size]");
  process.exit(1);
}
const svgPath = resolve(svgArg);
const outDir = resolve(outArg);
const durationMs = Number(durArg ?? 9000);
const fps = Number(fpsArg ?? 20);
const size = Number(sizeArg ?? 1080);
const frames = Math.round((durationMs / 1000) * fps);

const svg = readFileSync(svgPath, "utf8");
const html = `<!doctype html><html><head><meta charset="utf8"><style>
  html,body{margin:0;padding:0;background:#0d1117}
  svg{display:block;width:${size}px;height:${size}px}
</style></head><body>${svg}</body></html>`;

rmSync(outDir, { recursive: true, force: true });
mkdirSync(outDir, { recursive: true });

const browser = await puppeteer.launch({
  executablePath: findChrome(),
  headless: true,
  args: ["--force-device-scale-factor=1", "--hide-scrollbars", "--no-sandbox"],
});
const page = await browser.newPage();
await page.setViewport({ width: size, height: size, deviceScaleFactor: 1 });
await page.setContent(html, { waitUntil: "load" });

// Let the animations register on the timeline, then pause them so we drive
// time by hand instead of relying on jittery real-time screencast frames.
await page.evaluate(
  () =>
    new Promise((r) =>
      requestAnimationFrame(() =>
        requestAnimationFrame(() => {
          document.getAnimations().forEach((a) => a.pause());
          r();
        })
      )
    )
);

for (let i = 0; i < frames; i++) {
  // Sample across [0, durationMs) — excluding the endpoint avoids a duplicated
  // frame at the loop seam, so the GIF/MP4 loops cleanly.
  const t = (i / frames) * durationMs;
  await page.evaluate((t) => {
    document.getAnimations().forEach((a) => {
      a.currentTime = t;
    });
  }, t);
  const n = String(i).padStart(4, "0");
  await page.screenshot({ path: `${outDir}/f_${n}.png` });
}

await browser.close();
console.log(`wrote ${frames} frames (${fps}fps, ${durationMs}ms loop, ${size}px) -> ${outDir}`);
