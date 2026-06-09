---
name: animated-svg-to-gif
description: Convert an animated SVG (CSS @keyframes or SMIL) into a looping GIF and/or MP4 for social posts (LinkedIn, X, etc.). Use whenever the user wants to turn an animated/motion SVG into a GIF, MP4, or video — phrasings like "svg to gif", "make a gif from this svg", "animate this svg for LinkedIn", "render the svg animation as a video", or "post this svg as a gif". Reach for this even when the user only says "convert with ffmpeg": plain ffmpeg CANNOT do it (no SVG decoder / no animation timeline), so this skill's headless-Chrome capture step is required.
---

Turn an animated SVG into a clean, looping GIF and/or MP4.

## Why ffmpeg alone fails (read this first)

Two independent reasons, both worth saying out loud so the user understands the extra step:

1. **No decoder.** Most ffmpeg builds (incl. Homebrew's) have no `librsvg`, so they can't read SVG at all.
2. **No timeline.** Even *with* librsvg, ffmpeg/rsvg only rasterize a single static frame — they don't execute a CSS `@keyframes` / SMIL animation over time.

So the motion you see in a browser can only be captured *by* a browser. This skill drives **headless Chrome** to step the animation's timeline frame-by-frame, then hands the frames to ffmpeg to encode.

Stepping the timeline deterministically (pausing every animation and setting its `currentTime`) — rather than real-time screen-recording — gives perfectly even frames and a **seamless loop** with no hitch at the wrap point.

## Requirements

- **Google Chrome** (or Chromium) installed. Auto-detected; override with `--chrome /path` or `CHROME_PATH`.
- **ffmpeg** on PATH (`brew install ffmpeg`). It does *not* need librsvg.
- **node** + **npm** — the wrapper installs `puppeteer-core` into `scripts/` on first run (a few MB, no bundled browser).

## Usage

One command does capture + encode. From this skill's directory:

```bash
scripts/svg2video.sh <input.svg> [options]
```

Defaults emit both `<stem>.gif` (720px) and `<stem>.mp4` (1080px) next to the input.

| Option | Default | Notes |
|---|---|---|
| `--duration MS` | `9000` | **Set this to the SVG's loop length.** A wrong value clips or mistimes the loop. |
| `--fps N` | `20` | 25–30 = smoother motion, bigger files. |
| `--out BASE` | input stem | Output path without extension. |
| `--gif-size PX` | `720` | `0` skips the GIF. |
| `--mp4-size PX` | `1080` | `0` skips the MP4. |
| `--keep-frames` | off | Keep the intermediate PNGs. |
| `--chrome PATH` | auto | Chrome binary override. |

**Example** (the original use case):

```bash
scripts/svg2video.sh ~/Downloads/agent-memory.svg --duration 9000 --fps 20
```

### Finding the loop duration

`--duration` must match the SVG's longest animation cycle, or the loop will be clipped. Grep the SVG for the timing:

```bash
grep -oE 'animation[^;"}]*[0-9.]+s' input.svg   # e.g. "animation: reveal 9s ..." -> --duration 9000
```

Use the largest cycle length you find. For SMIL SVGs, look at `dur=` on `<animate>` elements instead.

## For LinkedIn / social, recommend the MP4

LinkedIn (and most platforms) accept GIFs but treat them poorly — they often downgrade quality or fall back to a static frame, and GIF's 256-color palette bands on dark gradients. The MP4 is sharper, several times smaller, and posts as native video. Deliver both, but tell the user the MP4 is the better upload. The skill produces a square 1:1 loop, ideal for a feed post.

## Internals

- `scripts/svg2anim.mjs` — headless-Chrome capture: loads the SVG at a fixed square viewport, pauses all animations, steps `currentTime` across `[0, duration)`, screenshots each frame.
- `scripts/svg2video.sh` — orchestrator: installs `puppeteer-core` if missing, captures once at the larger of the two sizes, then runs two ffmpeg encodes (GIF uses two-stage `palettegen`/`paletteuse`; MP4 uses H.264 `yuv420p` + `+faststart`).

To tweak encoder settings, edit the ffmpeg invocations in `svg2video.sh`.
