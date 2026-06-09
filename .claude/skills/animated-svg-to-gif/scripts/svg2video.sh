#!/usr/bin/env bash
# Convert an animated SVG (CSS @keyframes / SMIL) into a GIF and/or MP4.
#
# Pipeline: headless Chrome steps the animation timeline frame-by-frame
# (svg2anim.mjs) -> ffmpeg encodes the frames. ffmpeg alone can't do this:
# it has no SVG-animation timeline, only a single static rasterization.
#
# Usage:
#   svg2video.sh <input.svg> [options]
#
# Options:
#   --duration MS    animation loop length in ms          (default 9000)
#   --fps N          frames per second                    (default 20)
#   --out BASE       output path base (no extension)      (default: input stem)
#   --gif-size PX    GIF square size; 0 disables the GIF  (default 720)
#   --mp4-size PX    MP4 square size; 0 disables the MP4   (default 1080)
#   --keep-frames    keep the intermediate PNG frames
#   --chrome PATH    Chrome binary (or set CHROME_PATH env)
#
# Defaults produce both <stem>.gif and <stem>.mp4 next to the input.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SVG="" DURATION=9000 FPS=20 OUT="" GIF_SIZE=720 MP4_SIZE=1080 KEEP=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration) DURATION="$2"; shift 2;;
    --fps) FPS="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --gif-size) GIF_SIZE="$2"; shift 2;;
    --mp4-size) MP4_SIZE="$2"; shift 2;;
    --keep-frames) KEEP=1; shift;;
    --chrome) export CHROME_PATH="$2"; shift 2;;
    -*) echo "unknown option: $1" >&2; exit 1;;
    *) SVG="$1"; shift;;
  esac
done

[[ -n "$SVG" && -f "$SVG" ]] || { echo "usage: svg2video.sh <input.svg> [options]" >&2; exit 1; }
[[ "$GIF_SIZE" -gt 0 || "$MP4_SIZE" -gt 0 ]] || { echo "nothing to do: both sizes are 0" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg not found (brew install ffmpeg)" >&2; exit 1; }

[[ -z "$OUT" ]] && OUT="${SVG%.*}"

# First run installs puppeteer-core (pure JS, no bundled browser download).
if [[ ! -d "$DIR/node_modules/puppeteer-core" ]]; then
  echo "installing puppeteer-core (one time)..."
  ( cd "$DIR" && npm i puppeteer-core >/dev/null 2>&1 )
fi

# Capture once at the larger requested size, then let ffmpeg downscale per output.
MASTER=$(( GIF_SIZE > MP4_SIZE ? GIF_SIZE : MP4_SIZE ))
FRAMES="$(mktemp -d -t svg2video)"
trap '[[ $KEEP -eq 0 ]] && rm -rf "$FRAMES"' EXIT

node "$DIR/svg2anim.mjs" "$SVG" "$FRAMES" "$DURATION" "$FPS" "$MASTER"

if [[ "$GIF_SIZE" -gt 0 ]]; then
  # palettegen/paletteuse keeps dark gradients from banding in GIF's 256 colors.
  ffmpeg -y -framerate "$FPS" -i "$FRAMES/f_%04d.png" \
    -vf "scale=${GIF_SIZE}:${GIF_SIZE}:flags=lanczos,split[s0][s1];[s0]palettegen=stats_mode=full[p];[s1][p]paletteuse=dither=floyd_steinberg" \
    -loop 0 "${OUT}.gif" 2>/dev/null
  echo "GIF -> ${OUT}.gif"
fi

if [[ "$MP4_SIZE" -gt 0 ]]; then
  ffmpeg -y -framerate "$FPS" -i "$FRAMES/f_%04d.png" \
    -vf "scale=${MP4_SIZE}:${MP4_SIZE}:flags=lanczos,format=yuv420p" \
    -c:v libx264 -crf 18 -preset slow -movflags +faststart "${OUT}.mp4" 2>/dev/null
  echo "MP4 -> ${OUT}.mp4"
fi

[[ $KEEP -eq 1 ]] && echo "frames kept in $FRAMES"
exit 0
