#!/usr/bin/env bash
# Download a YouTube video's audio as mp3.
#
# Usage: yt_audio.sh [--mono] [--bitrate 128k|64k|...] [--out DIR] URL
#
# Bitrate rule (when --bitrate is not given):
#   128 kbps by default; if a 128 kbps file would exceed 200 MB, use 64 kbps.
# Stereo by default; --mono downmixes to 1 channel.
set -euo pipefail

YTDLP="$HOME/tech/yt-dlp/.venv/bin/yt-dlp"
OUT_DIR="$HOME/tech/yt-dlp"
LIMIT_BYTES=$((200 * 1000 * 1000))
MONO=0
BITRATE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mono) MONO=1; shift ;;
    --bitrate) BITRATE="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) URL="$1"; shift ;;
  esac
done
[[ -n "${URL:-}" ]] || { echo "usage: $0 [--mono] [--bitrate RATE] [--out DIR] URL" >&2; exit 2; }
command -v ffmpeg >/dev/null || { echo "ffmpeg not found (brew install ffmpeg)" >&2; exit 1; }

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# Plain requests get "Sign in to confirm you're not a bot", so use Chrome cookies.
# Download the original audio stream (no transcoding) so the mp3 is encoded only once.
SRC=$("$YTDLP" --cookies-from-browser chrome --no-playlist -f bestaudio \
  --no-progress --print after_move:filepath --no-simulate \
  -o "$WORK/%(title)s.%(ext)s" "$URL" | tail -n 1)
[[ -f "$SRC" ]] || { echo "download failed" >&2; exit 1; }

TITLE=$(basename "${SRC%.*}")
DURATION=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$SRC")

if [[ -z "$BITRATE" ]]; then
  EST_128=$(awk -v d="$DURATION" 'BEGIN { printf "%d", d * 128000 / 8 }')
  if (( EST_128 > LIMIT_BYTES )); then BITRATE=64k; else BITRATE=128k; fi
fi

CHANNEL_ARGS=()
SUFFIX=""
if (( MONO )); then CHANNEL_ARGS=(-ac 1); SUFFIX=" (mono)"; fi

mkdir -p "$OUT_DIR"
DEST="$OUT_DIR/$TITLE$SUFFIX.mp3"
[[ ! -e "$DEST" ]] || { echo "already exists: $DEST" >&2; exit 1; }

ffmpeg -nostdin -v error -i "$SRC" -map 0:a -metadata title="$TITLE" \
  -c:a libmp3lame -b:a "$BITRATE" ${CHANNEL_ARGS[@]+"${CHANNEL_ARGS[@]}"} "$DEST"

SIZE=$(stat -f %z "$DEST")
echo "file:     $DEST"
echo "duration: $(awk -v d="$DURATION" 'BEGIN { printf "%dh %02dm %02ds", d/3600, (d%3600)/60, d%60 }')"
echo "bitrate:  $BITRATE $( (( MONO )) && echo mono || echo stereo)"
echo "size:     $(awk -v s="$SIZE" 'BEGIN { printf "%.0f MB", s / 1000000 }')"
if (( SIZE > LIMIT_BYTES )); then
  echo "warning: still over 200 MB at $BITRATE; consider --bitrate 48k or 32k" >&2
fi
