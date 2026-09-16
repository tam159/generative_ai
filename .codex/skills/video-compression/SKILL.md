---
name: video-compression
description: Compress a local video with ffmpeg while preserving visual quality, resolution, audio, and compatibility. Use whenever the user asks to compress, shrink, reduce the size of, or optimize a video file.
---

# Video compression

Create a smaller, verified copy of the requested video. The source file must never be modified or deleted.

## Workflow

1. Resolve the exact input path and confirm it exists. Use `ffprobe` to inspect duration, dimensions, codecs, frame rate, and source size before encoding.
2. Write the result beside the source as `<stem>-compressed.mp4`. Do not overwrite an existing output: choose a distinct suffix such as `-compressed-2.mp4` instead.
3. Preserve the original resolution and frame timing unless the user explicitly requests resizing or a smaller frame rate.
4. On macOS, prefer Apple hardware HEVC when `hevc_videotoolbox` is available. It provides a strong quality/speed tradeoff. Use `-tag:v hvc1` so the resulting MP4 is recognized by Apple players.
5. Preserve audio with `-c:a copy` when it is MP4-compatible. Otherwise encode AAC at 160k or the source bitrate, whichever is higher. Include optional audio with `-map '0:a?'`.
6. Use `-movflags +faststart`, so the MP4 can begin playback before it is fully downloaded.
7. Wait for the encode to finish; do not report a partial file as complete. Then use `ffprobe` and `ls -lh` to verify its codecs, dimensions, duration, and final size.

## Default high-quality presets

Choose the hardware bitrate from the source height. This intentionally keeps screen recordings and text legible while materially reducing size.

| Source height | Video bitrate | Max rate | Buffer |
| --- | ---: | ---: | ---: |
| 2160p or higher | 4M | 6M | 12M |
| 1080p–1440p | 2M | 3M | 6M |
| 720p | 1M | 1.5M | 3M |
| Below 720p | 700k | 1M | 2M |

Use this template, substituting the chosen rates and the resolved paths:

```sh
ffmpeg -hide_banner -n -i INPUT \
  -map 0:v:0 -map '0:a?' -map_metadata 0 \
  -c:v hevc_videotoolbox -b:v VIDEO_RATE -maxrate MAX_RATE -bufsize BUFFER \
  -tag:v hvc1 -c:a copy -movflags +faststart OUTPUT
```

If copying audio fails because it is not compatible with MP4, rerun with `-c:a aac -b:a 160k` (or a higher source-equivalent rate).

## Smaller-file option

When the user explicitly prioritizes the smallest practical file and accepts a potentially long encode, use software HEVC instead:

```sh
ffmpeg -hide_banner -n -i INPUT \
  -map 0:v:0 -map '0:a?' -map_metadata 0 \
  -c:v libx265 -preset medium -crf 22 -tag:v hvc1 \
  -c:a copy -movflags +faststart OUTPUT
```

Use CRF 20 for near-transparent quality or CRF 24 for a smaller file. Explain that lower CRF means higher quality and larger output. For a faster software run, use `-preset fast`; it generally yields a somewhat larger file at the same CRF.

## Compatibility fallback

If the user requires broad device compatibility or HEVC is unavailable, encode H.264 instead:

```sh
ffmpeg -hide_banner -n -i INPUT \
  -map 0:v:0 -map '0:a?' -map_metadata 0 \
  -c:v libx264 -preset medium -crf 20 -c:a copy \
  -movflags +faststart OUTPUT
```

## Verification and handoff

After a successful encode, run:

```sh
ls -lh INPUT OUTPUT
ffprobe -v error \
  -show_entries format=duration,size:stream=codec_name,codec_type,width,height \
  -of default=noprint_wrappers=1 OUTPUT
```

Report the output path, original and compressed sizes, percentage reduction, codec, and whether dimensions/duration were preserved. Mention the original was left intact. If the video is lengthy, give concise progress updates at least once a minute while encoding.
