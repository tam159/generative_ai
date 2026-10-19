---
name: youtube-audio
description: Download the audio of a YouTube video as an mp3 using the local yt-dlp venv. Use whenever the user gives a YouTube link (youtube.com/watch, youtu.be, music.youtube.com) and wants the audio, an mp3, the sound, or "just the music/chant/podcast" from it. Picks the bitrate automatically to keep files under 200 MB, stereo by default.
---

Download a YouTube video's audio as an mp3 with `scripts/yt_audio.sh`.

## Defaults (user preferences)

- **Output folder:** `~/tech/yt-dlp/`, file named after the video title.
- **Bitrate:** 128 kbps. If a 128 kbps file would be **over 200 MB** (videos longer than about 3.5 hours), use **64 kbps** instead. The user finds little audible difference between the two.
- **Channels:** stereo. Only use mono (`--mono`) when the user asks for it; mono files get a ` (mono)` suffix.

## Run

```bash
~/tech/generative_ai/.claude/skills/youtube-audio/scripts/yt_audio.sh "<URL>"
```

Options: `--mono`, `--bitrate 96k` (overrides the automatic choice), `--out DIR`. Quote the URL, because `&` in links like `...?v=ID&t=9s` would otherwise break the shell command. Use a 600000 ms timeout, since long videos take a minute or two to encode.

The script prints the file path, duration, bitrate and size; report those to the user. It refuses to overwrite an existing file. If that happens, ask the user before deleting the old one.

## How it works and why

- **Chrome cookies are always passed** (`--cookies-from-browser chrome`). Without them, YouTube answers with HTTP 429 and "Sign in to confirm you're not a bot", and switching player clients only gets a 403. The user approved using their Chrome session for this.
- **It downloads the original audio stream** (`-f bestaudio`, usually Opus in webm) and encodes the mp3 from it in a single pass. Never re-encode from an existing mp3: a second lossy pass makes it sound worse. To make a different version (mono, another bitrate), run the script again on the URL. The download is fast.
- `--no-playlist` is set, so a link containing `&list=` downloads only that one video.

## Troubleshooting

- **Still says "not a bot" / 403 with cookies:** the user might not be signed in to YouTube in Chrome, or yt-dlp is out of date. Update it with `~/tech/yt-dlp/.venv/bin/python -m pip install -U yt-dlp` (or `uv pip install -U yt-dlp --python ~/tech/yt-dlp/.venv/bin/python` if the venv has no pip).
- **Still over 200 MB at 64 kbps** (videos longer than about 7 hours): the script prints a warning. Offer `--bitrate 48k` or `32k`, which still work for speech or chanting.
