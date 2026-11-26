# fal-shorts-broll-assembler

A lightweight async Python CLI that reads a Whisper word-timestamp JSON beat map, fans out parallel generative B-roll jobs on `fal.ai` (`fal_client.submit_async` + `asyncio.gather`) with per-shot latency and cost telemetry, and stitches the returned clips plus voiceover into a 9:16 (`1080x1920`) H.264/AAC Short in one FFmpeg pass.

## Why build this

In an automated short-form pipeline, generating B-roll one clip at a time turns a 5-shot, 15-second Short into a slow serial bottleneck.
Parsing Whisper timestamps into a deterministic beat manifest first lets all 5 shots dispatch concurrently over async serverless endpoints while tracking queue wait, inference time, and unit economics per finished video.

## Verified Offline Smoke Run (Real FFmpeg 9:16 Assembly)

Running `python -m broll_assembler.cli` against `data/sample_whisper_beats.json` fans out all 5 shots concurrently and renders a 15-second `1080x1920` H.264/AAC MP4 plus beat-synced `captions.srt`:

| Shot | Dur (s) | Queue Wait (ms) | Inference (ms) | Total (ms) | Est. Cost (USD) | Mode |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 3.0 | 0.0 | 719.4 | 719.4 | $0.0000 | `offline_ffmpeg` |
| 1 | 3.0 | 0.0 | 720.1 | 720.1 | $0.0000 | `offline_ffmpeg` |
| 2 | 3.0 | 0.0 | 719.2 | 719.2 | $0.0000 | `offline_ffmpeg` |
| 3 | 3.0 | 0.0 | 663.2 | 663.2 | $0.0000 | `offline_ffmpeg` |
| 4 | 3.0 | 0.0 | 716.6 | 716.6 | $0.0000 | `offline_ffmpeg` |

Fan-out wall clock: **722.8 ms** across 5 concurrent 3-second shots (vs ~3.5 s serial).
Verified via `ffprobe`: `artifacts/short_9x16.mp4` (`1080x1920`, `15.0s`, `h264/aac`).

## Quickstart

```bash
pip install -e ".[dev]"

# 1. Offline end-to-end run (real FFmpeg 9:16 render + captions.srt + telemetry JSON)
python -m broll_assembler.cli --whisper data/sample_whisper_beats.json

# 2. Live async fan-out on fal.ai
pip install -e ".[live]"
export FAL_KEY="your-fal-key"
export FAL_VIDEO_MODEL="fal-ai/ltx-2/text-to-video"
python -m broll_assembler.cli --whisper data/sample_whisper_beats.json --live

# 3. Run tests (3/3 passing: beat planner, mocked fal_client fan-out, real FFmpeg 9:16 smoke render)
pytest -q
```
