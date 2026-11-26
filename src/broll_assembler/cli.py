from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path

from .assemble import assemble_short, probe_video
from .beat_planner import plan_shots, write_srt
from .fal_fanout import ShotTelemetry, fanout_broll


def format_markdown(rows: list[ShotTelemetry], wall_ms: float) -> str:
    lines = [
        "| Shot | Dur (s) | Queue Wait (ms) | Inference (ms) | Total (ms) | Est. Cost (USD) | Mode |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.index} | {r.duration_sec:.1f} | {r.queue_wait_ms:.1f} | {r.inference_ms:.1f} | "
            f"{r.total_ms:.1f} | ${r.est_cost_usd:.4f} | `{r.mode}` |"
        )
    total_cost = sum(r.est_cost_usd for r in rows)
    lines.append(
        f"\nFan-out wall clock: **{wall_ms:.1f} ms** | Estimated unit cost per Short: **${total_cost:.4f}**"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Whisper beat map -> async fal.ai B-roll -> 9:16 FFmpeg Short")
    ap.add_argument("--whisper", type=Path, default=Path("data/sample_whisper_beats.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    ap.add_argument("--live", action="store_true", help="Dispatch live async jobs to fal.ai (requires FAL_KEY)")
    ap.add_argument("--voiceover", type=Path, default=None)
    args = ap.parse_args(argv)

    shots = plan_shots(args.whisper)
    srt_path = write_srt(shots, args.out_dir / "captions.srt")

    t0 = time.perf_counter()
    telemetry = asyncio.run(fanout_broll(shots, args.out_dir / "clips", live=args.live))
    wall_ms = (time.perf_counter() - t0) * 1000.0

    mp4_path = assemble_short(shots, telemetry, args.out_dir / "short_9x16.mp4", voiceover_path=args.voiceover)
    meta = probe_video(mp4_path)

    report = {
        "wall_ms": round(wall_ms, 1),
        "total_est_cost_usd": round(sum(t.est_cost_usd for t in telemetry), 4),
        "output_mp4": str(mp4_path),
        "captions_srt": str(srt_path),
        "video_meta": meta,
        "shots": [asdict(t) for t in telemetry],
    }
    (args.out_dir / "run_telemetry.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(format_markdown(telemetry, wall_ms))
    print(f"Rendered: {mp4_path} ({meta['width']}x{meta['height']}, {meta['duration_sec']}s, {meta['video_codec']}/{meta['audio_codec']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
