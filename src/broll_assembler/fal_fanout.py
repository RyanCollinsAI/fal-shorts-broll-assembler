from __future__ import annotations

import asyncio
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

from .beat_planner import ShotSpec

DEFAULT_MODEL = "fal-ai/ltx-2/text-to-video"
COST_PER_SECOND_USD = 0.015  # configurable estimate for serverless video inference


@dataclass
class ShotTelemetry:
    index: int
    duration_sec: float
    queue_wait_ms: float
    inference_ms: float
    total_ms: float
    est_cost_usd: float
    clip_path: str
    mode: str


async def _default_fal_submit(model: str, arguments: dict[str, Any]) -> Any:
    import fal_client  # imported lazily so offline tests run without fal-client installed

    return await fal_client.submit_async(model, arguments=arguments)


def _render_synthetic_clip(out_path: Path, shot: ShotSpec) -> None:
    """Generate a real 1080x1920 30fps synthetic MP4 clip with FFmpeg for offline smoke tests."""
    palette = ["#111827", "#1e293b", "#0f172a", "#1f2937", "#18181b"]
    color = palette[shot.index % len(palette)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s=1080x1920:r=30:d={shot.duration_sec}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "ultrafast",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg synthetic clip failed: {proc.stderr[-300:]}")


async def generate_one_live(
    shot: ShotSpec,
    clips_dir: Path,
    *,
    model: str = DEFAULT_MODEL,
    submit_fn: Callable[[str, dict[str, Any]], Awaitable[Any]] = _default_fal_submit,
    download_fn: Callable[[str, Path], Awaitable[None]] | None = None,
) -> ShotTelemetry:
    t_submit = time.perf_counter()
    handle = await submit_fn(
        model,
        {
            "prompt": shot.visual_prompt,
            "aspect_ratio": "9:16",
            "duration": max(2, int(round(shot.duration_sec))),
        },
    )
    t_queued = time.perf_counter()
    result = await handle.get()
    t_done = time.perf_counter()

    video_url = (result.get("video") or {}).get("url") or result.get("url")
    if not video_url:
        raise RuntimeError(f"fal response for shot {shot.index} missing video url: {result}")

    clip_path = clips_dir / f"shot_{shot.index:02d}.mp4"
    clips_dir.mkdir(parents=True, exist_ok=True)
    if download_fn is not None:
        await download_fn(video_url, clip_path)
    else:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.get(video_url)
            resp.raise_for_status()
            clip_path.write_bytes(resp.content)

    queue_ms = round((t_queued - t_submit) * 1000.0, 1)
    infer_ms = round((t_done - t_queued) * 1000.0, 1)
    return ShotTelemetry(
        index=shot.index,
        duration_sec=shot.duration_sec,
        queue_wait_ms=queue_ms,
        inference_ms=infer_ms,
        total_ms=round((time.perf_counter() - t_submit) * 1000.0, 1),
        est_cost_usd=round(shot.duration_sec * COST_PER_SECOND_USD, 4),
        clip_path=str(clip_path),
        mode="fal_live",
    )


async def generate_one_offline(shot: ShotSpec, clips_dir: Path) -> ShotTelemetry:
    t0 = time.perf_counter()
    clip_path = clips_dir / f"shot_{shot.index:02d}.mp4"
    await asyncio.to_thread(_render_synthetic_clip, clip_path, shot)
    elapsed = round((time.perf_counter() - t0) * 1000.0, 1)
    return ShotTelemetry(
        index=shot.index,
        duration_sec=shot.duration_sec,
        queue_wait_ms=0.0,
        inference_ms=elapsed,
        total_ms=elapsed,
        est_cost_usd=0.0,
        clip_path=str(clip_path),
        mode="offline_ffmpeg",
    )


async def fanout_broll(
    shots: list[ShotSpec],
    clips_dir: Path,
    *,
    live: bool = False,
    model: str | None = None,
    submit_fn: Callable[[str, dict[str, Any]], Awaitable[Any]] = _default_fal_submit,
    download_fn: Callable[[str, Path], Awaitable[None]] | None = None,
) -> list[ShotTelemetry]:
    chosen_model = model or os.environ.get("FAL_VIDEO_MODEL") or DEFAULT_MODEL
    if live:
        coros = [
            generate_one_live(s, clips_dir, model=chosen_model, submit_fn=submit_fn, download_fn=download_fn)
            for s in shots
        ]
    else:
        coros = [generate_one_offline(s, clips_dir) for s in shots]
    results = await asyncio.gather(*coros)
    return sorted(results, key=lambda r: r.index)
