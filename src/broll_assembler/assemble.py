from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .beat_planner import ShotSpec
from .fal_fanout import ShotTelemetry


def synthesize_voiceover_wav(duration_sec: float, out_wav: Path) -> Path:
    """Create a clean silent/soft-tone synthetic voiceover bed when no real VO track is passed."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=220:sample_rate=44100:duration={duration_sec}",
        "-af",
        "volume=0.04",
        str(out_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg synthetic audio failed: {proc.stderr[-300:]}")
    return out_wav


def assemble_short(
    shots: list[ShotSpec],
    telemetry: list[ShotTelemetry],
    out_mp4: Path,
    *,
    voiceover_path: Path | None = None,
) -> Path:
    """Normalize every B-roll shot to 1080x1920 30fps, concat cleanly, and mux audio into a 9:16 MP4."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    work = out_mp4.parent / "_norm"
    work.mkdir(parents=True, exist_ok=True)

    norm_files: list[Path] = []
    for shot, tel in zip(shots, telemetry):
        norm = work / f"norm_{shot.index:02d}.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            tel.clip_path,
            "-t",
            str(shot.duration_sec),
            "-vf",
            "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=30",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "ultrafast",
            "-an",
            str(norm),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg normalize shot {shot.index} failed: {proc.stderr[-300:]}")
        norm_files.append(norm)

    concat_list = work / "concat.txt"
    concat_list.write_text(
        "\n".join(f"file '{p.resolve().as_posix()}'" for p in norm_files) + "\n",
        encoding="utf-8",
    )

    total_dur = round(sum(s.duration_sec for s in shots), 3)
    audio = voiceover_path or synthesize_voiceover_wav(total_dur, work / "voiceover.wav")

    mux_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-i",
        str(audio),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(out_mp4),
    ]
    proc = subprocess.run(mux_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed: {proc.stderr[-300:]}")
    return out_mp4


def probe_video(mp4_path: Path) -> dict:
    """Return width, height, duration_sec, and stream codecs via ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,codec_name,width,height:format=duration",
        "-of",
        "json",
        str(mp4_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr[-300:]}")
    info = json.loads(proc.stdout)
    vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
    astream = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    return {
        "width": int(vstream["width"]),
        "height": int(vstream["height"]),
        "video_codec": vstream["codec_name"],
        "audio_codec": astream["codec_name"] if astream else None,
        "duration_sec": round(float(info["format"]["duration"]), 2),
    }
