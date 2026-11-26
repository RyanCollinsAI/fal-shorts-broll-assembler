from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ShotSpec:
    index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    caption: str
    visual_prompt: str


def plan_shots(whisper_json: Path) -> list[ShotSpec]:
    """Build beat-aligned ShotSpecs from Whisper segment/word timestamp JSON."""
    data = json.loads(whisper_json.read_text(encoding="utf-8"))
    segments = data.get("segments") or []
    if not segments:
        raise ValueError(f"No Whisper segments found in {whisper_json}")

    shots: list[ShotSpec] = []
    for idx, seg in enumerate(segments):
        start = float(seg["start"])
        end = float(seg["end"])
        dur = round(end - start, 3)
        if dur <= 0:
            raise ValueError(f"Segment {idx} has non-positive duration ({start} -> {end})")
        caption = str(seg.get("text") or "").strip()
        prompt = str(seg.get("visual_prompt") or f"Cinematic vertical B-roll illustrating: {caption}").strip()
        shots.append(
            ShotSpec(
                index=idx,
                start_sec=start,
                end_sec=end,
                duration_sec=dur,
                caption=caption,
                visual_prompt=prompt,
            )
        )
    return shots


def _srt_ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, rem = divmod(rem, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{rem:03d}"


def write_srt(shots: list[ShotSpec], out_path: Path) -> Path:
    blocks = []
    for i, shot in enumerate(shots, 1):
        blocks.append(f"{i}\n{_srt_ts(shot.start_sec)} --> {_srt_ts(shot.end_sec)}\n{shot.caption}\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(blocks), encoding="utf-8", newline="\n")
    return out_path
