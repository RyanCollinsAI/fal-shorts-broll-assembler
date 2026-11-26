from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from broll_assembler.assemble import assemble_short, probe_video
from broll_assembler.beat_planner import plan_shots, write_srt
from broll_assembler.fal_fanout import fanout_broll


SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample_whisper_beats.json"


class FakeFalHandle:
    def __init__(self, idx: int, delay: float = 0.02) -> None:
        self.idx = idx
        self.delay = delay

    async def get(self) -> dict:
        await asyncio.sleep(self.delay)
        return {"video": {"url": f"https://cdn.fal.example/shot_{self.idx}.mp4"}}


def test_plan_shots_and_srt_export(tmp_path: Path) -> None:
    shots = plan_shots(SAMPLE)
    assert len(shots) == 5
    assert round(sum(s.duration_sec for s in shots), 2) == 15.0
    srt = write_srt(shots, tmp_path / "captions.srt")
    text = srt.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:03,000" in text
    assert "00:00:12,000 --> 00:00:15,000" in text


def test_mocked_fal_async_fanout_parallel_telemetry(tmp_path: Path) -> None:
    shots = plan_shots(SAMPLE)
    submitted: list[tuple[str, dict]] = []

    async def fake_submit(model: str, arguments: dict) -> FakeFalHandle:
        idx = len(submitted)
        submitted.append((model, arguments))
        return FakeFalHandle(idx=idx, delay=0.02)

    async def fake_download(url: str, dest: Path) -> None:
        dest.write_bytes(b"fake-mp4-bytes:" + url.encode("utf-8"))

    telemetry = asyncio.run(
        fanout_broll(
            shots,
            tmp_path / "clips",
            live=True,
            model="fal-ai/ltx-2/text-to-video",
            submit_fn=fake_submit,
            download_fn=fake_download,
        )
    )

    assert len(submitted) == 5
    assert len(telemetry) == 5
    assert all(t.mode == "fal_live" for t in telemetry)
    assert all(Path(t.clip_path).exists() for t in telemetry)
    assert round(sum(t.est_cost_usd for t in telemetry), 4) == 0.225


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg not on PATH")
def test_real_ffmpeg_smoke_assembly_9x16(tmp_path: Path) -> None:
    shots = plan_shots(SAMPLE)[:2]  # fast 6-second 9:16 smoke render
    telemetry = asyncio.run(fanout_broll(shots, tmp_path / "clips", live=False))
    out_mp4 = assemble_short(shots, telemetry, tmp_path / "short_9x16.mp4")
    assert out_mp4.exists() and out_mp4.stat().st_size > 1000
    meta = probe_video(out_mp4)
    assert meta["width"] == 1080
    assert meta["height"] == 1920
    assert meta["video_codec"] == "h264"
    assert meta["audio_codec"] == "aac"
    assert 5.5 <= meta["duration_sec"] <= 6.5
