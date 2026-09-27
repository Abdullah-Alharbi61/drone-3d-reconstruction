"""Probe presentation timestamps so variable-rate footage also uses real time."""
from dataclasses import asdict, dataclass
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np


@dataclass
class VideoInfo:
    path: str
    duration_seconds: float
    fps: float
    width: int
    height: int
    total_frame_count: int
    timestamps: list[float]
    timestamp_origin_seconds: float
    variable_frame_rate: bool

    def metadata(self):
        return {k: v for k, v in asdict(self).items() if k != "timestamps"}


def probe_video(path: Path) -> VideoInfo:
    if not path.is_file():
        raise ValueError(f"Input video does not exist: {path}")
    if not shutil.which("ffprobe"):
        raise ValueError("ffprobe is required (install FFmpeg) for reliable frame timestamps")
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "stream=width,height,avg_frame_rate,duration,nb_frames:frame=best_effort_timestamp_time",
        "-of", "json", str(path.resolve()),
    ], capture_output=True, text=True, check=True)
    if result.stderr.strip():
        raise ValueError(f"Video decode errors during inspection: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    if not data.get("streams") or not data.get("frames"):
        raise ValueError("No readable video frames")
    stream = data["streams"][0]
    raw = [float(f["best_effort_timestamp_time"]) for f in data["frames"]]
    if not np.isfinite(raw).all() or np.any(np.diff(raw) <= 0):
        raise ValueError("Missing or non-increasing presentation timestamps")
    fps = float(Fraction(stream["avg_frame_rate"]))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Video has invalid FPS")
    if stream.get("nb_frames", "N/A") != "N/A" and int(stream["nb_frames"]) != len(raw):
        raise ValueError("Decoded frame count differs from declared count; input may be truncated")
    times = [t - raw[0] for t in raw]
    duration = float(stream.get("duration", times[-1] + 1 / fps))
    return VideoInfo(str(path.resolve()), duration, fps, int(stream["width"]),
                     int(stream["height"]), len(times), times, raw[0],
                     bool(len(times) > 2 and np.ptp(np.diff(times)) > 0.001))
