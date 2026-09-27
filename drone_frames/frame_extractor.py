import logging
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .video import VideoInfo

LOG = logging.getLogger(__name__)


def select_indices(timestamps: list[float], interval: float) -> list[int]:
    """First frame at/after each time-grid point, plus both timeline endpoints."""
    if interval <= 0:
        raise ValueError("Extraction interval must be positive")
    if not timestamps:
        return []
    # Never allocate a time grid larger than the video when interval < frame time.
    indices = [0]
    next_time = interval
    for i, timestamp in enumerate(timestamps[1:], 1):
        if timestamp + 1e-9 >= next_time:
            indices.append(i)
            next_time = (int((timestamp + 1e-9) / interval) + 1) * interval
    return sorted(set(indices + [len(timestamps) - 1]))


def extract_frames(video: VideoInfo, directory: Path, config: Config) -> list[dict]:
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError(f"Extraction directory must be empty: {directory}")
    selected = set(select_indices(video.timestamps, config.FRAME_EXTRACTION_INTERVAL))
    if len(selected) >= video.total_frame_count and video.total_frame_count > 2:
        raise ValueError("Interval would extract every frame; choose a larger interval")
    cap = cv2.VideoCapture(video.path)
    records = []
    try:
        if not cap.isOpened():
            raise ValueError("OpenCV could not open video")
        for index, timestamp in enumerate(video.timestamps):
            if not cap.grab():
                raise ValueError(f"Video decoding stopped at original frame {index}; output is incomplete")
            if index not in selected:
                continue
            ok, image = cap.retrieve()
            if not ok or image is None or not image.size:
                raise ValueError(f"Could not decode selected original frame {index}")
            name = f"frame_{len(records) + 1:06d}.jpg"
            if not cv2.imwrite(str(directory / name), image, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY]):
                raise OSError(f"Could not save {directory / name}")
            records.append({"frame": name, "timestamp_seconds": timestamp, "original_frame_number": index})
            if len(records) % 50 == 0:
                LOG.info("Extracted %d frames (%.1f / %.1f seconds)", len(records), timestamp, video.duration_seconds)
        if cap.grab():
            raise ValueError("OpenCV decoded more frames than ffprobe; timestamp mapping is unreliable")
    finally:
        cap.release()
    return records


def contact_sheet(items: list[tuple[Path, str]], destination: Path, columns: int = 4):
    tiles = []
    for path, label in items:
        im = cv2.imread(str(path))
        tile = np.zeros((204, 320, 3), np.uint8)
        if im is not None:
            scale = min(320 / im.shape[1], 180 / im.shape[0])
            resized = cv2.resize(im, (round(im.shape[1] * scale), round(im.shape[0] * scale)))
            tile[:resized.shape[0], :resized.shape[1]] = resized
        cv2.putText(tile, label, (5, 197), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1)
        tiles.append(tile)
    if not tiles:
        return
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)])
    if not cv2.imwrite(str(destination), sheet):
        raise OSError(f"Could not save {destination}")
