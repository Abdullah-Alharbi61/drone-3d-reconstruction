from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path


@dataclass
class Config:
    FRAME_EXTRACTION_INTERVAL: float = 0.4
    ANALYSIS_WIDTH: int = 640
    JPEG_QUALITY: int = 98
    BLUR_THRESHOLD: float = 100.0
    DARKNESS_THRESHOLD: float = 15.0
    OVEREXPOSURE_THRESHOLD: float = 245.0
    DARK_PIXEL_THRESHOLD: int = 10
    BRIGHT_PIXEL_THRESHOLD: int = 245
    CLIPPED_FRACTION_THRESHOLD: float = 0.85
    EXPOSURE_MAX_EDGE_FRACTION: float = 0.01
    DUPLICATE_THRESHOLD: float = 0.995
    DUPLICATE_MAX_MEAN_DIFFERENCE: float = 1.5
    DUPLICATE_PIXEL_DIFFERENCE: float = 10.0
    DUPLICATE_MAX_CHANGED_FRACTION: float = 0.02
    DUPLICATE_MAX_MOTION_PIXELS: float = 0.25
    DUPLICATE_MIN_TRACK_FRACTION: float = 0.8
    DUPLICATE_WINDOW_SECONDS: float = 1.2
    MAX_DUPLICATE_GAP_SECONDS: float = 1.2
    MIN_EDGE_FRACTION: float = 0.001
    MIN_CORNERS: int = 8
    MIN_ENTROPY: float = 3.0
    CANNY_LOW: float = 50.0
    CANNY_HIGH: float = 150.0
    FEATURE_QUALITY: float = 0.01
    FEATURE_MIN_DISTANCE: float = 8.0
    FEATURE_MAX_CORNERS: int = 500
    FLOW_MIN_TRACKS: int = 20
    FLOW_MAX_ERROR: float = 20.0
    FLOW_MAX_FORWARD_BACKWARD_ERROR: float = 1.0
    EXTREME_MOTION_FRACTION: float = 0.2
    MOTION_BLUR_THRESHOLD: float = 250.0
    OVERLAP_WARNING_TRACK_FRACTION: float = 0.5
    COVERAGE_WARNING_GAP_SECONDS: float = 1.6
    INSPECTION_SAMPLE_COUNT: int = 24
    # Optional human-reviewed intervals [start_seconds, end_seconds], inclusive.
    EXCLUDE_TIME_RANGES: list = field(default_factory=list)

    def validate(self):
        for key, value in asdict(self).items():
            if key == "EXCLUDE_TIME_RANGES":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be a finite nonnegative number")
        for key in ("ANALYSIS_WIDTH", "JPEG_QUALITY", "DARK_PIXEL_THRESHOLD", "BRIGHT_PIXEL_THRESHOLD",
                    "MIN_CORNERS", "FEATURE_MAX_CORNERS", "FLOW_MIN_TRACKS", "INSPECTION_SAMPLE_COUNT"):
            if not isinstance(getattr(self, key), int):
                raise ValueError(f"{key} must be an integer")
        if self.FRAME_EXTRACTION_INTERVAL <= 0 or self.ANALYSIS_WIDTH < 32 or self.INSPECTION_SAMPLE_COUNT < 2:
            raise ValueError("Interval must be positive, analysis width >= 32, sample count >= 2")
        if not 1 <= self.JPEG_QUALITY <= 100:
            raise ValueError("JPEG_QUALITY must be in [1, 100]")
        for key in ("DUPLICATE_THRESHOLD", "CLIPPED_FRACTION_THRESHOLD", "EXPOSURE_MAX_EDGE_FRACTION",
                    "DUPLICATE_MAX_CHANGED_FRACTION", "DUPLICATE_MIN_TRACK_FRACTION", "MIN_EDGE_FRACTION",
                    "FEATURE_QUALITY", "EXTREME_MOTION_FRACTION", "OVERLAP_WARNING_TRACK_FRACTION"):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key} must be in [0, 1]")
        if not 0 < self.FEATURE_QUALITY <= 1 or self.FEATURE_MAX_CORNERS < 1 or self.FLOW_MIN_TRACKS < 1:
            raise ValueError("Feature quality and feature/track counts must be positive")
        if not 0 <= self.DARKNESS_THRESHOLD < self.OVEREXPOSURE_THRESHOLD <= 255:
            raise ValueError("Invalid brightness thresholds")
        if not 0 <= self.DARK_PIXEL_THRESHOLD < self.BRIGHT_PIXEL_THRESHOLD <= 255:
            raise ValueError("Invalid clipping thresholds")
        if self.CANNY_LOW >= self.CANNY_HIGH:
            raise ValueError("CANNY_LOW must be below CANNY_HIGH")
        for pair in self.EXCLUDE_TIME_RANGES:
            if (not isinstance(pair, list) or len(pair) != 2 or
                any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in pair) or
                not 0 <= pair[0] <= pair[1]):
                raise ValueError("EXCLUDE_TIME_RANGES must contain [start, end] seconds")
        return self

    @classmethod
    def load(cls, path: Path | None):
        return cls(**(json.loads(path.read_text()) if path else {})).validate()
