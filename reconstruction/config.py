from dataclasses import dataclass, asdict
import json
import math


@dataclass
class Config:
    RECONSTRUCTION_ENGINE: str = "colmap"
    CAMERA_MODEL: str = "AUTO"
    CAMERA_PARAMS: str = ""
    REFINE_INTRINSICS: bool = True
    SINGLE_CAMERA: bool = True
    MATCHING_STRATEGY: str = "sequential"
    MATCH_OVERLAP: int = 10
    LOOP_DETECTION: bool = True
    LOOP_KEYFRAME_STRIDE: int = 15
    MAX_IMAGE_SIZE: int = 1600
    MAX_FEATURES: int = 4096
    USE_GPU: bool = True
    NUM_THREADS: int = 2
    RUN_DENSE_RECONSTRUCTION: bool = True
    DENSE_MAX_IMAGE_SIZE: int = 1280
    DENSE_NUM_SOURCE_IMAGES: int = 10
    DENSE_ITERATIONS: int = 3
    DENSE_CACHE_GB: float = 0.25
    FUSION_CACHE_GB: float = 2.0
    FUSION_MIN_PIXELS: int = 3
    GENERATE_MESH: bool = True
    MESH_DEPTH: int = 9
    MESH_MAX_DISTANCE_CELLS: float = 2.0
    MIN_MESH_POINTS: int = 10000
    MIN_REGISTRATION_RATE: float = 0.8
    MIN_SPARSE_POINTS: int = 1000
    MIN_MEAN_TRACK_LENGTH: float = 3.0
    MAX_REPROJECTION_ERROR: float = 2.0
    MIN_DENSE_POINTS: int = 10000
    MIN_DENSE_SPARSE_RATIO: float = 1.5
    INIT_MIN_TRI_ANGLE: float = 8.0
    SOURCE_VIDEO: str = "Drone Cam 1.mp4"

    def validate(self):
        if self.RECONSTRUCTION_ENGINE != "colmap":
            raise ValueError("Only COLMAP is implemented")
        if self.MATCHING_STRATEGY not in ("sequential", "exhaustive"):
            raise ValueError("MATCHING_STRATEGY must be sequential or exhaustive")
        for name, value in asdict(self).items():
            default = getattr(Config(), name)
            if isinstance(default, bool):
                if type(value) is not bool:
                    raise ValueError(f"{name} must be boolean")
            elif isinstance(default, (float, int)):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{name} must be finite and positive")
                if isinstance(default, int) and not isinstance(value, int):
                    raise ValueError(f"{name} must be an integer")
            elif not isinstance(value, str):
                raise ValueError(f"{name} must be a string")
        if self.MIN_REGISTRATION_RATE > 1:
            raise ValueError("MIN_REGISTRATION_RATE must be <= 1")
        if self.CAMERA_MODEL == "AUTO" and self.CAMERA_PARAMS:
            raise ValueError("Specify CAMERA_MODEL explicitly with calibrated CAMERA_PARAMS")
        if not self.RUN_DENSE_RECONSTRUCTION and self.GENERATE_MESH:
            raise ValueError("Mesh generation requires dense reconstruction")
        return self

    @classmethod
    def load(cls, path):
        return cls(**(json.loads(path.read_text()) if path else {})).validate()
