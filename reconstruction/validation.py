import numpy as np
from plyfile import PlyData, PlyElement


def validate_sparse(stats, config):
    failures = []
    for key, limit, comparison, text in [
        ("registration_rate", config.MIN_REGISTRATION_RATE, lambda x, y: x < y, "registration rate below minimum"),
        ("sparse_points", config.MIN_SPARSE_POINTS, lambda x, y: x < y, "too few triangulated points"),
        ("mean_observations_per_point", config.MIN_MEAN_TRACK_LENGTH, lambda x, y: x < y, "feature tracks too short"),
        ("mean_reprojection_error_px", config.MAX_REPROJECTION_ERROR, lambda x, y: x > y, "reprojection error too high"),
    ]:
        if not np.isfinite(stats[key]) or comparison(stats[key], limit):
            failures.append(f"{text}: {stats[key]} (threshold {limit})")
    return failures


def validate_ply(path, require_faces=False):
    data = PlyData.read(path, mmap="r")
    vertices = data["vertex"].data
    if not len(vertices):
        raise ValueError(f"Empty geometry: {path}")
    for key in ("x", "y", "z"):
        if not np.isfinite(vertices[key]).all():
            raise ValueError(f"Non-finite {key} in {path}")
    extent = [float(np.ptp(vertices[key])) for key in ("x", "y", "z")]
    if max(extent) <= 0:
        raise ValueError(f"Degenerate geometry: {path}")
    faces = len(data["face"].data) if "face" in data else 0
    if require_faces and faces == 0:
        raise ValueError(f"Mesh has no faces: {path}")
    if faces:
        for face in data["face"].data["vertex_indices"]:
            if len(face) < 3 or np.any(face < 0) or np.any(face >= len(vertices)):
                raise ValueError(f"Invalid face indices in {path}")
    return {"vertices": len(vertices), "faces": faces, "size_bytes": path.stat().st_size,
            "extent_arbitrary_units": extent, "finite_coordinates": True}


def write_camera_ply(path, positions):
    vertices = np.empty(len(positions), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                               ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for i, point in enumerate(positions):
        vertices[i] = (*point, 255, 40, 40)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)
