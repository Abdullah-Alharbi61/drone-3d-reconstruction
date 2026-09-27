"""COLMAP Poisson meshing with a validated fallback for broken native trimming."""
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

from .validation import validate_ply


def trim_to_observations(raw_mesh, dense_cloud, output, depth, distance_cells, threads):
    """Remove unsupported faces, without changing coordinates or adding geometry.

    The distance bound is relative to the meshing grid, never a metric distance.
    Test vertices AND face centroids so faces cannot simply bridge an empty region.
    """
    validate_ply(raw_mesh, require_faces=True)
    source = PlyData.read(dense_cloud, mmap="r")["vertex"].data
    points = np.column_stack([source[key] for key in ("x", "y", "z")]).astype(np.float64)
    tree = cKDTree(points)
    radius = float(np.ptp(points, axis=0).max() / 2 ** depth * distance_cells)
    mesh = PlyData.read(raw_mesh)
    vertices = mesh["vertex"].data
    positions = np.column_stack([vertices[key] for key in ("x", "y", "z")])
    faces = mesh["face"].data["vertex_indices"]
    if any(len(face) != 3 for face in faces):
        raise ValueError("Fallback trimming requires triangular Poisson faces")
    triangles = np.stack(faces)
    distances, _ = tree.query(positions, workers=threads, distance_upper_bound=radius)
    centroid_distances, _ = tree.query(positions[triangles].mean(axis=1), workers=threads, distance_upper_bound=radius)
    keep = np.isfinite(distances[triangles]).all(axis=1) & np.isfinite(centroid_distances)
    triangles = triangles[keep]
    if not len(triangles):
        raise ValueError("No mesh faces have sufficient support in the dense cloud")
    used = np.unique(triangles)
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[used] = np.arange(len(used))
    new_faces = np.empty(len(triangles), dtype=[("vertex_indices", "i4", (3,))])
    new_faces["vertex_indices"] = remap[triangles]
    PlyData([PlyElement.describe(vertices[used], "vertex"), PlyElement.describe(new_faces, "face")],
            text=False, byte_order="<").write(output)
    return {"raw_vertices": len(vertices), "raw_faces": len(faces), "removed_unsupported_faces": int((~keep).sum()),
            "support_distance_arbitrary_units": radius, "support_distance_cells": distance_cells,
            "support_test": "all triangle vertices and centroid within distance bound of observed dense points"}


def generate_mesh(dense_cloud, destination, config, logs):
    import pycolmap as p
    output = destination / "building_mesh.ply"
    options = p.PoissonMeshingOptions(depth=config.MESH_DEPTH, num_threads=config.NUM_THREADS)
    p.poisson_meshing(dense_cloud, output, options=options)
    diagnostics = {"method": "COLMAP Poisson", "options": options.todict()}
    try:
        validate_ply(output, require_faces=True)
    except (ValueError, OSError) as exc:
        # Do not reinterpret corrupt numbers or invent replacement coordinates.
        # Reconstruct again from the original oriented cloud, without native trim.
        if output.exists():
            output.replace(logs / "invalid_native_poisson_mesh.ply")
        diagnostics.update(native_trim_failure=str(exc), method="COLMAP Poisson with observed-point support trimming")
        options.trim = 0
        raw = logs / "untrimmed_poisson_mesh.ply"
        p.poisson_meshing(dense_cloud, raw, options=options)
        diagnostics["fallback_options"] = options.todict()
        diagnostics["support_trimming"] = trim_to_observations(
            raw, dense_cloud, output, config.MESH_DEPTH, config.MESH_MAX_DISTANCE_CELLS, config.NUM_THREADS)
    return validate_ply(output, require_faces=True), diagnostics
