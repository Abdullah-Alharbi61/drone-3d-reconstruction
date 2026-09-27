"""One COLMAP step per process: contain native failures and release caches."""
import json
from pathlib import Path
import sqlite3
import sys

import pycolmap as p

from drone_frames.io import save_json
from .config import Config
from .inputs import json_safe
from .validation import validate_ply, write_camera_ply


def wider_pairs(names, stride, overlap):
    selected = list(range(0, len(names), stride))
    if len(names) - 1 not in selected:
        selected.append(len(names) - 1)
    return [(names[a], names[b]) for i, a in enumerate(selected) for b in selected[i + 1:] if b - a > overlap]


def main():
    root, step = Path(sys.argv[1]), sys.argv[2]
    config = Config.load(root / "config.json")
    summary = json.loads((root / "input_summary.json").read_text())
    hardware = json.loads((root / "hardware.json").read_text())
    names = [im["frame"] for im in summary["images"]]
    images = Path(summary["input_directory"])
    database = root / "database.db"
    sparse, dense = root / "sparse", root / "dense"
    gpu = hardware["gpu_enabled"] and "--cpu" not in sys.argv
    device = p.Device.cuda if gpu else p.Device.cpu
    effective = {}
    if step == "features":
        reader = p.ImageReaderOptions(camera_model=config.CAMERA_MODEL if config.CAMERA_MODEL != "AUTO" else "SIMPLE_RADIAL",
                                      camera_params=config.CAMERA_PARAMS)
        options = p.FeatureExtractionOptions(num_threads=config.NUM_THREADS, max_image_size=config.MAX_IMAGE_SIZE, use_gpu=gpu)
        options.sift.max_num_features = config.MAX_FEATURES
        effective = {"reader": reader.todict(), "extraction": options.todict(), "device": str(device)}
        p.extract_features(database, images, image_names=names,
                           camera_mode=p.CameraMode.SINGLE if config.SINGLE_CAMERA else p.CameraMode.AUTO,
                           reader_options=reader, extraction_options=options, device=device)
        with sqlite3.connect(database) as db:
            rows = db.execute("SELECT images.name, keypoints.rows FROM images LEFT JOIN keypoints USING(image_id)").fetchall()
        if len(rows) != len(names) or any(count is None or count == 0 for _, count in rows):
            raise ValueError("Feature extraction did not produce features for every input image")
        save_json(root / "feature_stats.json", {"images": len(rows), "total_features": sum(c for _, c in rows),
                                               "min_features": min(c for _, c in rows), "max_features": max(c for _, c in rows)})
    elif step == "matching":
        options = p.FeatureMatchingOptions(num_threads=config.NUM_THREADS, use_gpu=gpu, guided_matching=True,
                                            max_num_matches=config.MAX_FEATURES)
        effective = {"matching": options.todict(), "device": str(device)}
        if config.MATCHING_STRATEGY == "sequential":
            pairing = p.SequentialPairingOptions(overlap=config.MATCH_OVERLAP, quadratic_overlap=False,
                                                  loop_detection=False, num_threads=config.NUM_THREADS)
            effective["pairing"] = pairing.todict()
            p.match_sequential(database, matching_options=options, pairing_options=pairing, device=device)
            if config.LOOP_DETECTION:
                pairs = wider_pairs(names, config.LOOP_KEYFRAME_STRIDE, config.MATCH_OVERLAP)
                pair_path = root / "loop_pairs.txt"
                pair_path.write_text("".join(f"{a} {b}\n" for a, b in pairs))
                effective["loop_method"] = "exhaustive matching between periodic keyframes, with geometric verification"
                effective["loop_pair_count"] = len(pairs)
                p.match_image_pairs(database, matching_options=options,
                                    pairing_options=p.ImportedPairingOptions(match_list_path=pair_path, block_size=100), device=device)
        else:
            p.match_exhaustive(database, matching_options=options, device=device)
        with sqlite3.connect(database) as db:
            count, inliers = db.execute("SELECT COUNT(*), COALESCE(SUM(rows),0) FROM two_view_geometries WHERE rows > 0").fetchone()
        if count == 0:
            raise ValueError("No geometrically verified matches; inspect overlap, texture, blur, and repetitive patterns")
        save_json(root / "matching_stats.json", {"verified_pairs": count, "verified_correspondences": inliers})
    elif step == "sparse":
        models = root / "sparse_models"
        models.mkdir(exist_ok=True)
        options = p.IncrementalPipelineOptions(num_threads=config.NUM_THREADS, image_names=names, random_seed=0)
        options.mapper.init_min_tri_angle = config.INIT_MIN_TRI_ANGLE
        options.ba_refine_focal_length = config.REFINE_INTRINSICS
        options.ba_refine_extra_params = config.REFINE_INTRINSICS
        options.mapper.abs_pose_refine_focal_length = config.REFINE_INTRINSICS
        options.mapper.abs_pose_refine_extra_params = config.REFINE_INTRINSICS
        options.ba_local_max_num_iterations = 25
        options.ba_global_max_num_iterations = 50
        options.ba_global_max_refinements = 2
        options.snapshot_path = root / "snapshots"
        options.snapshot_path.mkdir(exist_ok=True)
        options.snapshot_frames_freq = 50
        effective = options.todict()
        maps = p.incremental_mapping(database, images, models, options=options)
        if not maps:
            raise ValueError("COLMAP produced no sparse model: insufficient baseline, overlap, or reliable matches")
        model = max(maps.values(), key=lambda m: (m.num_reg_images(), m.num_points3D()))
        sparse.mkdir(exist_ok=True)
        model.write(sparse)
        model.write_text(sparse)
        model.export_PLY(sparse / "sparse.ply")
        registered_names = {model.image(i).name for i in model.reg_image_ids()}
        stats = {"input_images": len(names), "registered_images": model.num_reg_images(),
                 "unregistered_images": len(names) - model.num_reg_images(), "registration_rate": model.num_reg_images() / len(names),
                 "unregistered_image_names": [n for n in names if n not in registered_names],
                 "sparse_points": model.num_points3D(), "mean_observations_per_point": model.compute_mean_track_length(),
                 "mean_reprojection_error_px": model.compute_mean_reprojection_error(),
                 "components": [{"id": i, "registered_images": m.num_reg_images(), "sparse_points": m.num_points3D()} for i, m in maps.items()]}
        save_json(root / "sparse_stats.json", stats)
        metadata = {im["frame"]: im for im in summary["images"]}
        poses = []
        for image_id in sorted(model.reg_image_ids(), key=lambda i: model.image(i).name):
            im = model.image(image_id)
            camera = model.camera(im.camera_id)
            transform = im.cam_from_world()
            poses.append({"frame": im.name, "image_id": image_id, "camera_id": im.camera_id,
                          "timestamp_seconds": metadata[im.name]["timestamp_seconds"],
                          "position_world": im.projection_center().tolist(),
                          "rotation_world_to_camera": transform.rotation.matrix().tolist(),
                          "quaternion_world_to_camera_xyzw": transform.rotation.quat.tolist(),
                          "translation_world_to_camera": transform.translation.tolist(),
                          "camera_model": camera.model.name, "camera_params": camera.params.tolist(),
                          "intrinsic_matrix": camera.calibration_matrix().tolist()})
        save_json(sparse / "camera_poses.json", {"units": "arbitrary", "convention": "x right, y down, z forward in camera space; X_cam = R X_world + t", "images": poses})
        write_camera_ply(sparse / "camera_locations.ply", [im["position_world"] for im in poses])
        validate_ply(sparse / "sparse.ply")
    elif step == "undistort":
        options = p.UndistortCameraOptions(max_image_size=config.DENSE_MAX_IMAGE_SIZE)
        effective = options.todict()
        p.undistort_images(dense, sparse, images, num_patch_match_src_images=config.DENSE_NUM_SOURCE_IMAGES,
                           undistort_options=options, num_threads=config.NUM_THREADS)
    elif step == "depth":
        if not gpu or not p.has_cuda:
            raise ValueError("COLMAP dense stereo requires CUDA; sparse results remain available")
        options = p.PatchMatchOptions(max_image_size=config.DENSE_MAX_IMAGE_SIZE, gpu_index="0", num_threads=config.NUM_THREADS,
                                     num_iterations=config.DENSE_ITERATIONS, cache_size=config.DENSE_CACHE_GB, geom_consistency=True)
        effective = options.todict()
        p.patch_match_stereo(dense, options=options)
        expected = p.Reconstruction(sparse).num_reg_images()
        maps = list((dense / "stereo" / "depth_maps").glob("*.geometric.bin"))
        if len(maps) != expected or any(path.stat().st_size == 0 for path in maps):
            raise ValueError(f"Expected {expected} geometric depth maps, found {len(maps)}")
    elif step == "fusion":
        options = p.StereoFusionOptions(num_threads=config.NUM_THREADS, max_image_size=config.DENSE_MAX_IMAGE_SIZE,
                                       min_num_pixels=config.FUSION_MIN_PIXELS, use_cache=True, cache_size=config.FUSION_CACHE_GB)
        effective = options.todict()
        p.stereo_fusion(dense / "building_dense.ply", dense, options=options, output_type="ply")
        save_json(root / "dense_stats.json", validate_ply(dense / "building_dense.ply"))
    elif step == "mesh":
        from .meshing import generate_mesh
        destination = root / "mesh"
        destination.mkdir(exist_ok=True)
        stats, effective = generate_mesh(dense / "building_dense.ply", destination, config, root / "logs")
        save_json(root / "mesh_stats.json", stats)
        save_json(root / "mesh_diagnostics.json", effective)
    else:
        raise ValueError(f"Unknown step: {step}")
    save_json(root / "logs" / f"{step}_options.json", json_safe(effective))


if __name__ == "__main__":
    main()
