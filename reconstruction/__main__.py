import argparse
import ctypes
import fcntl
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from drone_frames.io import save_json
from .config import Config
from .inputs import inspect_inputs, preserve_video_metadata
from .validation import validate_ply, validate_sparse

LOG = logging.getLogger(__name__)


def detect_hardware(config):
    import pycolmap
    result = {"colmap_version": pycolmap.__version__, "colmap_cuda_support": bool(pycolmap.has_cuda),
              "existing_colmap_cli": shutil.which("colmap"), "cuda_available": False, "gpu_enabled": False}
    try:
        driver = ctypes.CDLL("libcuda.so.1")
        count = ctypes.c_int()
        result["cuda_available"] = driver.cuInit(0) == 0 and driver.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value > 0
        result["cuda_device_count"] = count.value
    except OSError:
        pass
    if shutil.which("nvidia-smi"):
        result["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                              capture_output=True, text=True).stdout.strip()
    result["gpu_enabled"] = config.USE_GPU and result["cuda_available"] and result["colmap_cuda_support"]
    return result


def run_worker(root, step, cpu=False):
    command = [sys.executable, "-m", "reconstruction.worker", str(root.resolve()), step]
    if cpu:
        command.append("--cpu")
    environment = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", QT_QPA_PLATFORM="offscreen")
    log_path = root / "logs" / f"{step}{'_cpu' if cpu else ''}.log"
    with log_path.open("a") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        last_update = time.monotonic()
        try:
            while process.poll() is None:
                time.sleep(1)
                if time.monotonic() - last_update >= 30:
                    LOG.info("%s is running; progress log: %s", step, log_path)
                    last_update = time.monotonic()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if process.returncode:
        tail = log_path.read_text(errors="replace")[-3000:]
        raise RuntimeError(f"COLMAP {step} exited with code {process.returncode}; see {log_path}\n{tail}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stage 4 only: COLMAP SfM and dense reconstruction in arbitrary units")
    parser.add_argument("--images", type=Path, default=Path("output/accepted_frames"))
    parser.add_argument("--frame-report", type=Path, default=Path("output/frame_report.json"))
    parser.add_argument("--output", type=Path, default=Path("output/reconstruction"))
    parser.add_argument("--config", type=Path, default=Path("config.reconstruction.json"))
    parser.add_argument("--resume", action="store_true", help="Reuse completed steps with identical inputs and configuration")
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = args.output
    report = None
    started = time.monotonic()
    previous_runtime = 0
    lock = None
    handler = None
    try:
        config = Config.load(args.config)
        if root.exists() and any(root.iterdir()) and not args.resume:
            raise ValueError(f"Output already exists: {root}; use --resume or choose another output")
        LOG.info("[1/6] Loading accepted frames")
        summary = inspect_inputs(args.images, args.frame_report)
        if config.SINGLE_CAMERA and len(summary["resolutions"]) != 1:
            raise ValueError("SINGLE_CAMERA requires consistent resolution")
        if args.resume and (root / "config.json").exists():
            old_config = asdict(Config.load(root / "config.json"))
            changed_keys = {key for key, value in asdict(config).items() if old_config.get(key) != value}
            if changed_keys - {"DENSE_CACHE_GB", "FUSION_CACHE_GB"}:
                raise ValueError("Resume configuration changed; use a new output to avoid stale geometry")
            old_summary = json.loads((root / "input_summary.json").read_text())
            if old_summary["input_fingerprint"] != summary["input_fingerprint"]:
                raise ValueError("Resume input images changed")
        root.mkdir(parents=True, exist_ok=True)
        lock = (root / ".run.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"Another reconstruction process is using {root}")
        (root / "logs").mkdir(exist_ok=True)
        handler = logging.FileHandler(root / "pipeline.log")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(handler)
        hardware = detect_hardware(config)
        summary["calibrated_intrinsics_available"] = bool(config.CAMERA_PARAMS)
        summary["intrinsics_source"] = "explicit camera parameters" if config.CAMERA_PARAMS else "COLMAP EXIF focal initialization if available, otherwise estimated; refined by SfM"
        summary["source_metadata"] = preserve_video_metadata(Path(config.SOURCE_VIDEO), root / "metadata")
        save_json(root / "input_summary.json", summary)
        save_json(root / "config.json", asdict(config))
        save_json(root / "hardware.json", hardware)
        (root / "image_list.txt").write_text("\n".join(im["frame"] for im in summary["images"]) + "\n")
        report_path = root / "reconstruction_report.json"
        report = {"stage": 4, "status": "running", "reconstruction_engine": "COLMAP via official PyCOLMAP",
                  "engine_version": hardware["colmap_version"], "parameters": asdict(config), "hardware": hardware,
                  "camera_model": config.CAMERA_MODEL if config.CAMERA_MODEL != "AUTO" else "SIMPLE_RADIAL",
                  "matching_method": config.MATCHING_STRATEGY, "loop_matching_method": "periodic keyframe pairs" if config.LOOP_DETECTION else "disabled",
                  "input_images": summary["input_images"], "registered_images": 0, "registration_rate": 0,
                  "unregistered_images": summary["input_images"], "sparse_points": 0, "dense_points": 0,
                  "mean_reprojection_error_px": None, "dense_reconstruction_completed": False, "mesh_generated": False,
                  "coordinate_units": "arbitrary; no real-world scale applied", "completed_steps": [], "step_runtime_seconds": {},
                  "runtime_scope": "Recorded previous invocations plus current invocation; uncheckpointed work from interrupted processes may be excluded.",
                  "warnings": [], "errors": []}
        if args.resume and report_path.exists():
            old_report = json.loads(report_path.read_text())
            previous_runtime = old_report.get("runtime_seconds", 0)
            report.update(old_report)
            if old_report.get("parameters") != asdict(config):
                report.setdefault("previous_parameters", []).append(old_report["parameters"])
                report["warnings"].append("Runtime settings/defaults updated on resume; completed geometry stages were reused under compatible settings.")
            report["parameters"] = asdict(config)
            report["hardware"] = hardware
            if old_report.get("errors"):
                report.setdefault("previous_attempt_errors", []).extend(old_report["errors"])
            report.update(status="running", errors=[])
        if not hardware["gpu_enabled"]:
            report["warnings"].append("GPU unavailable or disabled: feature extraction and matching use CPU. COLMAP dense stereo requires CUDA.")
        if len(summary["source_metadata"].get("data_streams", [])):
            report["warnings"].append("Proprietary DJI metadata preserved without decoding; GPS/RTK availability inside it is unknown.")
        LOG.info("Input images: %d | Resolution: %s | EXIF images: %d | Calibrated intrinsics: %s | GPU: %s",
                 summary["input_images"], summary["resolutions"], summary["exif_images"], bool(config.CAMERA_PARAMS), hardware["gpu_enabled"])

        def checkpoint():
            report["runtime_seconds"] = previous_runtime + time.monotonic() - started
            report["warnings"] = list(dict.fromkeys(report["warnings"]))
            save_json(report_path, report)

        def step(name):
            if name in report["completed_steps"]:
                required = {
                    "features": ["database.db", "feature_stats.json"],
                    "matching": ["database.db", "matching_stats.json"],
                    "sparse": ["sparse/cameras.bin", "sparse/images.bin", "sparse/points3D.bin", "sparse/camera_poses.json", "sparse_stats.json"],
                    "undistort": ["dense/sparse/cameras.bin", "dense/stereo/patch-match.cfg"],
                    "depth": ["dense/stereo/patch-match.cfg"],
                    "fusion": ["dense/building_dense.ply", "dense_stats.json"],
                    "mesh": ["mesh/building_mesh.ply", "mesh_stats.json"],
                }[name]
                if any(not (root / path).is_file() or (root / path).stat().st_size == 0 for path in required):
                    raise ValueError(f"Completed step {name} has missing/empty artifacts; use a new output directory")
                if name == "depth":
                    expected = json.loads((root / "sparse_stats.json").read_text())["registered_images"]
                    maps = list((root / "dense/stereo/depth_maps").glob("*.geometric.bin"))
                    if len(maps) != expected or any(path.stat().st_size == 0 for path in maps):
                        raise ValueError("Completed depth step has missing/empty maps")
                LOG.info("Resuming: %s already completed", name)
                return
            report["current_step"] = name
            checkpoint()
            start = time.monotonic()
            try:
                run_worker(root, name)
            except RuntimeError as exc:
                if name in ("features", "matching") and hardware["gpu_enabled"]:
                    report["warnings"].append(f"{name}: GPU execution failed; retried on CPU. Original error: {exc}")
                    LOG.warning("GPU %s failed; retrying on CPU", name)
                    run_worker(root, name, cpu=True)
                else:
                    raise
            report["step_runtime_seconds"][name] = time.monotonic() - start
            report["completed_steps"].append(name)
            checkpoint()

        checkpoint()
        if args.inspect_only:
            report["status"] = "inspected"
            checkpoint()
            return 0
        LOG.info("[2/6] Extracting features")
        step("features")
        report["feature_statistics"] = json.loads((root / "feature_stats.json").read_text())
        LOG.info("[3/6] Matching features")
        step("matching")
        report["matching_statistics"] = json.loads((root / "matching_stats.json").read_text())
        LOG.info("[4/6] Sparse reconstruction")
        step("sparse")
        stats = json.loads((root / "sparse_stats.json").read_text())
        report.update(stats)
        report["registration_percentage"] = stats["registration_rate"] * 100
        checkpoint()
        failures = validate_sparse(stats, config)
        if failures:
            raise ValueError("Sparse validation failed: " + "; ".join(failures) +
                             ". Possible causes: insufficient overlap/baseline, weak texture, repetitive structures, or bad matches.")
        validate_ply(root / "sparse" / "sparse.ply")
        if len(stats["components"]) > 1:
            report["warnings"].append("Multiple sparse components found. Dense reconstruction uses the largest; other components have independent coordinates.")
        LOG.info("Sparse validated: %d/%d registered, %d points, %.3f px error, %.2f observations/point",
                 stats["registered_images"], stats["input_images"], stats["sparse_points"], stats["mean_reprojection_error_px"], stats["mean_observations_per_point"])
        LOG.info("[5/6] Dense reconstruction")
        if config.RUN_DENSE_RECONSTRUCTION:
            if not hardware["gpu_enabled"]:
                raise RuntimeError("Sparse reconstruction completed, but CUDA is unavailable/disabled; dense stereo cannot run. No CPU dense substitute is fabricated.")
            step("undistort")
            step("depth")
            step("fusion")
            dense_stats = validate_ply(root / "dense" / "building_dense.ply")
            report["dense_points"] = dense_stats["vertices"]
            report["dense_validation"] = dense_stats
            if report["dense_points"] < max(config.MIN_DENSE_POINTS, stats["sparse_points"] * config.MIN_DENSE_SPARSE_RATIO):
                raise ValueError("Dense cloud is too small relative to the sparse model; inspect depth maps and camera calibration")
            report["dense_reconstruction_completed"] = True
            checkpoint()
            if config.GENERATE_MESH:
                if report["dense_points"] >= config.MIN_MESH_POINTS:
                    step("mesh")
                    report["mesh_validation"] = validate_ply(root / "mesh" / "building_mesh.ply", require_faces=True)
                    diagnostics = json.loads((root / "mesh_diagnostics.json").read_text())
                    report["mesh_diagnostics"] = diagnostics
                    if diagnostics.get("native_trim_failure"):
                        report["warnings"].append("Native Poisson trimming produced invalid vertices; regenerated with COLMAP and trimmed against observed dense points. See mesh_diagnostics.json.")
                    report["mesh_generated"] = True
                else:
                    report["warnings"].append("Mesh skipped: dense cloud below MIN_MESH_POINTS")
        LOG.info("[6/6] Saving outputs")
        report["status"] = "success" if report["dense_reconstruction_completed"] else "sparse_only"
        report["current_step"] = "complete"
        checkpoint()
        text = (f"3D RECONSTRUCTION {'COMPLETE' if report['dense_reconstruction_completed'] else 'SPARSE ONLY'}\n\n"
                f"Input images: {report['input_images']}\nRegistered: {report['registered_images']} / {report['input_images']}\n"
                f"Registration rate: {report['registration_percentage']:.1f}%\n"
                f"Sparse points: {report['sparse_points']:,}\nDense points: {report['dense_points']:,}\n"
                f"Mean reprojection error: {report['mean_reprojection_error_px']:.3f} px\n"
                f"Sparse model: {root / 'sparse'}\nDense point cloud: {root / 'dense/building_dense.ply' if report['dense_reconstruction_completed'] else 'not generated'}\n"
                f"Mesh: {root / 'mesh/building_mesh.ply' if report['mesh_generated'] else 'not generated'}\n\n"
                "Coordinates are in arbitrary units. No scaling, segmentation, or measurements performed.\n")
        if report["dense_reconstruction_completed"]:
            text += "Ready for Stage 5 — Real-World Scale.\n"
        (root / "summary.txt").write_text(text)
        print(text)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        LOG.error("STATUS: FAILED — %s", exc)
        if report is not None:
            report.update(status="failed", runtime_seconds=previous_runtime + time.monotonic() - started)
            report["errors"].append(str(exc) or "Interrupted")
            save_json(root / "reconstruction_report.json", report)
        return 1
    finally:
        if handler is not None:
            logging.getLogger().removeHandler(handler)
            handler.close()
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    sys.exit(main())
