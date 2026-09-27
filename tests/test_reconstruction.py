from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from reconstruction.config import Config
from reconstruction.inputs import inspect_inputs
from reconstruction.validation import validate_ply, validate_sparse
from reconstruction.worker import wider_pairs
from reconstruction.__main__ import main
from reconstruction.meshing import trim_to_observations


class ReconstructionTests(unittest.TestCase):
    def test_sparse_gate_rejects_poor_registration_and_bad_error(self):
        stats = {"registration_rate": .95, "sparse_points": 10000, "mean_observations_per_point": 4.5,
                 "mean_reprojection_error_px": .6}
        self.assertEqual(validate_sparse(stats, Config()), [])
        stats.update(registration_rate=.2, mean_reprojection_error_px=8)
        self.assertEqual(len(validate_sparse(stats, Config())), 2)
        stats.update(mean_reprojection_error_px=float("nan"))
        self.assertEqual(len(validate_sparse(stats, Config())), 2)

    def test_only_accepted_report_members_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted = root / "accepted_frames"
            accepted.mkdir()
            records = []
            for i in range(3):
                name = f"frame_{i:06d}.jpg"
                Image.new("RGB", (128, 72), (i * 40, 100, 150)).save(accepted / name)
                records.append({"frame": name, "status": "accepted", "timestamp_seconds": i * .4, "original_frame_number": i * 24})
            records.append({"frame": "rejected.jpg", "status": "rejected"})
            report = root / "report.json"
            report.write_text(json.dumps({"frames": records}))
            summary = inspect_inputs(accepted, report)
            self.assertEqual(summary["input_images"], 3)
            self.assertEqual(summary["resolutions"], ["128x72"])
            self.assertEqual(summary["exif_images"], 0)
            Image.new("RGB", (128, 72)).save(accepted / "rejected.jpg")
            with self.assertRaisesRegex(ValueError, "disagree"):
                inspect_inputs(accepted, report)

    def test_loop_pairs_include_distant_views_without_local_duplicates(self):
        names = [f"frame_{i:06d}.jpg" for i in range(32)]
        pairs = wider_pairs(names, stride=10, overlap=10)
        self.assertIn((names[0], names[-1]), pairs)
        self.assertNotIn((names[0], names[10]), pairs)
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_ply_actual_read_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "points.ply"
            vertices = np.array([(0, 0, 0), (1, 2, 3)], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
            PlyData([PlyElement.describe(vertices, "vertex")]).write(path)
            self.assertEqual(validate_ply(path)["vertices"], 2)
            with self.assertRaisesRegex(ValueError, "no faces"):
                validate_ply(path, require_faces=True)
            vertices[1]["z"] = np.nan
            PlyData([PlyElement.describe(vertices, "vertex")]).write(path)
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                validate_ply(path)

    def test_mesh_trimming_removes_only_unsupported_faces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
            observed = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1/3, 1/3, 0)], dtype=dtype)
            vertices = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (10, 0, 0), (11, 0, 0), (10, 1, 0)], dtype=dtype)
            faces = np.empty(2, dtype=[("vertex_indices", "i4", (3,))])
            faces["vertex_indices"] = [[0, 1, 2], [3, 4, 5]]
            PlyData([PlyElement.describe(observed, "vertex")]).write(root / "dense.ply")
            PlyData([PlyElement.describe(vertices, "vertex"), PlyElement.describe(faces, "face")]).write(root / "raw.ply")
            stats = trim_to_observations(root / "raw.ply", root / "dense.ply", root / "trimmed.ply", 2, 2, 1)
            self.assertEqual(stats["removed_unsupported_faces"], 1)
            self.assertEqual(validate_ply(root / "trimmed.ply", require_faces=True)["faces"], 1)
            result = PlyData.read(root / "trimmed.ply")["vertex"].data
            np.testing.assert_array_equal(result, vertices[:3])

    def test_config_rejects_invalid_settings(self):
        for config in [replace(Config(), MATCH_OVERLAP=0), replace(Config(), DENSE_CACHE_GB=float("nan")),
                       replace(Config(), USE_GPU="true"), replace(Config(), MIN_REGISTRATION_RATE=1.1),
                       replace(Config(), CAMERA_PARAMS="100,50,50,0"), replace(Config(), RUN_DENSE_RECONSTRUCTION=False)]:
            with self.assertRaises(ValueError):
                config.validate()

    def test_native_failure_is_reported_and_stops_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted = root / "accepted_frames"
            accepted.mkdir()
            records = []
            for i in range(3):
                name = f"frame_{i:06d}.jpg"
                Image.new("RGB", (128, 72), (i * 40, 100, 150)).save(accepted / name)
                records.append({"frame": name, "status": "accepted", "timestamp_seconds": i * .4, "original_frame_number": i * 24})
            frames = root / "frames.json"
            frames.write_text(json.dumps({"frames": records}))
            config = root / "config.json"
            config.write_text(json.dumps({"USE_GPU": False, "FUSION_CACHE_GB": .25, "SOURCE_VIDEO": str(root / "missing.mp4")}))
            output = root / "reconstruction"
            with patch("reconstruction.__main__.run_worker", side_effect=RuntimeError("native engine failed")) as worker:
                code = main(["--images", str(accepted), "--frame-report", str(frames), "--config", str(config), "--output", str(output)])
            self.assertEqual(code, 1)
            self.assertEqual(worker.call_count, 1)
            report = json.loads((output / "reconstruction_report.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["dense_reconstruction_completed"])
            self.assertIn("native engine failed", report["errors"][0])
            settings = json.loads(config.read_text())
            settings["FUSION_CACHE_GB"] = 2.0
            config.write_text(json.dumps(settings))
            arguments = ["--images", str(accepted), "--frame-report", str(frames), "--config", str(config), "--output", str(output), "--resume"]
            with patch("reconstruction.__main__.run_worker", side_effect=RuntimeError("retry failed")) as worker:
                self.assertEqual(main(arguments), 1)
                self.assertEqual(worker.call_count, 1)
            resumed = json.loads((output / "reconstruction_report.json").read_text())
            self.assertEqual(resumed["parameters"]["FUSION_CACHE_GB"], 2.0)
            self.assertEqual(resumed["previous_parameters"][0]["FUSION_CACHE_GB"], .25)
            settings["MATCH_OVERLAP"] = 5
            config.write_text(json.dumps(settings))
            with patch("reconstruction.__main__.run_worker") as worker:
                self.assertEqual(main(arguments), 1)
                worker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
