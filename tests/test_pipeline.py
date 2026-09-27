from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from drone_frames.__main__ import main
from drone_frames.config import Config
from drone_frames.frame_extractor import select_indices
from drone_frames.quality_filter import analysis_gray, filter_frames, image_metrics, is_duplicate, pair_metrics, quality_reasons


def textured_image():
    rng = np.random.default_rng(42)
    image = np.full((360, 640, 3), 120, dtype=np.uint8)
    for _ in range(200):
        x, y = rng.integers([0, 0], [620, 340])
        color = int(rng.integers(20, 240))
        cv2.rectangle(image, (int(x), int(y)), (int(x + 15), int(y + 15)), (color,) * 3, -1)
    return image


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.image = textured_image()
        self.gray = analysis_gray(self.image, self.config)

    def test_sharp_structure_survives(self):
        self.assertEqual(quality_reasons(image_metrics(self.gray, self.config), self.config), [])

    def test_heavy_blur_rejected(self):
        blurred = cv2.GaussianBlur(self.gray, (0, 0), 4)
        self.assertIn("blur", quality_reasons(image_metrics(blurred, self.config), self.config))

    def test_dark_bright_and_featureless_rejected(self):
        for value, reason in [(0, "underexposed"), (255, "overexposed"), (120, "insufficient_structure")]:
            with self.subTest(value=value):
                self.assertIn(reason, quality_reasons(image_metrics(np.full_like(self.gray, value), self.config), self.config))

    def test_bright_sky_with_textured_facade_survives(self):
        self.gray[:180] = 255
        reasons = quality_reasons(image_metrics(self.gray, self.config), self.config)
        self.assertNotIn("overexposed", reasons)
        self.assertNotIn("insufficient_structure", reasons)

    def test_identical_and_tiny_noise_are_duplicates(self):
        self.assertTrue(is_duplicate(pair_metrics(self.gray, self.gray, self.config), self.config))
        noisy = self.gray.copy()
        noisy[::5, ::5] += 1
        self.assertTrue(is_duplicate(pair_metrics(self.gray, noisy, self.config), self.config))

    def test_small_viewpoint_change_is_preserved(self):
        moved = cv2.warpAffine(self.gray, np.float32([[1, 0, 2], [0, 1, 0]]), (640, 360), borderMode=cv2.BORDER_REFLECT)
        pair = pair_metrics(self.gray, moved, self.config)
        self.assertFalse(is_duplicate(pair, self.config))
        self.assertGreater(pair["motion_median_pixels"], 1.5)
        self.assertGreater(pair["tracked_fraction"], .8)

    def test_duplicate_anchor_corruption_and_last_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "extracted"
            source.mkdir()
            records = []
            for index, timestamp in enumerate([0, .4, .8, 1.2, 1.6, 2.0]):
                name = f"frame_{index + 1:06d}.jpg"
                cv2.imwrite(str(source / name), self.image)
                records.append({"frame": name, "timestamp_seconds": timestamp, "original_frame_number": index * 24})
            (source / records[4]["frame"]).write_bytes(b"not an image")
            frames = filter_frames(records, source, root / "accepted", root / "rejected", self.config)
            self.assertEqual([f["status"] for f in frames], ["accepted", "rejected", "rejected", "accepted", "rejected", "accepted"])
            self.assertEqual(frames[4]["rejection_reason"], "corrupted_image")
            self.assertEqual(frames[2]["duplicate_reference"], records[0]["frame"])
            self.assertEqual((root / "rejected" / records[4]["frame"]).read_bytes(), b"not an image")

    def test_motion_alone_does_not_reject_sharp_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "extracted"
            source.mkdir()
            records = []
            for i in range(2):
                name = f"frame_{i:06d}.jpg"
                cv2.imwrite(str(source / name), self.image)
                records.append({"frame": name, "timestamp_seconds": i * .4, "original_frame_number": i * 24})
            pair = pair_metrics(self.gray, self.gray, self.config)
            pair["motion_median_pixels"] = 200
            with patch("drone_frames.quality_filter.pair_metrics", return_value=pair):
                frames = filter_frames(records, source, root / "accepted", root / "rejected", self.config)
            self.assertEqual(frames[1]["status"], "accepted")
            self.assertIn("large_view_change_review_overlap", frames[1]["warnings"])
            with patch("drone_frames.quality_filter.pair_metrics", return_value=pair):
                frames = filter_frames(records, source, root / "accepted2", root / "rejected2", replace(self.config, MOTION_BLUR_THRESHOLD=10000))
            self.assertEqual(frames[1]["rejection_reason"], "extreme_motion")

    def test_manual_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "extracted"
            source.mkdir()
            cv2.imwrite(str(source / "one.jpg"), self.image)
            frames = filter_frames([{"frame": "one.jpg", "timestamp_seconds": 2.0, "original_frame_number": 120}], source,
                                   root / "accepted", root / "rejected", replace(self.config, EXCLUDE_TIME_RANGES=[[1, 3]]))
            self.assertEqual(frames[0]["rejection_reason"], "manual_exclusion")


class ExtractionTests(unittest.TestCase):
    def test_fractional_fps_full_timeline(self):
        times = np.arange(7451) / (60000 / 1001)
        indices = select_indices(times.tolist(), .4)
        self.assertEqual(len(indices), 312)
        self.assertEqual((indices[0], indices[-1]), (0, 7450))
        self.assertLessEqual(np.diff(times[indices]).max(), .4 + 1 / (60000 / 1001))

    def test_variable_rate_selection(self):
        self.assertEqual(select_indices([0, .02, .13, .41, .8, .82, 1.3], .4), [0, 3, 4, 6])

    def test_config_validation(self):
        for config in [replace(Config(), FRAME_EXTRACTION_INTERVAL=0), replace(Config(), BLUR_THRESHOLD=float("nan")),
                       replace(Config(), DUPLICATE_THRESHOLD=1.2), replace(Config(), ANALYSIS_WIDTH=640.5),
                       replace(Config(), EXCLUDE_TIME_RANGES=[[4, 2]])]:
            with self.assertRaises(ValueError):
                config.validate()

    @unittest.skipUnless(shutil.which("ffprobe"), "ffprobe required")
    def test_end_to_end_video_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "input.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 360))
            self.assertTrue(writer.isOpened())
            im = textured_image()
            for i in range(31):
                moved = cv2.warpAffine(im, np.float32([[1, 0, i], [0, 1, 0]]), (640, 360), borderMode=cv2.BORDER_REFLECT)
                writer.write(moved)
            writer.release()
            output = root / "output"
            arguments = ["run", str(path), "--output", str(output), "--interval", "0.4"]
            self.assertEqual(main(arguments), 0)
            report = json.loads((output / "frame_report.json").read_text())
            self.assertEqual(report["video"]["total_frame_count"], 31)
            self.assertEqual(report["summary"]["extracted"], 9)
            self.assertEqual(report["frames"][-1]["original_frame_number"], 30)
            self.assertEqual(report["frames"][-1]["timestamp_seconds"], 3.0)
            accepted = {p.name for p in (output / "accepted_frames").iterdir()}
            rejected = {p.name for p in (output / "rejected_frames").iterdir()}
            extracted = {p.name for p in (output / "extracted_frames").iterdir()}
            self.assertFalse(accepted & rejected)
            self.assertEqual(accepted | rejected, extracted)
            before = (output / "frame_report.json").read_bytes()
            self.assertEqual(main(arguments), 1)
            self.assertEqual((output / "frame_report.json").read_bytes(), before)
            self.assertEqual(main(["run", str(path), "--output", str(root / "all"), "--interval", ".01"]), 1)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required")
    def test_real_variable_rate_timestamps(self):
        from drone_frames.video import probe_video
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cv2.imwrite(str(root / "sample.png"), textured_image())
            (root / "list.txt").write_text("file 'sample.png'\nduration 0.12\nfile 'sample.png'\nduration 0.68\nfile 'sample.png'\nduration 0.2\nfile 'sample.png'\n")
            subprocess.run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(root / "list.txt"),
                            "-vsync", "vfr", "-c:v", "ffv1", str(root / "variable.mkv")], check=True)
            video = probe_video(root / "variable.mkv")
            self.assertTrue(video.variable_frame_rate)
            np.testing.assert_allclose(video.timestamps, [0, .12, .8, 1.0], atol=.001)


if __name__ == "__main__":
    cv2.setNumThreads(2)
    unittest.main()
