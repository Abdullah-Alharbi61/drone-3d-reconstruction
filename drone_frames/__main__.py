import argparse
from collections import Counter
from dataclasses import asdict
import json
import logging
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from .config import Config
from .frame_extractor import contact_sheet, extract_frames
from .quality_filter import analysis_gray, filter_frames, image_metrics, pair_metrics
from .video import probe_video
from .io import save_json

LOG = logging.getLogger(__name__)


def inspect(video, output, config):
    indices = sorted(set(np.linspace(0, video.total_frame_count - 1, config.INSPECTION_SAMPLE_COUNT, dtype=int).tolist()))
    cap = cv2.VideoCapture(video.path)
    samples, sheet = [], []
    try:
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, im = cap.read()
            if not ok:
                raise ValueError(f"Cannot inspect frame {index}")
            name = f"sample_{index:06d}.jpg"
            if not cv2.imwrite(str(output / name), im, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY]):
                raise OSError(f"Cannot write {name}")
            gray = analysis_gray(im, config)
            # Calibration helper only: deliberately degraded source, never output data.
            synthetic = cv2.GaussianBlur(gray, (0, 0), 1.0)
            sample = {"frame": name, "original_frame_number": index, "timestamp_seconds": video.timestamps[index],
                      **image_metrics(gray, config), "synthetic_sigma1_blur_score": image_metrics(synthetic, config)["blur_score"]}
            next_index = int(np.searchsorted(video.timestamps, video.timestamps[index] + config.FRAME_EXTRACTION_INTERVAL))
            if next_index < video.total_frame_count:
                cap.set(cv2.CAP_PROP_POS_FRAMES, next_index)
                ok, next_im = cap.read()
                if ok:
                    sample["next_interval_metrics"] = pair_metrics(gray, analysis_gray(next_im, config), config)
            samples.append(sample)
            sheet.append((output / name, f'{video.timestamps[index]:.2f}s | blur {sample["blur_score"]:.0f}'))
    finally:
        cap.release()
    contact_sheet(sheet, output / "contact_sheet.jpg")
    save_json(output / "inspection.json", {"video": video.metadata(), "config": asdict(config), "samples": samples})
    LOG.info("Inspection samples and metrics: %s", output)


def run(video, output, config):
    extracted, accepted, rejected = [output / name for name in ("extracted_frames", "accepted_frames", "rejected_frames")]
    records = extract_frames(video, extracted, config)
    save_json(output / "extraction_manifest.json", {"video": video.metadata(), "frames": records})
    LOG.info("Frames extracted: %d", len(records))
    frames = filter_frames(records, extracted, accepted, rejected, config)
    kept = [f for f in frames if f["status"] == "accepted"]
    primary = Counter(f["rejection_reason"] for f in frames if f["status"] == "rejected")
    all_reasons = Counter(reason for frame in frames for reason in frame["rejection_reasons"])
    warnings = []
    if not kept:
        warnings.append("No frames accepted; revise settings or input footage.")
    else:
        boundaries = [0.0] + [f["timestamp_seconds"] for f in kept] + [video.timestamps[-1]]
        for start, end in zip(boundaries, boundaries[1:]):
            if end - start > config.COVERAGE_WARNING_GAP_SECONDS:
                warnings.append(f"Coverage gap {start:.3f}–{end:.3f}s: inspect rejected images or reduce extraction interval.")
    overlap_warnings = sum(bool(f["warnings"]) for f in kept)
    if overlap_warnings:
        warnings.append(f"{overlap_warnings} accepted frames have overlap review flags; see per-frame warnings.")
    summary = {"extracted": len(frames), "accepted": len(kept), "rejected": len(frames) - len(kept),
               "rejected_by_primary_reason": dict(primary), "all_rejection_reason_counts": dict(all_reasons),
               "max_accepted_gap_seconds": max((b["timestamp_seconds"] - a["timestamp_seconds"] for a, b in zip(kept, kept[1:])), default=None),
               "warnings": warnings}
    report = {"schema_version": 1, "stages": [2, 3], "video": video.metadata(), "config": asdict(config),
              "original_frame_number_base": 0, "timestamp_basis": "presentation time relative to first video frame",
              "limitations": ["Structural metrics do not identify a specific building; use reviewed EXCLUDE_TIME_RANGES when needed.",
                              "Track retention is an overlap diagnostic, not proof of reconstructability or complete building coverage."],
              "summary": summary, "frames": frames}
    save_json(output / "frame_report.json", report)
    review = output / "review"
    review.mkdir()
    for offset in range(0, len(frames), 48):
        page = frames[offset:offset + 48]
        contact_sheet([(extracted / f["frame"], f'{f["frame"][6:12]} {f["timestamp_seconds"]:.1f}s {f["status"]}') for f in page],
                      review / f"page_{offset // 48 + 1:03d}.jpg")
    for warning in warnings:
        LOG.warning(warning)
    LOG.info("Frames accepted: %d", len(kept))
    LOG.info("Frames rejected: %d", summary["rejected"])
    LOG.info("Rejected for blur: %d", all_reasons["blur"])
    LOG.info("Rejected for exposure: %d", all_reasons["underexposed"] + all_reasons["overexposed"])
    LOG.info("Rejected as duplicates: %d", all_reasons["near_duplicate"])
    LOG.info("Rejected for insufficient structure: %d", all_reasons["insufficient_structure"])
    LOG.info("Rejected for extreme motion: %d", all_reasons["extreme_motion"])
    text = (f"Input video: {Path(video.path).name}\nDuration: {video.duration_seconds:.3f} seconds\n"
            f"FPS: {video.fps:.6f}\nResolution: {video.width} x {video.height}\n"
            f"Original frames: {video.total_frame_count}\nExtracted: {len(frames)} frames\n"
            f"Accepted: {len(kept)} frames\nRejected: {summary['rejected']} frames\n\n")
    if kept:
        text += "Accepted frames are ready for Stage 4 — 3D Reconstruction.\n"
        text += "This means quality-filtered input; reconstruction success and building coverage are not verified.\n"
    else:
        text += "No accepted dataset is ready for Stage 4.\n"
    if warnings:
        text += "\nReview warnings:\n" + "\n".join(warnings) + "\n"
    (output / "summary.txt").write_text(text)
    print("\n" + text)
    return 0 if kept else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stages 2 and 3: inspect, extract, and filter overlapping drone frames")
    parser.add_argument("command", choices=("inspect", "run"))
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, help="New/empty output directory (default: inspection or output)")
    parser.add_argument("--config", type=Path, help="JSON overrides for configurable thresholds")
    parser.add_argument("--interval", type=float, help="Override FRAME_EXTRACTION_INTERVAL in seconds")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cv2.setNumThreads(2)
    try:
        config = Config.load(args.config)
        if args.interval is not None:
            config.FRAME_EXTRACTION_INTERVAL = args.interval
            config.validate()
        output = args.output or Path("inspection" if args.command == "inspect" else "output")
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError(f"Refusing to mix results with existing files in {output}; choose a new --output directory")
        video = probe_video(args.video)
        output.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(output / "pipeline.log")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(handler)
        try:
            LOG.info("Video duration: %.3f seconds", video.duration_seconds)
            LOG.info("Video FPS: %.6f | Resolution: %d x %d | Total frame count: %d", video.fps, video.width, video.height, video.total_frame_count)
            if args.command == "inspect":
                inspect(video, output, config)
                return 0
            return run(video, output, config)
        finally:
            logging.getLogger().removeHandler(handler)
            handler.close()
    except (ValueError, OSError, KeyError, TypeError, ZeroDivisionError, subprocess.CalledProcessError, cv2.error) as exc:
        LOG.error("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
