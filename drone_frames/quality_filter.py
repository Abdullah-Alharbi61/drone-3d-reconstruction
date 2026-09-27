"""Conservative quality checks; no detection, segmentation, or reconstruction."""
import logging
from pathlib import Path
import shutil

import cv2
import numpy as np

from .config import Config

LOG = logging.getLogger(__name__)


def analysis_gray(image: np.ndarray, config: Config) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.shape[1] > config.ANALYSIS_WIDTH:
        gray = cv2.resize(gray, (config.ANALYSIS_WIDTH, round(gray.shape[0] * config.ANALYSIS_WIDTH / gray.shape[1])),
                          interpolation=cv2.INTER_AREA)
    return gray


def corners(gray: np.ndarray, config: Config):
    return cv2.goodFeaturesToTrack(gray, config.FEATURE_MAX_CORNERS, config.FEATURE_QUALITY,
                                  config.FEATURE_MIN_DISTANCE)


def image_metrics(gray: np.ndarray, config: Config) -> dict:
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    probabilities = hist[hist > 0] / gray.size
    features = corners(gray, config)
    return {
        "blur_score": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness_score": float(gray.mean()),
        "dark_fraction": float((gray <= config.DARK_PIXEL_THRESHOLD).mean()),
        "bright_fraction": float((gray >= config.BRIGHT_PIXEL_THRESHOLD).mean()),
        "edge_fraction": float((cv2.Canny(gray, config.CANNY_LOW, config.CANNY_HIGH) > 0).mean()),
        "entropy": float(-(probabilities * np.log2(probabilities)).sum()),
        "corner_count": 0 if features is None else len(features),
    }


def structural_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Local SSIM, without image alignment: alignment could erase real parallax."""
    a, b = a.astype(np.float32), b.astype(np.float32)
    smooth = lambda im: cv2.GaussianBlur(im, (11, 11), 1.5)
    ma, mb = smooth(a), smooth(b)
    va = np.maximum(0, smooth(a * a) - ma * ma)
    vb = np.maximum(0, smooth(b * b) - mb * mb)
    covariance = smooth(a * b) - ma * mb
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    score = ((2 * ma * mb + c1) * (2 * covariance + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
    return float(np.clip(score.mean(), 0, 1))


def pair_metrics(previous: np.ndarray, current: np.ndarray, config: Config) -> dict:
    result = {"duplicate_score": None, "mean_pixel_difference": None, "changed_pixel_fraction": None,
              "motion_median_pixels": None, "motion_p90_pixels": None, "tracked_fraction": None,
              "tracked_corners": 0, "exact_duplicate": False}
    if previous.shape != current.shape:
        return result
    difference = cv2.absdiff(previous, current)
    result.update(duplicate_score=structural_similarity(previous, current),
                  mean_pixel_difference=float(difference.mean()),
                  changed_pixel_fraction=float((difference > config.DUPLICATE_PIXEL_DIFFERENCE).mean()),
                  exact_duplicate=bool(not difference.any()))
    points = corners(previous, config)
    if points is None:
        return result
    tracked, status, error = cv2.calcOpticalFlowPyrLK(previous, current, points, None, winSize=(21, 21), maxLevel=3)
    if tracked is None:
        return result
    back, back_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, tracked, None, winSize=(21, 21), maxLevel=3)
    if back is None:
        return result
    fb_error = np.linalg.norm(back - points, axis=2).ravel()
    xy = tracked.reshape(-1, 2)
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < current.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < current.shape[0])
    good = ((status.ravel() == 1) & (back_status.ravel() == 1) & inside &
            (error.ravel() <= config.FLOW_MAX_ERROR) & (fb_error <= config.FLOW_MAX_FORWARD_BACKWARD_ERROR))
    result.update(tracked_fraction=float(good.mean()), tracked_corners=int(good.sum()))
    if good.sum() >= config.FLOW_MIN_TRACKS:
        movement = np.linalg.norm(tracked - points, axis=2).ravel()[good]
        result.update(motion_median_pixels=float(np.median(movement)), motion_p90_pixels=float(np.percentile(movement, 90)))
    return result


def quality_reasons(metrics: dict, config: Config) -> list[str]:
    reasons = []
    # Require both extreme exposure and very little surviving edge information.
    if metrics["edge_fraction"] < config.EXPOSURE_MAX_EDGE_FRACTION:
        if metrics["brightness_score"] < config.DARKNESS_THRESHOLD and metrics["dark_fraction"] >= config.CLIPPED_FRACTION_THRESHOLD:
            reasons.append("underexposed")
        if metrics["brightness_score"] > config.OVEREXPOSURE_THRESHOLD and metrics["bright_fraction"] >= config.CLIPPED_FRACTION_THRESHOLD:
            reasons.append("overexposed")
    if (metrics["edge_fraction"] < config.MIN_EDGE_FRACTION and metrics["corner_count"] < config.MIN_CORNERS
            and metrics["entropy"] < config.MIN_ENTROPY):
        reasons.append("insufficient_structure")
    if metrics["blur_score"] < config.BLUR_THRESHOLD:
        reasons.append("blur")  # A single image cannot reliably distinguish defocus from motion blur.
    return reasons


def is_duplicate(metrics: dict, config: Config) -> bool:
    if metrics["exact_duplicate"]:
        return True
    return bool(metrics["duplicate_score"] is not None and metrics["duplicate_score"] >= config.DUPLICATE_THRESHOLD
                and metrics["mean_pixel_difference"] <= config.DUPLICATE_MAX_MEAN_DIFFERENCE
                and metrics["changed_pixel_fraction"] <= config.DUPLICATE_MAX_CHANGED_FRACTION
                and metrics["motion_p90_pixels"] is not None
                and metrics["motion_p90_pixels"] <= config.DUPLICATE_MAX_MOTION_PIXELS
                and metrics["tracked_fraction"] >= config.DUPLICATE_MIN_TRACK_FRACTION)


def filter_frames(records: list[dict], extracted: Path, accepted: Path, rejected: Path, config: Config) -> list[dict]:
    for directory in (accepted, rejected):
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise ValueError(f"Filter output directory must be empty: {directory}")
    results = []
    last_gray = None
    last_record = None
    for index, record in enumerate(records):
        source = extracted / record["frame"]
        if not source.is_file():
            raise ValueError(f"Extracted file missing: {source}")
        result = {**record, "status": "accepted", "blur_score": None, "brightness_score": None,
                  "duplicate_score": None, "duplicate_reference": None, "rejection_reason": None,
                  "rejection_reasons": [], "warnings": []}
        image = cv2.imread(str(source))
        gray = None
        reasons = []
        if image is None or not image.size:
            reasons = ["corrupted_image"]
        else:
            gray = analysis_gray(image, config)
            result.update(image_metrics(gray, config))
            reasons = quality_reasons(result, config)
            time = record["timestamp_seconds"]
            if any(start <= time <= end for start, end in config.EXCLUDE_TIME_RANGES):
                reasons.insert(0, "manual_exclusion")
            # Only compare to the last accepted image. This prevents chains of
            # tiny changes from erasing a whole slow-moving camera trajectory.
            if last_gray is not None:
                gap = time - last_record["timestamp_seconds"]
                pair = pair_metrics(last_gray, gray, config)
                result.update(pair)
                result["duplicate_reference"] = last_record["frame"]
                if (pair["tracked_fraction"] is not None and
                        pair["tracked_fraction"] < config.OVERLAP_WARNING_TRACK_FRACTION):
                    result["warnings"].append("low_track_retention_review_overlap")
                motion = pair["motion_median_pixels"]
                if motion is not None and motion / gray.shape[1] > config.EXTREME_MOTION_FRACTION:
                    result["warnings"].append("large_view_change_review_overlap")
                    # A sharp new viewpoint is valuable; large displacement alone
                    # never rejects an image. Only unreliable blurred movement does.
                    if gap <= config.COVERAGE_WARNING_GAP_SECONDS and result["blur_score"] < config.MOTION_BLUR_THRESHOLD:
                        reasons.insert(0, "extreme_motion")
                if (not reasons and index != len(records) - 1 and
                        gap <= config.DUPLICATE_WINDOW_SECONDS and gap < config.MAX_DUPLICATE_GAP_SECONDS - 1e-9
                        and is_duplicate(pair, config)):
                    reasons.append("near_duplicate")
        if reasons:
            result.update(status="rejected", rejection_reason=reasons[0], rejection_reasons=reasons)
            shutil.copy2(source, rejected / source.name)
        else:
            shutil.copy2(source, accepted / source.name)
            last_gray, last_record = gray, record
        results.append(result)
        if (index + 1) % 50 == 0:
            LOG.info("Quality checked %d / %d frames", index + 1, len(records))
    return results
