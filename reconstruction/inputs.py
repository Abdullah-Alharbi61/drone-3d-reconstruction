import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from PIL import Image, ExifTags

from drone_frames.io import save_json


def json_safe(value):
    if isinstance(value, bytes):
        return {"encoding": "hex", "data": value.hex()}
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def inspect_inputs(directory, frame_report):
    if directory.name != "accepted_frames":
        raise ValueError("Input must be the accepted_frames directory; rejected images are not enabled")
    report = json.loads(frame_report.read_text())
    records = [r for r in report["frames"] if r["status"] == "accepted"]
    records.sort(key=lambda r: (r["timestamp_seconds"], r["original_frame_number"]))
    names = [r["frame"] for r in records]
    if len(names) < 3 or len(names) != len(set(names)):
        raise ValueError("At least three unique accepted images are required")
    discovered = {p.name for p in directory.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    if discovered != set(names):
        raise ValueError("Accepted directory and frame report disagree; refusing stale or extra images")
    if names != sorted(names):
        raise ValueError("COLMAP sequential matching needs filenames in chronological order")
    images = []
    digest = hashlib.sha256()
    for record in records:
        if Path(record["frame"]).name != record["frame"]:
            raise ValueError("Frame names must not contain paths")
        path = directory / record["frame"]
        if path.is_symlink():
            raise ValueError(f"Input symlinks are not accepted: {path}")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update((record["frame"] + sha).encode())
        with Image.open(path) as im:
            im.load()
            exif = im.getexif()
            tags = {ExifTags.TAGS.get(k, str(k)): json_safe(v) for k, v in exif.items()}
            if 34853 in exif:
                tags["GPSInfo"] = json_safe(exif.get_ifd(34853))
            images.append({**record, "width": im.width, "height": im.height, "sha256": sha, "exif": tags})
    return {"input_images": len(images), "input_directory": str(directory.resolve()),
            "resolutions": sorted({f"{im['width']}x{im['height']}" for im in images}),
            "ordering": "frame report timestamps, verified against lexical filenames",
            "exif_images": sum(bool(im["exif"]) for im in images),
            "gps_exif_images": sum("GPSInfo" in im["exif"] for im in images),
            "calibrated_intrinsics_available": False,
            "input_fingerprint": digest.hexdigest(), "images": images}


def preserve_video_metadata(source, destination):
    destination.mkdir(exist_ok=True)
    if not source.is_file():
        return {"available": False, "warning": f"Source video unavailable: {source}"}
    if not shutil.which("ffprobe"):
        return {"available": False, "warning": "ffprobe unavailable; original video metadata remains in source"}
    command = ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(source.resolve())]
    metadata = json.loads(subprocess.check_output(command, text=True))
    save_json(destination / "source_video_metadata.json", metadata)
    streams = []
    for stream in metadata.get("streams", []):
        if stream.get("codec_type") != "data":
            continue
        index = stream["index"]
        packets = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", str(index), "-show_packets",
            "-show_entries", "packet=pts_time,dts_time,duration_time,size,pos,flags", "-of", "json", str(source.resolve())
        ], text=True))
        save_json(destination / f"stream_{index}_packets.json", packets)
        # Preserve payload bytes and per-packet boundaries without interpreting DJI telemetry.
        with source.open("rb") as original, (destination / f"stream_{index}.bin").open("wb") as target:
            for packet in packets["packets"]:
                original.seek(int(packet["pos"]))
                payload = original.read(int(packet["size"]))
                if len(payload) != int(packet["size"]):
                    raise ValueError("Truncated source metadata packet")
                target.write(payload)
        streams.append({"index": index, "codec": stream.get("codec_tag_string"), "packets": len(packets["packets"])})
    return {"available": True, "source_video": str(source.resolve()), "data_streams": streams,
            "camera_metadata": metadata.get("format", {}).get("tags", {}),
            "gps_status": "unknown in proprietary telemetry; raw data preserved, not decoded or used",
            "coordinates_used_for_scale": False}
