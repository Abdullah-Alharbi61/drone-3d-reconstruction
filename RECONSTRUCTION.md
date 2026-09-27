# Stage 4 — COLMAP reconstruction

This stage consumes only `output/accepted_frames/` and the accepted entries in
`output/frame_report.json`. It estimates relative camera poses and scene geometry.
It does not segment a building, apply real-world scale, or calculate measurements.

## Validated result for this video

The completed COLMAP 4.2.0 run produced:

| Result | Value |
| --- | ---: |
| Accepted input images | 312 |
| Registered images | 312 (100%) |
| Sparse connected models | 1 |
| Sparse points | 106,500 |
| Mean observations per sparse point | 7.045 |
| Mean reprojection error | 0.546 pixels |
| Checked geometric depth maps | 312 |
| Dense points | 5,382,370 |
| Mesh vertices / faces | 88,551 / 166,739 |

The actual dense and mesh PLY files were opened and checked for finite coordinates;
mesh face indices were validated. Camera centers and rotations were checked against
their exported transforms. All geometric depth maps were read with COLMAP and had
finite values. All **21 automated tests passed**. The dense cloud contains about
50.5 times as many points as the sparse model. Preview images are saved as
`sparse_preview.png`, `dense_preview.png`, `mesh_preview.png`, and `depth_preview.png`
under `output/reconstruction/`. The primary artifact is
`output/reconstruction/dense/building_dense.ply` (145,324,225 bytes).

The run resumed after interruption, increased the fusion cache to resolve repeated
reloads, and used the validated mesh-trimming fallback described below. The report
preserves prior failures and parameter changes. `runtime_seconds` includes recorded
previous invocations and the final invocation; uncheckpointed work from an abruptly
interrupted process can be excluded, as stated in `runtime_scope`.

## Run and resume

The project-local environment contains the official CUDA-enabled COLMAP 4.2.0
Python bindings. There was no existing COLMAP installation on this machine.

```bash
# Installation on Linux with an NVIDIA GPU, if not already installed:
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-reconstruction.txt

# Run, or inspect metadata without reconstructing:
.venv/bin/python -m reconstruction --config config.reconstruction.json
.venv/bin/python -m reconstruction --inspect-only --output output/reconstruction_inspection

# Resume completed steps after interruption/failure:
.venv/bin/python -m reconstruction --resume

# Tests, including the earlier extraction/filtering tests:
.venv/bin/python -m unittest discover -s tests -v
```

Existing nonempty output directories require `--resume`. Resume checks the exact
geometry-affecting configuration and image-content hashes; it cannot silently reuse
matches from changed images. Cache budgets (`DENSE_CACHE_GB`, `FUSION_CACHE_GB`)
may change on resume, with the previous configuration retained in the report.
Choose a new `--output` directory after changing other settings.
COLMAP depth computation can reuse already completed depth maps within an
interrupted depth stage. Native engine output is recorded in `logs/`; the outer
process records failure even when a native worker exits unexpectedly. Failed
steps are retried on resume; completed steps are retained.

For a fresh environment without the existing Stage 2/3 OpenCV installation,
install `opencv-python-headless<4.12` alongside the pinned reconstruction
requirements. The NumPy pin also keeps the system OpenCV 4.6 build compatible.
For CPU-only systems install `pycolmap==4.2.0` instead of `pycolmap-cuda12`.
Feature extraction, matching, and sparse reconstruction support CPU. COLMAP
PatchMatch dense stereo requires CUDA in this setup: requesting dense output
without it preserves the sparse output and reports failure clearly. To request
only sparse output explicitly, set both `RUN_DENSE_RECONSTRUCTION` and
`GENERATE_MESH` to false; status will be `sparse_only`, not dense success.

## Inputs and metadata

The initial input is 312 chronologically ordered, full-resolution 1280×720 JPEGs.
The images have no EXIF or calibrated camera parameters. The source video identifies
a DJI Mini 4 Pro and contains a proprietary `djmd` timed metadata stream.

`input_summary.json` contains every input filename, original frame number,
timestamp, resolution, hash, and available EXIF, including GPS tags when present.
`metadata/` preserves source container metadata plus the raw DJI packet payloads
and packet timestamps/boundaries. GPS/RTK availability **inside** the proprietary
stream is unknown; the stream has not been interpreted. The source video is read
only for metadata. Its rejected or unsampled images never enter reconstruction.

With `CAMERA_MODEL=AUTO`, the current implementation selects `SIMPLE_RADIAL` and
uses one shared camera for this fixed-resolution, single-camera clip. COLMAP uses
EXIF focal initialization if available; otherwise it initializes focal length
from image dimensions and refines intrinsics during SfM. This is self-calibration,
not known calibration. Supply an explicit `CAMERA_MODEL` and `CAMERA_PARAMS` in
COLMAP parameter order when calibrated intrinsics exist. Set `REFINE_INTRINSICS`
false to hold those parameters fixed. Principal point refinement is disabled.

## Configuration and quality gates

All principal settings are in `config.reconstruction.json`; partial JSON overrides
are also accepted. `MAX_IMAGE_SIZE=1600` and `DENSE_MAX_IMAGE_SIZE=1280` preserve
the input image resolution. Two worker threads and configurable dense/fusion caches
limit RAM consumption. The run initially used 0.25 GB for both caches while RAM
was scarce; fusion was raised to 2 GB after more memory became available because
the smaller budget caused repeated data reloads. Feature extraction targets 4,096
features; SIFT may add orientations, and the matcher caps its input at 4,096.

- `MATCHING_STRATEGY=sequential`, `MATCH_OVERLAP=10`: match chronological neighbors
  with geometric verification and guided matching. `exhaustive` is also supported.
- `LOOP_DETECTION=true`: additionally match all pairs of periodic keyframes
  (`LOOP_KEYFRAME_STRIDE=15`), excluding already local pairs. This is a wider
  keyframe matching pass, **not** vocabulary-tree retrieval. It can connect
  revisited views without matching every full-resolution image pair. Lower the
  stride if brief revisits are missed. It cannot reconstruct unseen building sides.
- Sparse validation requires at least 80% of images in the **largest connected
  model**, at least 1,000 points, mean track length at least 3 observations, and
  mean reprojection error at most 2 pixels. Gates are configurable and stop dense
  processing on failure. Other components remain in `sparse_models/` with their
  own coordinate systems; they are never falsely merged or summed for registration.
- Dense stereo computes photometric and geometrically consistent depth/normal
  maps, then fuses them with at least three supporting pixels. Dense output must
  have finite, nondegenerate coordinates, at least 10,000 points, and at least
  1.5 times as many points as the sparse model. Point count alone does not prove
  complete surfaces; visually inspect the result.
- `GENERATE_MESH=true` enables COLMAP Poisson meshing of the fused normals, with
  configurable depth (9 by default). Mesh validation requires nonempty faces.
  On this COLMAP wheel, native trimming produced invalid numeric vertex data.
  The validated fallback reruns COLMAP Poisson without native trimming and removes
  faces whose vertices or centroid are unsupported by the dense cloud. Its
  configurable distance bound is `MESH_MAX_DISTANCE_CELLS` (default 2) times the
  largest dense-cloud extent divided by `2**MESH_DEPTH`. Coordinates are never
  replaced or repaired heuristically. The original invalid output, untrimmed
  mesh, and diagnostic options remain in `logs/` for inspection. This is surface
  support filtering, not building segmentation or real-world scaling.

Full COLMAP options used for each completed worker are saved in
`logs/*_options.json`; these include library defaults in addition to the user
configuration. GPU driver availability and COLMAP CUDA support are recorded in
`hardware.json`. GPU feature/matching failures retry on CPU and are reported.

## Outputs and inspection

```text
output/reconstruction/
├── database.db
├── input_summary.json
├── metadata/
├── sparse_models/             # all disconnected COLMAP models, if any
├── sparse/                    # selected largest model
│   ├── cameras.bin / images.bin / points3D.bin
│   ├── rigs.bin / frames.bin  # COLMAP 4 camera/rig bookkeeping
│   ├── *.txt                 # readable model, including feature tracks
│   ├── sparse.ply
│   ├── camera_locations.ply
│   └── camera_poses.json
├── dense/
│   ├── images/                # undistorted images
│   ├── sparse/                # matching undistorted cameras
│   ├── stereo/depth_maps/
│   ├── stereo/normal_maps/
│   └── building_dense.ply
├── mesh/building_mesh.ply
├── logs/
├── reconstruction_report.json
└── summary.txt
```

Camera poses include camera centers, world-to-camera rotation and translation,
quaternions in XYZW order, intrinsic matrices, distortion parameters, timestamps,
and camera IDs. COLMAP uses `X_camera = R X_world + t`; the camera center is
`-R.T @ t`, **not** the translation vector. Camera axes are right, down, forward.
All world coordinates remain in arbitrary units; no GPS alignment is performed.

```bash
# Lightweight interactive point viewer (mesh files show vertices):
.venv/bin/python view_reconstruction.py output/reconstruction/dense/building_dense.ply

# Headless three-view preview with camera path:
.venv/bin/python view_reconstruction.py output/reconstruction/dense/building_dense.ply \
  --save output/reconstruction/dense_preview.png

# Optional full mesh viewer, if Open3D is installed separately:
.venv/bin/python view_reconstruction.py output/reconstruction/mesh/building_mesh.ply --open3d
```

The viewer subsamples only the display, preserving the full PLY file. No complex
frontend is needed. PLY files can also be opened in MeshLab or CloudCompare.
The default display treats negative Y as up, matching the approximate initial
camera orientation in this footage. `--vertical-axis z --no-invert-vertical`
changes only the view; it never rotates or scales the saved reconstruction.
The default view zooms to the central 99% of displayed points plus all camera
centers, so distant outliers do not obscure the scene. Use
`--bounds-percentile 0` to inspect the full extent. No points are removed from disk.

The footage moves along streets past multiple buildings, rather than completing
an orbit around one isolated building. Geometry includes the visible surroundings;
the name `building_dense.ply` does not imply segmentation. Unseen walls/roofs,
moving vehicles, repetitive facades, and weakly textured surfaces can remain
incomplete or produce artifacts even with good sparse statistics. Poisson meshes
may bridge gaps, so use the point cloud as the primary evidence of observed geometry.

Implementation follows the official [PyCOLMAP APIs](https://colmap.github.io/pycolmap/index.html)
and [COLMAP reconstruction guidance](https://colmap.github.io/tutorial.html).
