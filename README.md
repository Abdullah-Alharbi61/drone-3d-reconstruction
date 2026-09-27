# Drone video to 3D reconstruction

Python pipeline for overlapping frame extraction, conservative quality filtering,
and COLMAP sparse/dense reconstruction with a point-cloud viewer (Stages 2–4).
The source video, generated images/models, and local Python environment are not
included in this repository. Supply your own video to reproduce the pipeline;
the results below document the original local run.

Stage 4 is now implemented separately: see [RECONSTRUCTION.md](RECONSTRUCTION.md)
for COLMAP reconstruction, configuration, outputs, and the viewer. The extraction
and filtering commands below continue to run only Stages 2 and 3.

Extract overlapping full-resolution images, then conservatively reject unusable
images. These commands perform no reconstruction, measurement, building detection,
or segmentation.

## Run

Python 3.10+ and FFmpeg (`ffprobe` on PATH) are required. Install Python dependencies
in your environment if needed:

```bash
python3 -m pip install -r requirements.txt
python3 -m drone_frames inspect 'Drone Cam 1.mp4' --config config.drone.json
python3 -m drone_frames run 'Drone Cam 1.mp4' --config config.drone.json
python3 -m unittest discover -s tests -p 'test_pipeline.py' -v
```

`inspect` writes full-resolution samples, `inspection/contact_sheet.jpg`, and
`inspection/inspection.json` with source metrics, adjacent-frame tracking, and a
deliberately blurred comparison. Inspect these before tuning another video.

Use `--interval 0.25` to sample more densely or `--output output_v2` for a new run.
Output directories must be empty; existing results are never silently overwritten.
A partially failed run remains available for diagnosis; rerun into a new directory.

## This video's initial settings

`Drone Cam 1.mp4`: **124.307517 seconds, 59.940060 FPS, 1280 × 720,
7,451 frames**. The initial 24 evenly spaced samples show sharp, well-exposed
street footage passing several buildings, with a turn around the middle of the
video. This is not a complete orbit around one isolated building. Preserve visible
architecture throughout; target-specific coverage cannot be inferred from quality scores.

At 640-pixel analysis width, those source samples had Laplacian variance
**1,211–2,593**, mean grayscale brightness **111–127**, and edge fractions
**0.097–0.172**. Almost-black pixels occupied about **0.8–2.3%**, almost-white
pixels **0–1.5%** at full resolution. These observations motivated:

| Setting | Initial value | Rationale |
| --- | ---: | --- |
| `FRAME_EXTRACTION_INTERVAL` | 0.4 s | About 24 original frames apart; 312 images including endpoints, retaining the continuous trajectory. |
| `BLUR_THRESHOLD` | 100 | Far below observed sharp samples; a source sample deliberately blurred with sigma 1 at analysis size scored about 74. This is a conservative initial screen, not learned ground truth. |
| `DARKNESS_THRESHOLD` / `OVEREXPOSURE_THRESHOLD` | 15 / 245 | Reject only severe exposure, jointly with clipping and little surviving detail. |
| `CLIPPED_FRACTION_THRESHOLD` | 0.85 | Requires nearly whole-frame loss; ordinary shadows/sky are retained. |
| `DUPLICATE_THRESHOLD` | 0.995 SSIM | Very strict appearance similarity, combined with pixel and motion gates. |
| `DUPLICATE_MAX_MOTION_PIXELS` | 0.25 | Subpixel 90th-percentile displacement at analysis size; useful parallax is retained. |
| `MAX_DUPLICATE_GAP_SECONDS` | 1.2 s | Keep periodic anchors even during a stationary view. |

Every decision threshold is configurable in `config.drone.json`. JSON files may
also specify only changed settings. Metrics are measured on grayscale images
downscaled to at most `ANALYSIS_WIDTH`; changing that width requires recalibration.
JPEG quality is 98 by default; saved images retain source dimensions and color.

The inspection command also checked 23 adjacent pairs at 0.4-second spacing:
track retention was **55.4–99.8% (median 78.4%)**, median feature displacement per
pair was **0.50–18.63 pixels**, and SSIM was **0.302–0.900**. These samples support
dense overlap while showing real viewpoint changes, well outside the strict
duplicate gate. No rejection quota is applied; a clean video may keep every sample.

The completed run extracted **312**, accepted **312**, and rejected **0** frames.
Across all extracted images, sharpness was **1,177–2,928**, brightness **109–129**,
and the highest adjacent SSIM was **0.943**, below the duplicate gate. The largest
accepted-frame gap was **0.4004 seconds**; no overlap warnings fired. All seven
contact sheets and the lowest-sharpness image were visually reviewed. Original
resolution, first/last source-frame coverage, and byte-identical accepted copies
were verified. All **14 automated tests passed**, including actual variable-rate
timestamps and synthetic bad-frame rejection. `rejected_frames/` is intentionally
empty for this video under the supplied conservative settings.

## Output

```text
output/
├── extracted_frames/       # all sampled originals, at original resolution
├── accepted_frames/        # independent copies of accepted images
├── rejected_frames/        # independent copies for manual review
├── frame_report.json       # metadata, exact config, summary, every frame's decision
├── extraction_manifest.json
├── pipeline.log
├── summary.txt
└── review/                 # contact sheets of every extracted frame
```

`frame_report.json` is an object with `video`, `config`, `summary`, and a `frames`
list. Each frame includes `frame`, `timestamp_seconds`, `original_frame_number`,
`status`, `blur_score`, `brightness_score`, `duplicate_score`, and
`rejection_reason`, plus diagnostic measurements and all applicable reasons.
Source frame numbers are **zero-based**; filenames start at `frame_000001.jpg`.
Timestamps are presentation timestamps relative to the first video frame, not
rounded nominal-FPS estimates. Missing/unmeasurable scores are JSON `null`.
`duplicate_score` is local SSIM against `duplicate_reference` (the last accepted
frame); 1 means identical. The first frame has no reference.

## Filtering and overlap

- Extraction samples a time grid over the entire video and always includes its
  first and last frames. Sequential OpenCV decoding is checked against ffprobe's
  frame count; timestamp or decoding inconsistencies fail explicitly.
- Blur uses Laplacian variance. The generic `blur` reason covers both defocus and
  motion blur because this metric alone cannot distinguish them reliably.
- Exposure rejection requires extreme mean brightness, extensive clipped pixels,
  and few edges. Featureless frames require **all three** of low edge density,
  low corner count, and low entropy. This can catch blank sky, lens obstruction,
  or scenes with almost no structural information.
- Duplicates require high local SSIM, low mean pixel difference, almost no changed
  pixels, and negligible feature displacement with good track retention. Exact
  pixel matches also qualify. Comparison is local in time and against the last
  **accepted** frame, so cumulative slow movement is preserved. No geometric
  alignment removes parallax before comparison. Quality-passing timeline
  endpoints and periodic stationary anchors survive duplicate filtering.
- Sparse optical flow uses forward/backward consistency checks. Large motion
  rejects an image only with accompanying low sharpness; sharp new viewpoints
  receive an overlap warning. Low track retention and long accepted-frame gaps
  are reported for review, not used as automatic rejection rules.
- Corrupted extracted files are copied unchanged into `rejected_frames` and
  reported. Undecodable video frames cannot produce genuine JPEGs; those stop
  extraction with an error instead of fabricating frames or a success report.

Structural checks do **not** prove that a particular building is visible. A
textured unrelated scene can pass. Review the contact sheets; if a building is
out of view or an image has a visual defect the metrics missed, add inclusive
`EXCLUDE_TIME_RANGES`, for example `[[12.0, 14.4]]`, and rerun into a fresh output.
Such frames receive `manual_exclusion`. Identifying a building automatically
would require functionality outside these two stages.

Optical-flow track retention is a diagnostic, not a physical overlap percentage
or reconstruction guarantee. Moving cars and people can affect it. Review flags
may call for a shorter extraction interval; no 3D matching/reconstruction is run.
Rejection totals use the primary reason; `all_rejection_reason_counts` can count
one rejected frame under multiple failed checks.

The implementation uses OpenCV's documented
[feature tracking APIs](https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html).
