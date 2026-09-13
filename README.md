# Multi-Instance Object Recognition in Cluttered Scenes

This project finds several copies of one object in a busy photograph. It was written for **Computer Vision — Assignment 1** and uses a red notebook as the template object. The scene deliberately includes rotation, scale changes, occlusion, and distracting desk clutter.

Rather than treating the task as a one-shot image match, the program narrows the evidence in stages: local features suggest possible correspondences, a Hough space gathers mutually consistent votes, RANSAC tests each promising group geometrically, and NMS keeps the final view readable.

![Final localization result](outputs/final_detections.jpg)

> The green boxes are candidate detections produced by the current parameter set. This is a challenging scene: the result should be inspected visually, and the report discusses remaining false positives as well as successful localizations.

## What is implemented from scratch

OpenCV is used only where the assignment permits it: image I/O, colour conversion, SIFT feature extraction, and drawing output images. The matching and recognition logic is implemented in this repository:

1. **SIFT extraction** — obtains 128-dimensional keypoint descriptors for the template and the scene.
2. **Scene-to-template matching** — computes Euclidean descriptor distances with NumPy and applies Lowe's ratio test (`0.75`) to reject ambiguous matches.
3. **4D Generalized Hough voting** — each surviving match predicts an object centre, scale, and rotation. Votes are quantized into `(x, y, scale, angle)` bins.
4. **Affine RANSAC** — samples three correspondences, rejects collinear samples, estimates an affine transform with a least-squares solve, and retains low-reprojection-error inliers.
5. **Greedy multi-instance extraction** — after a valid instance is found, its inlier matches are removed and the remaining matches are processed again.
6. **Non-maximum suppression** — ranks detections by inlier count and removes overlapping boxes using IoU.

No OpenCV matcher, OpenCV affine-estimation routine, homography routine, or external clustering package is used for the recognition logic.

## Quick start

### Prerequisites

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/) (recommended), or a Python environment with the dependencies in `pyproject.toml`

### Install and run

```bash
uv sync
uv run python main.py
```

If the project environment has already been created, this also works:

```bash
.venv/bin/python main.py
```

The script reads:

- `data/template.jpg` — the isolated notebook template
- `data/scene.jpg` — the cluttered scene containing multiple notebook instances

It writes these visualizations to `outputs/`:

| File | What it shows |
| --- | --- |
| `naive_matches.jpg` | Ratio-test matches before geometric verification; useful for seeing why matching alone is not enough. |
| `verified_matches.jpg` | Matches retained as RANSAC inliers during greedy extraction. |
| `final_detections.jpg` | The scene with the final NMS-filtered candidate boxes. |

Generated output is intentionally ignored by Git, so running the pipeline will not clutter commits.

## Repository guide

```text
.
├── data/                   # Template, cluttered scene, and captured photos
├── docs/
│   └── report.tex          # Assignment report source (LaTeX)
├── outputs/                # Generated visualizations
├── src/
│   ├── sift.py             # SIFT wrapper and descriptor validation
│   ├── matcher.py          # Descriptor distance + Lowe ratio test
│   ├── hough.py            # 4D Generalized Hough votes and bins
│   ├── affine.py           # Affine model and least-squares estimator
│   ├── ransac.py           # Robust affine verification
│   ├── detection.py        # Template-corner projection and boxes
│   ├── nms.py              # IoU non-maximum suppression
│   └── visualization.py    # Match and detection images
└── main.py                 # Pipeline configuration and orchestration
```

## Tunable settings

The main parameters live near the top of `main.py`:

| Setting | Value | Purpose |
| --- | ---: | --- |
| Maximum SIFT features | 6000 | Retains more scene detail for challenging multi-instance matching. |
| Ratio threshold | 0.75 | Lower values are stricter about descriptor ambiguity. |
| Maximum processing dimension | 2560 px | Preserves useful detail while bounding memory use. |\n| Position / scale / angle bins | 20 px / 0.1 / 15° | Controls how similar Hough votes must be to form a cluster. |
| Minimum Hough votes | 3 | Minimum evidence before a cluster is verified. |
| RANSAC iterations | 500 | Number of affine hypotheses sampled per cluster. |
| Reprojection threshold | 5 px | Maximum error for an affine inlier. |
| NMS IoU threshold | 0.5 | Overlap at which lower-scoring boxes are suppressed. |

For a new object or scene, start by checking `naive_matches.jpg`. If the object receives very few matches, use a sharper, more textured template or adjust the ratio threshold. If background objects are boxed, increasing the minimum votes or tightening geometric thresholds is a reasonable next experiment.

## Report

The accompanying write-up is available at [`docs/report.tex`](docs/report.tex). From the `docs/` directory, compile it with a LaTeX engine that provides `graphicx` and `amsmath` (for example, `pdflatex report.tex`). It references the project images directly, so keep the repository layout intact.

## Notes on reproducibility

RANSAC uses a fixed random seed (`42`), making its sampling sequence repeatable for the same images and dependencies. Small changes in OpenCV/SIFT versions can still alter detected keypoints, so treat image outputs as reproducible within a fixed environment rather than as bit-for-bit guarantees across every platform.
