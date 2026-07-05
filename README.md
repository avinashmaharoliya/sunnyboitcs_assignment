# Sunnybotics — Solar Panel Inspection

Short: a reproducible pipeline that ingests images (or simulates captures), links each image to a panel on a solar-farm grid, quality-checks and classifies the image, assigns a cleaning priority score, annotates the image, and exports results for review and visualization.

This repository contains a complete prototype with the following components:

- Image ingestion and robust OpenCV-based I/O (`sunnybotics/image_io.py`)
- Per-image simulated robot metadata and GPS (`sunnybotics/geometry.py`)
- OpenCV quality gate for blur/glare/shadow (`sunnybotics/quality.py`)
- YOLOv8-backed classifier with safe fallback (`sunnybotics/classifier.py`)
- Priority scoring for operations (`sunnybotics/scoring.py`)
- Annotated review images (`sunnybotics/annotate.py`)
- End-to-end orchestration and exports (`main.py`, `sunnybotics/pipeline.py`)
- Lightweight Streamlit dashboard (`dashboard.py`)

Contents
--------

- `main.py` — run the full pipeline
- `prepare_dataset.py` — create the 3-class YOLO dataset in `prepared/solar_cls`
- `train_yolo.py` — helper to run Ultralytics YOLO training (optional)
- `dashboard.py` — Streamlit app that reads `outputs/results.csv`
- `models/` — place your YOLO weights here (not committed by default)
- `outputs/` — runtime outputs (annotated images, CSV/JSON exports)
- `prepared/solar_cls/` — prepared train/val folders (large; ignored by git)
- `sunnybotics/` — package with pipeline modules

Quick start (single-machine, Windows)
-------------------------------------

Requirements

- Python 3.9 (this project used a 3.9 venv named `venv39` during development)
- System with OpenCV (binary wheel), pandas, numpy, Pillow, and Streamlit

Create & activate venv (example using system Python 3.9):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell
# or: .\.venv\Scripts\activate.bat (cmd)
```

Install runtime dependencies:

```powershell
python -m pip install -r requirements.txt
```

Prepare dataset (if you have original raw folders in `dataset/`):

```powershell
python prepare_dataset.py --dataset-dir dataset --output-dir prepared/solar_cls
```

Run full pipeline

```powershell
# clears and writes outputs/ annotated images and CSV/JSON exports
Remove-Item -Recurse -Force .\outputs\*  # PowerShell; optional, clears previous outputs
python main.py --dataset-dir prepared/solar_cls --output-dir outputs
```

Run only a single image (two options):

1. Process the first image discovered:

```powershell
python main.py --dataset-dir prepared/solar_cls --output-dir outputs --limit 1
```

2. Or process a specific file: copy it into a temporary folder and run on that folder:

```powershell
mkdir prepared\single_image
copy "C:\path\to\your.jpg" prepared\single_image\
python main.py --dataset-dir prepared\single_image --output-dir outputs
```

Use real YOLO weights (optional)
--------------------------------

Place your Ultralytics YOLOv8 weights under `models/` (example name used in this repo: `models/solar_yolov8n_best.pt`). To require the model and fail if it's missing:

```powershell
python main.py --dataset-dir prepared/solar_cls --output-dir outputs --require-model
```

What the pipeline produces
--------------------------

- `outputs/results.csv` — row per image with metadata, panel assignment, condition, confidence, priority, issues
- `outputs/results.json` — same data in JSON
- `outputs/panel_registry.json` — farm registry used for matching
- `outputs/run_metadata.json` — summary and configuration used for the run
- `outputs/annotated/*.jpg` — annotated review images

How GPS matching works (short)
------------------------------

- The repo models a `rows x columns` farm grid (`sunnybotics/config.py`) with a center lat/lon and panel spacing.
- Each image receives simulated latitude/longitude by adding Gaussian noise to the expected panel coordinates (`sunnybotics/geometry.py`).
- `pipeline.py` computes `nearest_panel` (Haversine) and `expected_panel_for_timestamp`. If the nearest panel is far (> `gps_snap_max_distance_m`) the GPS is `uncertain_gps`; if the nearest differs from the time-sequence panel the record is `sequence_mismatch`.

Quality gating and classification
---------------------------------

- `sunnybotics/quality.py` computes Laplacian variance, mean brightness, saturated and dark pixel ratios to detect blur, glare, and shadow. Images failing these checks are labelled and skipped for classification.
- `sunnybotics/classifier.py` wraps Ultralytics YOLOv8 if available; otherwise it falls back to a `folder_label_fallback` that maps source folder names into `clean|dirty|damaged`.

Annotated images
----------------

Each annotated image shows a colored border and a small overlay with:

- CONDITION (UPPERCASE) and confidence
- cleaning priority score
- panel id and GPS status

This is saved to `outputs/annotated/<image_id>.jpg` and the path is included in the CSV/JSON exports.

Developer notes
---------------

- `prepare_dataset.py` converts a 5-folder raw dataset into a 3-class YOLO `train/` and `val/` split and performs a basic readability filter.

Limitations & next steps
------------------------

- GPS is simulated for demo purposes; in production ingest GPS from EXIF or robot telemetry.
- The current classifier falls back to folder labels when YOLO weights are missing — include trained weights for production evaluation.
- Consider adding `results.geojson` and a machine-readable manifest for easier downstream integrations.

Troubleshooting
---------------

- If `main.py` fails to load YOLO, check `models/` and `ultralytics` installation.
- If images are skipped as unreadable, confirm your OpenCV wheel supports your image formats (HEIC files may need conversion).
- Streamlit dashboard: run `streamlit run dashboard.py` and open `http://localhost:8501`.

Contact
-------

For questions about the internship challenge referenced in `Sunnybotics - Internship.pdf`, follow the contact instructions in that document.
