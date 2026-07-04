# Sunnybotics Solar Panel Inspection

This project implements the YOLOv8-centered inspection approach. The current dashboard reads the committed result files in `outputs/`, and the local prepared image dataset lives at `prepared/solar_cls`.

- ingest all dataset images with UUIDs and corrupt-file handling
- simulate robot metadata on a 5 x 10 solar-farm grid near El Cerrito, California
- quality-gate images with OpenCV for blur, glare, and shadow
- classify valid images as `clean`, `dirty`, or `damaged` with YOLOv8
- compute a 0-100 cleaning priority score
- save annotated images, JSON, CSV, panel registry, and a Streamlit map/table dashboard

The original raw dataset had five folders. The prepared dataset already merges them into three operational classes:

| Raw folder | YOLO class |
| --- | --- |
| Clean | clean |
| Dusty | dirty |
| Bird-drop | dirty |
| Electrical-damage | damaged |
| Physical-Damage | damaged |

## Commands

Run the current YOLO-backed pipeline:

```powershell
.\venv39\Scripts\python.exe main.py
.\venv39\Scripts\streamlit.exe run dashboard.py
```

The prepared image dataset is generated data and is not pushed to Git because it is large. Keep `prepared/solar_cls` locally if you want to rerun `main.py`; the dashboard can open from the committed `outputs/results.csv`.

Train YOLOv8 in the Python 3.9 training venv:

```powershell
.\venv39\Scripts\python.exe train_yolo.py --epochs 20
.\venv39\Scripts\python.exe main.py --model <path printed by train_yolo.py>
```

If PyTorch is not available for your current Python, create a Python 3.9-3.12 venv for training. This workspace uses `venv39` for YOLO, the pipeline, and the dashboard.

## Developer setup

Install runtime dependencies:

```powershell
python -m pip install -r requirements.txt
```

Install development dependencies and run tests:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```

## Output Files

`python main.py` writes:

- `outputs/results.csv`
- `outputs/results.json`
- `outputs/panel_registry.json`
- `outputs/run_metadata.json`
- `outputs/annotated/*.jpg`

Without trained YOLO weights, `main.py` uses a clearly marked `folder_label_fallback` so the rest of the system can be reviewed. Run with `--require-model` to enforce real YOLO weights.
