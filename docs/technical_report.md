# Technical Report

## System Overview

The pipeline follows the requested RF-01 through RF-07 flow. Images are loaded from the prepared dataset folders, assigned UUIDs, paired with simulated robot metadata, checked by an OpenCV quality gate, classified by YOLOv8 when weights are available, scored for cleaning priority, annotated, snapped to the nearest registered panel, and exported to JSON/CSV for the Streamlit dashboard.

## RF-01 Image Ingestion

The image loader walks `prepared/solar_cls` recursively in a deterministic order and accepts common image extensions. Each record receives a UUID. OpenCV decoding is used instead of assuming every file is valid. If an image cannot be read, the pipeline logs it as `uncertain` with `unreadable_image` in `detected_issues` and continues processing the rest of the mission.

## RF-02 Metadata Simulation

The simulated farm is a 5 row by 10 column grid with panels spaced 2 meters apart and centered near El Cerrito, California. Each panel receives a stable ID like `SB-R01-C01`. The robot visits one panel every 30 seconds in row-major order. GPS readings are simulated from the panel coordinate with Gaussian noise in north/east meters using a 2 meter standard deviation. The assumptions are stored in `outputs/run_metadata.json`.

## RF-03 Condition Analysis

The pipeline uses two stages:

1. OpenCV quality gate: Laplacian variance flags blur, mean brightness and saturated pixel ratio flag glare, and low brightness/dark pixel ratio flag shadowed images. These images do not go to the classifier because the visual evidence is unreliable.
2. YOLOv8 classifier: Valid images are classified into `clean`, `dirty`, or `damaged`. The trained weights are stored at `models/solar_yolov8n_best.pt`, and the training script uses the prepared three-class dataset.

The raw five classes are merged operationally: Dusty and Bird-drop become `dirty`; Electrical-damage and Physical-Damage become `damaged`.

## RF-04 Priority Score

The score is `base_score * confidence`, rounded to 0-100. Base scores are:

- damaged: 90
- dirty: 55
- glare: 40
- uncertain: 30
- shadowed: 25
- clean: 8

This makes high-confidence damage urgent, lower-confidence soiling moderate, and clean panels low priority.

## RF-05 Annotated Images

Every readable image is saved to `outputs/annotated/` with a colored border and text overlay. Green means clean, orange means dirty, red means damaged, and gray means glare, shadowed, or uncertain. The overlay includes condition, confidence, priority score, panel ID, and GPS status.

## RF-06 Spatial Association

The pipeline writes a panel registry to `outputs/panel_registry.json`. Each incoming GPS point is snapped to the nearest panel using Haversine distance. If the nearest panel is more than 2.5 meters away, the record is marked `uncertain_gps` and the sequence-derived panel is used for review. If the GPS snap and expected sequence panel disagree, the record is marked `sequence_mismatch`. Otherwise it is marked `matched`.

## RF-07 Export And Visualization

Results are exported to both JSON and CSV with the required fields: image ID, timestamp, latitude, longitude, robot ID, mission ID, panel row, panel ID, condition, confidence, cleaning priority score, annotated image path, and detected issues. The Streamlit dashboard loads `outputs/results.csv`, provides filters for condition, priority, and GPS status, shows a sortable table, and renders a Folium map with color-coded panel markers.
