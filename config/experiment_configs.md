# CrowdFlow experiment configuration guide

## Frozen final baseline: 7.85m x 10m

Use this setup for final reports, regression tests, dashboard captures, and
poster evidence.

```text
Config:      config/app_config.json
Alias:       config/app_config_7_85x10.json
Calibration: config/calibration_7_85x10.json
Model:       weights/yolo11l_crowdflow.pt
Area:        7.85m x 10.0m
Grid:        1.0m
Margin:      0.15m
```

Run the frozen default video:

```powershell
python verify_baseline.py --skip-hash
python main.py
```

Run a different representative video without editing JSON:

```powershell
python main.py --config config/app_config.json --source data/tello_recordings/<recording>.mp4
```

Representative recordings:

1. `tello_flight_recording_20260711_123722.mp4`: single-person position test
2. `tello_flight_recording_20260711_155226.mp4`: three-person density test
3. `tello_flight_recording_20260711_164845.mp4`: transition/regression test

The numbered calibration files are archived candidates. The final baseline
uses only `config/calibration_7_85x10.json`.

## Calibration point order

Click the four reference points in this exact order:

1. left_top
2. right_top
3. left_bottom
4. right_bottom

The calibration frame and analyzed recording must come from the same drone
position, height, and camera angle.

## Creating a candidate calibration

Do not overwrite the frozen calibration while experimenting.

```powershell
python calibrate_from_image.py data/calibration_frames/<frame>.jpg --area-width 7.85 --area-height 10 --out config/calibration_7_85x10_candidate.json --preview output/calibration_preview_7_85x10_candidate.jpg
```

After validation, update the manifest and its SHA-256 only when the team has
explicitly approved a new baseline.

## Historical configurations

### Indoor 1m x 3m

```powershell
python main.py --config config/app_config_indoor_1x3.json
```

This is retained for indoor reference only.

### Outdoor 4m x 6m

```powershell
python main.py --config config/app_config_6x4.json
```

This is a completed preliminary field test, not the final poster baseline.

### 10m x 10m candidate

```powershell
python main.py --config config/app_config_10x10.json
```

This was an initial target. It is not the final baseline because the Tello
camera angle and motion produced larger projection uncertainty.
