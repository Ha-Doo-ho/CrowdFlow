# CrowdFlow experiment config guide

## 6m x 4m field test

Use this when the measured field is width 4m and height 6m in the clicked point order.

1. Save a calibration frame from the same drone position used for analysis.
2. Create calibration:

```powershell
python calibrate_from_image.py data/calibration_frames/<frame>.jpg --area-width 4 --area-height 6 --out config/calibration_6x4.json --preview output/calibration_preview_6x4.jpg
```

3. Run analysis:

```powershell
python main.py --config config/app_config_6x4.json
```

## 10m x 10m field test

Use this only after placing actual 10m x 10m reference points and creating a new calibration file.

```powershell
python calibrate_from_image.py data/calibration_frames/<frame>.jpg --area-width 10 --area-height 10 --out config/calibration_10x10.json --preview output/calibration_preview_10x10.jpg
python main.py --config config/app_config_10x10.json
```

## 7.85m x 10m field test

Use this when the measured field width is exactly 7.85m and height is 10m.
The grid size can stay at 1m; the last column is treated as a 0.85m-wide partial cell.

```powershell
python calibrate_from_image.py data/calibration_frames/<frame>.jpg --area-width 7.85 --area-height 10 --out config/calibration_7_85x10.json --preview output/calibration_preview_7_85x10.jpg
python main.py --config config/app_config_7_85x10.json --source data/tello_recordings/<recording>.mp4
```

## Point order

Click the four reference points in this exact order:

1. left_top
2. right_top
3. left_bottom
4. right_bottom

## Important

- Do not reuse indoor calibration for field tests.
- If the drone position, height, or camera angle changes, create calibration again.
- If the field is width 6m and height 4m instead of width 4m and height 6m, swap the values in both `app_config_6x4.json` and the calibration command.
