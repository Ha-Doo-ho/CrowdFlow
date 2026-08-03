# Representative recordings

MP4 files are intentionally excluded from Git. The frozen baseline uses these
recordings, which must be transferred through the team's shared storage.

| Recording | Purpose | SHA-256 |
|---|---|---|
| `tello_flight_recording_20260711_123722.mp4` | Single-person IN/OUT and boundary validation | `68AA3C9A4C40689FA47CBD22E2F575E09C9157E692F2C0583FB79C2C33F0E976` |
| `tello_flight_recording_20260711_155226.mp4` | Three-person distributed/dense placement | `C90746BC26526AB49E98658332E12B34FFFEBA1363554F421A6826CB4BE189E5` |
| `tello_flight_recording_20260711_164845.mp4` | Two-to-three-person transition and regression test | `86EBAA9E66DE79D380B2DDECECA6476024D552CF54711473C764F72B4EB51BBA` |

All three files are 960x720 at 30 FPS. Run `python verify_baseline.py` after
placing them in this directory.
