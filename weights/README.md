# Model weights

The model files are intentionally excluded from Git because they are large.
Place the files below directly in this directory before running CrowdFlow.

| Role | File | Size | SHA-256 |
|---|---|---:|---|
| Primary | `yolo11l_crowdflow.pt` | 152,704,991 bytes | `B0CE802698DB88BA0FEEAB01D5E154FCEE5DC1148D0B4D7589D6BAE7D394A270` |
| Comparison | `rtdetr_l.pt` | 66,492,534 bytes | `71C954ED768E088D725CADF9ED4B3D6D7AFD28D3C56A7FE464F51EF6789516EF` |

These files must be transferred through the team's shared storage. Do not
rename a different checkpoint to one of the names above. Run
`python verify_baseline.py` after transferring the files.
