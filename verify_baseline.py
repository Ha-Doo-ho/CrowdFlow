import argparse
import hashlib
import json
from pathlib import Path

from core.settings import AppSettings


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "config" / "baseline_manifest.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify the frozen CrowdFlow baseline files."
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to the baseline manifest JSON.",
    )
    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help="Check paths and settings without hashing large model/video files.",
    )
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_json(path):
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def verify_settings(manifest, errors):
    primary_path = ROOT / manifest["primary_config"]
    alias_path = ROOT / manifest["config_alias"]
    settings = AppSettings.load(primary_path)
    area = manifest["measured_area_m"]

    if primary_path.read_bytes() != alias_path.read_bytes():
        errors.append("Primary config and 7.85x10 alias differ.")
    if settings.area_width != float(area["width"]):
        errors.append("Config area_width differs from the manifest.")
    if settings.area_height != float(area["height"]):
        errors.append("Config area_height differs from the manifest.")
    if settings.grid_size != 1.0:
        errors.append("Frozen baseline grid_size must be 1.0m.")
    if settings.boundary_margin_m != 0.15:
        errors.append("Frozen baseline boundary_margin_m must be 0.15m.")
    if settings.calibration_path != manifest["primary_calibration"]:
        errors.append("Config calibration_path differs from the manifest.")
    if settings.source != manifest["default_video"]:
        errors.append("Config source differs from the default video.")
    if settings.model_path != manifest["primary_model"]:
        errors.append("Config model_path differs from the primary model.")


def verify_calibration(manifest, errors):
    calibration = load_json(ROOT / manifest["primary_calibration"])
    src_points = calibration.get("src_points", [])
    dst_points = calibration.get("dst_points", [])
    area = manifest["measured_area_m"]
    expected_dst = [
        [0.0, 0.0],
        [float(area["width"]), 0.0],
        [0.0, float(area["height"])],
        [float(area["width"]), float(area["height"])],
    ]

    if len(src_points) != 4:
        errors.append("Calibration must contain four src_points.")
    if dst_points != expected_dst:
        errors.append("Calibration dst_points differ from the measured area.")


def main():
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)
    errors = []

    for relative_path, expected_hash in manifest["sha256"].items():
        path = ROOT / relative_path
        if not path.is_file():
            errors.append(f"Missing file: {relative_path}")
            continue
        if not args.skip_hash:
            actual_hash = sha256(path)
            if actual_hash != expected_hash.upper():
                errors.append(f"SHA-256 mismatch: {relative_path}")

    verify_settings(manifest, errors)
    verify_calibration(manifest, errors)

    mode = "paths/settings" if args.skip_hash else "paths/settings/SHA-256"
    if errors:
        print(f"[FAIL] {manifest['baseline_id']} ({mode})")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print(f"[PASS] {manifest['baseline_id']} ({mode})")
    print(f"  config: {manifest['primary_config']}")
    print(f"  calibration: {manifest['primary_calibration']}")
    print(f"  default video: {manifest['default_video']}")
    print(f"  primary model: {manifest['primary_model']}")


if __name__ == "__main__":
    main()
