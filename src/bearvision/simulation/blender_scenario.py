"""Generate a video regression scenario from a Blender rider-motion export."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any

import yaml

from bearvision.contracts import ScenarioDefinition


STANDARD_GRAVITY_MPS2 = 9.80665


def simulated_rssi_dbm(
    distance_m: float,
    *,
    reference_rssi_dbm_at_1m: int = -50,
    path_loss_exponent: float = 2.0,
) -> int:
    """Calculate received RSSI using the log-distance path-loss model."""

    if distance_m <= 0:
        raise ValueError("distance_m must be positive")
    if path_loss_exponent <= 0:
        raise ValueError("path_loss_exponent must be positive")
    value = reference_rssi_dbm_at_1m - 10 * path_loss_exponent * math.log10(
        max(1.0, distance_m)
    )
    return min(20, max(-127, round(value)))


def _single_file(scene_dir: Path, pattern: str, description: str) -> Path:
    matches = sorted(scene_dir.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"expected one {description} matching {pattern!r} in {scene_dir}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _repository_path(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"scenario source must stay inside the repository: {path}") from exc


def _xyz(value: Any, description: str) -> tuple[float, float, float]:
    try:
        return float(value["x"]), float(value["y"]), float(value["z"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{description} must contain numeric x, y and z values") from exc


def generate_blender_scenario(
    scene_dir: str | Path,
    *,
    repository_root: str | Path | None = None,
    tag_id: str | None = None,
    rider_id: str | None = None,
    sample_rate_hz: float = 10.0,
    video_sample_fps: float = 10.0,
    reference_rssi_dbm_at_1m: int = -50,
    path_loss_exponent: float = 2.0,
    battery_voltage_mv: int = 3000,
    detector_confidence_threshold: float = 0.6,
) -> ScenarioDefinition:
    """Build a strict scenario from one scene directory without writing it."""

    if sample_rate_hz <= 0 or sample_rate_hz > 100:
        raise ValueError("sample_rate_hz must be in (0, 100]")
    if video_sample_fps <= 0 or video_sample_fps > 60:
        raise ValueError("video_sample_fps must be in (0, 60]")
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[3]
    )
    directory = Path(scene_dir).resolve()
    motion_path = _single_file(directory, "*_rider_motion.json", "rider motion file")
    scene_name = motion_path.name.removesuffix("_rider_motion.json")
    identity_suffix = re.sub(r"[^A-Za-z0-9]+", "-", scene_name).strip("-").lower()
    tag_id = tag_id or f"tag-{identity_suffix}"
    rider_id = rider_id or f"rider-{identity_suffix}"
    camera_path = directory / f"{scene_name}_camera_info.yaml"
    video_path = directory / f"{scene_name}.mp4"
    if not video_path.is_file():
        raise ValueError(f"scene video does not exist: {video_path}")

    motion = json.loads(motion_path.read_text(encoding="utf-8"))
    has_external_camera = camera_path.is_file()
    camera_document = (
        yaml.safe_load(camera_path.read_text(encoding="utf-8"))
        if has_external_camera
        else motion
    )
    if motion.get("schema_version") not in {"1.0", "1.1"}:
        raise ValueError("unsupported Blender rider-motion schema; expected 1.0 or 1.1")
    if camera_document.get("camera", {}).get("static") is not True:
        raise ValueError("blender-motion-v1 requires a static camera")

    camera_xyz = _xyz(
        camera_document.get("camera", {}).get("transform_world", {}).get("location_m"),
        "camera location",
    )
    embedded_camera_xyz = _xyz(
        motion.get("camera", {}).get("transform_world", {}).get("location_m"),
        "motion-file camera location",
    )
    if has_external_camera and any(
        abs(left - right) > 1e-6 for left, right in zip(camera_xyz, embedded_camera_xyz)
    ):
        raise ValueError("camera YAML and rider-motion JSON disagree on camera location")

    timing = motion.get("timing", {})
    frames = motion.get("frames")
    try:
        fps = float(timing["fps"])
        duration_s = float(timing["duration_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("motion timing must contain numeric fps and duration_s") from exc
    if fps <= 0 or duration_s <= 0 or not isinstance(frames, list) or not frames:
        raise ValueError("motion file must contain positive timing and at least one frame")

    last_time_s = float(frames[-1]["time_s"])
    sample_count = math.floor(last_time_s * sample_rate_hz + 1e-9) + 1
    samples: list[dict[str, Any]] = []
    for sample_index in range(sample_count):
        at_s = round(sample_index / sample_rate_hz, 9)
        frame_index = min(round(at_s * fps), len(frames) - 1)
        frame = frames[frame_index]
        rider_xyz = (
            float(frame["x_m"]),
            float(frame["y_m"]),
            float(frame["z_m"]),
        )
        distance_m = math.dist(camera_xyz, rider_xyz)
        # Blender exports world kinematic acceleration. A physical accelerometer
        # reports specific force a-g; world gravity is (0, 0, -g).
        acceleration = {
            "x": round(float(frame["ax_ms2"]), 6),
            "y": round(float(frame["ay_ms2"]), 6),
            "z": round(float(frame["az_ms2"]) + STANDARD_GRAVITY_MPS2, 6),
        }
        samples.append(
            {
                "at_s": at_s,
                "rssi_dbm": simulated_rssi_dbm(
                    distance_m,
                    reference_rssi_dbm_at_1m=reference_rssi_dbm_at_1m,
                    path_loss_exponent=path_loss_exponent,
                ),
                "acceleration_mps2": acceleration,
                "battery_voltage_mv": battery_voltage_mv,
                "source_frame": int(frame["frame"]),
                "source_distance_m": round(distance_m, 6),
            }
        )

    return ScenarioDefinition.model_validate(
        {
            "scenario_schema_version": "3.1",
            "name": f"{identity_suffix}-blender-regression",
            "seed": 0,
            "duration_s": duration_s,
            "timeline": [],
            "faults": {},
            "expect": {
                "rider_id": rider_id,
                "assignment_status": "assigned",
                "capture_triggered": True,
                "clip_uploaded": True,
                "minimum_person_detections": 1,
            },
            "components": {
                "frames": "video",
                "detector": "yolo",
                "bear_tag": "synthetic",
                "camera": "recorded_video",
                "storage": "memory",
            },
            "video": {
                "path": _repository_path(video_path, root),
                "sample_fps": video_sample_fps,
            },
            "detector": {
                "model": "yolov8n",
                "confidence_threshold": detector_confidence_threshold,
            },
            "synthetic_bear_tags": [
                {
                    "tag_id": tag_id,
                    "rider_id": rider_id,
                    "start_s": 0,
                    "end_s": duration_s,
                    "sample_rate_hz": sample_rate_hz,
                    "samples": samples,
                }
            ],
            "generated_from": {
                "generator": "blender-motion-v1",
                "motion_path": _repository_path(motion_path, root),
                "camera_path": (
                    _repository_path(camera_path, root) if has_external_camera else None
                ),
                "reference_rssi_dbm_at_1m": reference_rssi_dbm_at_1m,
                "path_loss_exponent": path_loss_exponent,
                "gravity_mps2": STANDARD_GRAVITY_MPS2,
            },
        }
    )


def write_scenario(
    scenario: ScenarioDefinition,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write generated YAML while refusing accidental replacement by default."""

    output = Path(output_path).resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"scenario already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(
            scenario.model_dump(mode="json", exclude_none=True),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a BearVision regression scenario from Blender exports"
    )
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tag-id")
    parser.add_argument("--rider-id")
    parser.add_argument("--sample-rate-hz", type=float, default=10.0)
    parser.add_argument("--video-sample-fps", type=float, default=10.0)
    parser.add_argument("--reference-rssi-dbm-at-1m", type=int, default=-50)
    parser.add_argument("--path-loss-exponent", type=float, default=2.0)
    parser.add_argument("--battery-voltage-mv", type=int, default=3000)
    parser.add_argument("--detector-confidence-threshold", type=float, default=0.6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    scenario = generate_blender_scenario(
        args.scene_dir,
        repository_root=root,
        tag_id=args.tag_id,
        rider_id=args.rider_id,
        sample_rate_hz=args.sample_rate_hz,
        video_sample_fps=args.video_sample_fps,
        reference_rssi_dbm_at_1m=args.reference_rssi_dbm_at_1m,
        path_loss_exponent=args.path_loss_exponent,
        battery_voltage_mv=args.battery_voltage_mv,
        detector_confidence_threshold=args.detector_confidence_threshold,
    )
    output = args.output or root / "specs" / "scenarios" / f"{scenario.name}.yaml"
    print(write_scenario(scenario, output, overwrite=args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
