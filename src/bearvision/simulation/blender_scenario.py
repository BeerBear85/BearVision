"""Generate a video regression scenario from one or more Blender rider exports."""

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


def _motion_files(scene_dir: Path) -> tuple[Path, ...]:
    numbered: list[tuple[int, Path]] = []
    for path in scene_dir.glob("*_rider*_motion.json"):
        match = re.search(r"_rider(\d+)_motion\.json$", path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    if numbered:
        numbered.sort(key=lambda item: (item[0], item[1].name))
        return tuple(path for _, path in numbered)

    legacy = sorted(scene_dir.glob("*_rider_motion.json"))
    if len(legacy) != 1:
        raise ValueError(
            f"expected numbered rider motion files or one legacy rider motion file in "
            f"{scene_dir}, found {len(legacy)}"
        )
    return (legacy[0],)


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
    """Build a strict single- or multi-rider scenario without writing it."""

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
    motion_paths = _motion_files(directory)
    motions = [json.loads(path.read_text(encoding="utf-8")) for path in motion_paths]
    source_clips = {motion.get("source", {}).get("clip") for motion in motions}
    if len(source_clips) != 1 or not next(iter(source_clips), None):
        raise ValueError("all rider motion files must identify the same source clip")
    source_clip = str(next(iter(source_clips)))
    if Path(source_clip).name != source_clip or Path(source_clip).suffix.lower() != ".mp4":
        raise ValueError("motion source clip must be an MP4 filename without a directory")
    scene_name = Path(source_clip).stem
    identity_suffix = re.sub(r"[^A-Za-z0-9]+", "-", scene_name).strip("-").lower()
    if len(motions) > 1 and (tag_id is not None or rider_id is not None):
        raise ValueError("tag_id and rider_id overrides require a single-rider scene")
    camera_path = directory / f"{scene_name}_camera_info.yaml"
    video_path = directory / source_clip
    if not video_path.is_file():
        raise ValueError(f"scene video does not exist: {video_path}")

    has_external_camera = camera_path.is_file()
    embedded_camera = next((motion for motion in motions if motion.get("camera")), None)
    camera_document = yaml.safe_load(camera_path.read_text(encoding="utf-8")) if (
        has_external_camera
    ) else embedded_camera
    if camera_document is None:
        raise ValueError(f"scene camera metadata does not exist: {camera_path}")
    for motion in motions:
        if motion.get("schema_version") not in {"1.0", "1.1", "2.0"}:
            raise ValueError(
                "unsupported Blender rider-motion schema; expected 1.0, 1.1 or 2.0"
            )
    if camera_document.get("camera", {}).get("static") is not True:
        raise ValueError("blender-motion-v1 requires a static camera")

    camera_xyz = _xyz(
        camera_document.get("camera", {}).get("transform_world", {}).get("location_m"),
        "camera location",
    )
    if has_external_camera:
        for motion in motions:
            if not motion.get("camera"):
                continue
            embedded_camera_xyz = _xyz(
                motion.get("camera", {}).get("transform_world", {}).get("location_m"),
                "motion-file camera location",
            )
            if any(
                abs(left - right) > 1e-6
                for left, right in zip(camera_xyz, embedded_camera_xyz)
            ):
                raise ValueError("camera YAML and rider-motion JSON disagree on camera location")

    parsed: list[tuple[Path, dict[str, Any], float, float, int, list[dict[str, Any]]]] = []
    for motion_path, motion in zip(motion_paths, motions):
        timing = motion.get("timing", {})
        frames = motion.get("frames")
        try:
            fps = float(timing["fps"])
            duration_s = float(timing["duration_s"])
            frame_start = int(timing.get("frame_start", frames[0]["frame"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "motion timing must contain numeric fps, frame_start and duration_s"
            ) from exc
        if fps <= 0 or duration_s <= 0 or not isinstance(frames, list) or not frames:
            raise ValueError("motion file must contain positive timing and at least one frame")
        parsed.append((motion_path, motion, fps, duration_s, frame_start, frames))

    fps_values = {item[2] for item in parsed}
    if len(fps_values) != 1:
        raise ValueError("all rider motion files must use the same frame rate")
    fps = next(iter(fps_values))
    scene_frame_start = min(item[4] for item in parsed)
    generated_series: list[dict[str, Any]] = []
    used_tag_ids: set[str] = set()
    scenario_duration_s = 0.0
    expected_rider_id: str | None = None

    for rider_index, (motion_path, motion, _, duration_s, frame_start, frames) in enumerate(
        parsed, start=1
    ):
        start_s = round((frame_start - scene_frame_start) / fps, 9)
        end_s = round(start_s + duration_s, 9)
        scenario_duration_s = max(scenario_duration_s, end_s)
        embedded_tag_id = motion.get("bear_tag_id")
        resolved_tag_id = tag_id if len(parsed) == 1 and tag_id else embedded_tag_id
        if resolved_tag_id is None:
            resolved_tag_id = f"tag-{identity_suffix}"
        resolved_tag_id = str(resolved_tag_id)
        if resolved_tag_id in used_tag_ids:
            raise ValueError(f"duplicate BearTag ID in Blender exports: {resolved_tag_id}")
        used_tag_ids.add(resolved_tag_id)

        resolved_rider_id = rider_id if len(parsed) == 1 and rider_id else None
        if resolved_rider_id is None:
            suffix = "" if len(parsed) == 1 else f"-rider{rider_index}"
            resolved_rider_id = f"rider-{identity_suffix}{suffix}"
        if expected_rider_id is None:
            expected_rider_id = resolved_rider_id

        last_time_s = float(frames[-1]["time_s"])
        sample_count = math.floor(last_time_s * sample_rate_hz + 1e-9) + 1
        samples: list[dict[str, Any]] = []
        for sample_index in range(sample_count):
            local_at_s = round(sample_index / sample_rate_hz, 9)
            at_s = round(start_s + local_at_s, 9)
            frame_index = min(round(local_at_s * fps), len(frames) - 1)
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
        generated_series.append(
            {
                "tag_id": resolved_tag_id,
                "rider_id": resolved_rider_id,
                "start_s": start_s,
                "end_s": end_s,
                "sample_rate_hz": sample_rate_hz,
                "samples": samples,
            }
        )

    assert expected_rider_id is not None
    motion_provenance = (
        {"motion_path": _repository_path(motion_paths[0], root)}
        if len(motion_paths) == 1
        else {"motion_paths": [_repository_path(path, root) for path in motion_paths]}
    )

    return ScenarioDefinition.model_validate(
        {
            "scenario_schema_version": "3.1",
            "name": f"{identity_suffix}-blender-regression",
            "seed": 0,
            "duration_s": scenario_duration_s,
            "timeline": [],
            "faults": {},
            "expect": {
                "rider_id": expected_rider_id,
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
            "synthetic_bear_tags": generated_series,
            "generated_from": {
                "generator": "blender-motion-v1",
                **motion_provenance,
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
        description="Generate a BearVision regression scenario from Blender rider exports"
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
