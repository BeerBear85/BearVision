"""Typed events emitted by the Edge runtime process."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .jobs import JobResultManifest
from .models import BoundingBox, PersonDetection, Vector3


RuntimeEventKind: TypeAlias = Literal[
    "tag_enters_range",
    "tag_observation",
    "preview_frame",
    "person_detected",
    "capture_started",
    "finalize_clip",
    "capture_completed",
    "tracking_observation",
    "virtual_cameraman_completed",
    "clip_uploaded",
    "server_assignment",
    "expectation_failed",
    "component_failed",
    "hardware_initializing",
    "hardware_stopping",
]


class EventModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CoordinateSpace(EventModel):
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)


class Point(EventModel):
    x_px: float
    y_px: float


class TagPayload(EventModel):
    tag_id: str = Field(min_length=1)
    rider_id: str | None = Field(default=None, min_length=1)
    rssi_dbm: int = Field(ge=-127, le=20)
    acceleration_mps2: Vector3
    battery_voltage_mv: int | None = Field(default=None, ge=0, le=10_000)


class PreviewFramePayload(EventModel):
    frame_id: str = Field(min_length=1)


class PersonDetectedPayload(EventModel):
    frame_id: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    bounding_box: BoundingBox
    coordinate_space: CoordinateSpace


class CaptureStartedPayload(EventModel):
    asset_id: str = Field(min_length=1)
    clip_end_s: float = Field(ge=0)


class FinalizeClipPayload(EventModel):
    request_id: str = Field(min_length=1)


class CaptureCompletedPayload(EventModel):
    asset_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    clip_start_s: float = Field(ge=0)
    clip_duration_s: float = Field(gt=0)


class LengthAdjustmentPayload(EventModel):
    padding_s: float = Field(ge=0)
    source_start_frame_idx: int = Field(ge=0)
    source_end_frame_idx_exclusive: int = Field(gt=0)
    first_visible_frame_idx: int = Field(ge=0)
    last_visible_frame_idx: int = Field(ge=0)
    source_start_s: float = Field(ge=0)
    source_end_s: float = Field(gt=0)
    source_duration_s: float = Field(gt=0)
    output_duration_s: float = Field(gt=0)
    adjusted: bool


class VirtualCameramanCompletedPayload(EventModel):
    source_filename: str = Field(min_length=1)
    processed_filename: str = Field(min_length=1)
    tracking_filename: str = Field(min_length=1)
    debug_video_filename: str = Field(min_length=1)
    source_size_bytes: int = Field(ge=0)
    processed_size_bytes: int = Field(ge=0)
    size_reduction_ratio: float
    output_width_px: int = Field(gt=0)
    output_height_px: int = Field(gt=0)
    state_estimator: Literal["kalman_rts_smoother"]
    camera_path: Literal["zero_phase_butterworth"]
    length_adjustment: LengthAdjustmentPayload


class TrackingObservationPayload(EventModel):
    frame_idx: int = Field(ge=0)
    at_s: float = Field(ge=0)
    source_frame_idx: int = Field(ge=0)
    source_at_s: float = Field(ge=0)
    estimate: Point
    confidence_radius_95_px: float = Field(ge=0)
    position_covariance_px2: tuple[tuple[float, float], tuple[float, float]]
    camera_center: Point
    crop_box: BoundingBox
    detection: PersonDetection | None
    coordinate_space: CoordinateSpace


class ClipUploadedPayload(EventModel):
    asset_id: str = Field(min_length=1)
    object_key: str = Field(min_length=1)


class MessagePayload(EventModel):
    message: str = Field(min_length=1)


class ComponentFailedPayload(EventModel):
    component: str = Field(min_length=1)
    error: str = Field(min_length=1)


class HardwareInitializingPayload(EventModel):
    config: str = Field(min_length=1)


class HardwareStoppingPayload(EventModel):
    reason: str = Field(min_length=1)


class RuntimeEventBase(EventModel):
    control_event_version: Literal["1.0"] = "1.0"
    at_s: float | None = Field(default=None, ge=0)


class TagRuntimeEvent(RuntimeEventBase):
    kind: Literal["tag_enters_range", "tag_observation"]
    payload: TagPayload


class PreviewFrameRuntimeEvent(RuntimeEventBase):
    kind: Literal["preview_frame"]
    payload: PreviewFramePayload


class PersonDetectedRuntimeEvent(RuntimeEventBase):
    kind: Literal["person_detected"]
    payload: PersonDetectedPayload


class CaptureStartedRuntimeEvent(RuntimeEventBase):
    kind: Literal["capture_started"]
    payload: CaptureStartedPayload


class FinalizeClipRuntimeEvent(RuntimeEventBase):
    kind: Literal["finalize_clip"]
    payload: FinalizeClipPayload


class CaptureCompletedRuntimeEvent(RuntimeEventBase):
    kind: Literal["capture_completed"]
    payload: CaptureCompletedPayload


class TrackingObservationRuntimeEvent(RuntimeEventBase):
    kind: Literal["tracking_observation"]
    payload: TrackingObservationPayload


class VirtualCameramanCompletedRuntimeEvent(RuntimeEventBase):
    kind: Literal["virtual_cameraman_completed"]
    payload: VirtualCameramanCompletedPayload


class ClipUploadedRuntimeEvent(RuntimeEventBase):
    kind: Literal["clip_uploaded"]
    payload: ClipUploadedPayload


class ServerAssignmentRuntimeEvent(RuntimeEventBase):
    kind: Literal["server_assignment"]
    payload: JobResultManifest


class ExpectationFailedRuntimeEvent(RuntimeEventBase):
    kind: Literal["expectation_failed"]
    payload: MessagePayload


class ComponentFailedRuntimeEvent(RuntimeEventBase):
    kind: Literal["component_failed"]
    payload: ComponentFailedPayload


class HardwareInitializingRuntimeEvent(RuntimeEventBase):
    kind: Literal["hardware_initializing"]
    payload: HardwareInitializingPayload


class HardwareStoppingRuntimeEvent(RuntimeEventBase):
    kind: Literal["hardware_stopping"]
    payload: HardwareStoppingPayload


RuntimeEvent = Annotated[
    TagRuntimeEvent
    | PreviewFrameRuntimeEvent
    | PersonDetectedRuntimeEvent
    | CaptureStartedRuntimeEvent
    | FinalizeClipRuntimeEvent
    | CaptureCompletedRuntimeEvent
    | TrackingObservationRuntimeEvent
    | VirtualCameramanCompletedRuntimeEvent
    | ClipUploadedRuntimeEvent
    | ServerAssignmentRuntimeEvent
    | ExpectationFailedRuntimeEvent
    | ComponentFailedRuntimeEvent
    | HardwareInitializingRuntimeEvent
    | HardwareStoppingRuntimeEvent,
    Field(discriminator="kind"),
]


RUNTIME_EVENT_ADAPTER: TypeAdapter[RuntimeEvent] = TypeAdapter(RuntimeEvent)


def serialize_runtime_event(
    kind: RuntimeEventKind,
    payload: object,
    *,
    at_s: float | None = None,
) -> str:
    """Validate one complete event before it crosses the process seam."""

    event = RUNTIME_EVENT_ADAPTER.validate_python(
        {
            "control_event_version": "1.0",
            "at_s": at_s,
            "kind": kind,
            "payload": payload,
        }
    )
    return event.model_dump_json(by_alias=True)
