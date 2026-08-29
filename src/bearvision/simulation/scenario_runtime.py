"""Shared assembly and result rules for behavioural Scenario runners."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, TypeAlias
from uuid import NAMESPACE_URL, UUID, uuid5

from bearvision.config import AssignmentConfig
from bearvision.contracts import (
    JobResultManifest,
    RuntimeEventKind,
    ScenarioDefinition,
    StorageReceipt,
    TagRegistryEntry,
)
from bearvision.edge.orchestrator import OrchestrationResult
from bearvision.ports import Clock, JobQueue
from bearvision.server import (
    BearTagAssignment,
    BearTagRecord,
    InMemoryUserRegistry,
    RegistryData,
    ServerWorker,
    UserRecord,
    normalize_user_email,
)

from .engine import TraceEntry


TraceEvent: TypeAlias = tuple[float, RuntimeEventKind, dict[str, Any]]


def scenario_email(rider_id: str) -> str:
    return normalize_user_email(
        rider_id if "@" in rider_id else f"{rider_id}@scenario.invalid"
    )


def scenario_user_id(rider_id: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"bearvision:scenario-user:{scenario_email(rider_id)}")


def build_scenario_registry(
    entries: Iterable[TagRegistryEntry],
    clock: Clock,
) -> InMemoryUserRegistry:
    """Build the deterministic user/tag assignments shared by Scenario runners."""

    ordered = tuple(sorted(entries, key=lambda item: item.tag_id))
    riders = tuple(sorted({item.rider_id for item in ordered}))
    valid_at = clock.utc_now()
    return InMemoryUserRegistry(
        RegistryData(
            users=tuple(
                UserRecord(
                    id=scenario_user_id(rider_id),
                    email=scenario_email(rider_id),
                    displayName=rider_id,
                )
                for rider_id in riders
            ),
            bearTags=tuple(BearTagRecord(id=item.tag_id) for item in ordered),
            assignments=tuple(
                BearTagAssignment(
                    id=f"scenario-{item.tag_id}",
                    userId=scenario_user_id(item.rider_id),
                    bearTagId=item.tag_id,
                    validFrom=valid_at - timedelta(days=1),
                    validTo=valid_at + timedelta(days=1),
                )
                for item in ordered
            ),
        )
    )


def build_scenario_worker(
    *,
    entries: Iterable[TagRegistryEntry],
    queue: JobQueue,
    clock: Clock,
    assignment_policy: AssignmentConfig | None,
    enabled: bool,
) -> ServerWorker | None:
    """Honor the Scenario server boundary while sharing worker assembly."""

    if not enabled:
        return None
    return ServerWorker(
        queue,
        build_scenario_registry(entries, clock),
        clock,
        assignment_policy,
    )


@dataclass(frozen=True, slots=True)
class ScenarioRunResult:
    trace: tuple[TraceEntry, ...]
    assignments: tuple[JobResultManifest, ...]
    captures: tuple[str, ...]
    uploads: tuple[StorageReceipt, ...]
    failures: tuple[dict[str, str], ...]
    expectation_failures: tuple[str, ...] = ()


def order_trace_events(events: Sequence[TraceEvent]) -> tuple[TraceEntry, ...]:
    """Order logical timestamps stably and assign contiguous trace sequence IDs."""

    ordered = sorted(enumerate(events), key=lambda item: (item[1][0], item[0]))
    return tuple(
        TraceEntry(at_s=at_s, sequence=sequence, kind=kind, payload=payload)
        for sequence, (_, (at_s, kind, payload)) in enumerate(ordered)
    )


def published_receipts(
    results: Iterable[OrchestrationResult],
) -> tuple[StorageReceipt, ...]:
    return tuple(
        StorageReceipt(
            asset_id=result.media.asset.asset_id,
            object_key=f"input-queue/ready/{result.request_id}",
            stored_at_utc=result.manifest.created_at,
            checksum_sha256=result.manifest.video.sha256,
        )
        for result in results
        if result.published
    )


def evaluate_expectations(
    scenario: ScenarioDefinition,
    assignments: tuple[JobResultManifest, ...],
    captures: tuple[str, ...],
    uploads: tuple[StorageReceipt, ...],
    *,
    detection_times_s: tuple[float, ...] = (),
    evaluate_server: bool = True,
) -> tuple[str, ...]:
    expected = scenario.expect
    failures: list[str] = []
    first = assignments[0] if assignments else None
    if evaluate_server and expected.rider_ids:
        expected_user_ids = tuple(scenario_user_id(rider) for rider in expected.rider_ids)
        actual_user_ids = tuple(item.selected_user_id for item in assignments)
        if actual_user_ids != expected_user_ids:
            failures.append(
                f"expected rider_ids={expected.rider_ids!r}, got {actual_user_ids!r}"
            )
    elif evaluate_server and expected.rider_id is not None:
        expected_user_id = scenario_user_id(expected.rider_id)
        if first is None or first.selected_user_id != expected_user_id:
            actual = first.selected_user_id if first is not None else None
            failures.append(f"expected rider_id={expected.rider_id!r}, got {actual!r}")
    if evaluate_server and expected.bear_tag_ids:
        actual_tag_ids = tuple(item.selected_bear_tag_id for item in assignments)
        if actual_tag_ids != expected.bear_tag_ids:
            failures.append(
                f"expected bear_tag_ids={expected.bear_tag_ids!r}, got {actual_tag_ids!r}"
            )
    elif evaluate_server and expected.bear_tag_id is not None:
        actual_tag_id = first.selected_bear_tag_id if first is not None else None
        if actual_tag_id != expected.bear_tag_id:
            failures.append(
                f"expected bear_tag_id={expected.bear_tag_id!r}, got {actual_tag_id!r}"
            )
    if evaluate_server and expected.assignment_status is not None:
        actual_status: str | None = None
        if first is not None:
            if first.status == "processed":
                actual_status = "assigned"
            elif first.error_code == "AMBIGUOUS_BEARTAG":
                actual_status = "ambiguous"
            else:
                actual_status = "unassigned"
        if actual_status != expected.assignment_status:
            failures.append(
                f"expected assignment_status={expected.assignment_status!r}, "
                f"got {actual_status!r}"
            )
    if expected.capture_triggered is not None and bool(captures) != expected.capture_triggered:
        failures.append(
            f"expected capture_triggered={expected.capture_triggered}, got {bool(captures)}"
        )
    if expected.clip_uploaded is not None and bool(uploads) != expected.clip_uploaded:
        failures.append(f"expected clip_uploaded={expected.clip_uploaded}, got {bool(uploads)}")
    if (
        expected.minimum_person_detections is not None
        and len(detection_times_s) < expected.minimum_person_detections
    ):
        failures.append(
            f"expected at least {expected.minimum_person_detections} person detections, "
            f"got {len(detection_times_s)}"
        )
    if expected.first_detection_between_s is not None:
        start, end = expected.first_detection_between_s
        detection_at = detection_times_s[0] if detection_times_s else None
        if detection_at is None or not start <= detection_at <= end:
            failures.append(
                f"expected first detection in [{start}, {end}], got {detection_at!r}"
            )
    return tuple(failures)


def finalize_scenario_run(
    *,
    scenario: ScenarioDefinition,
    trace_events: Sequence[TraceEvent],
    assignments: Iterable[JobResultManifest],
    captures: Iterable[str],
    edge_results: Iterable[OrchestrationResult],
    failures: Iterable[dict[str, str]],
    detection_times_s: Iterable[float],
    evaluate_server: bool,
) -> ScenarioRunResult:
    """Apply the result, receipt, ordering and expectation rules once."""

    assignment_results = tuple(assignments)
    capture_results = tuple(captures)
    uploads = published_receipts(edge_results)
    return ScenarioRunResult(
        trace=order_trace_events(trace_events),
        assignments=assignment_results,
        captures=capture_results,
        uploads=uploads,
        failures=tuple(failures),
        expectation_failures=evaluate_expectations(
            scenario,
            assignment_results,
            capture_results,
            uploads,
            detection_times_s=tuple(detection_times_s),
            evaluate_server=evaluate_server,
        ),
    )
