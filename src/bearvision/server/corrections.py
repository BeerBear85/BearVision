"""Authoritative operator corrections, impact analysis, and recomputation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from bearvision.config import AssignmentConfig
from bearvision.contracts import BearTagJobObservation, EdgeJobManifest, JobResultManifest
from bearvision.domain import ALGORITHM_VERSION
from bearvision.ports import ManagedJobQueue

from .registry import BearTagAssignment, FileUserRegistry, RegistryData


class AssignmentCorrectionService:
    def __init__(
        self,
        queue: ManagedJobQueue,
        registry: FileUserRegistry,
        assignment_policy: AssignmentConfig,
    ) -> None:
        self.queue = queue
        self.registry = registry
        self.assignment_policy = assignment_policy

    async def manual_reassign(self, job_id: str, user_id: UUID, reason: str) -> dict[str, Any]:
        data = self.registry.load()
        target = next((item for item in data.users if item.id == user_id), None)
        if target is None:
            raise FileNotFoundError("The selected user no longer exists.")
        current = JobResultManifest.model_validate_json(
            await self.queue.admin_read(job_id, "result.json")
        )
        if current.status not in {"processed", "unresolved"}:
            raise ValueError("Only processed or unresolved clips can be reassigned.")
        if current.selected_user_id == user_id:
            raise ValueError("Choose a different user for the manual assignment.")
        changed_at = datetime.now(timezone.utc)
        result = JobResultManifest(
            schemaVersion=3,
            jobId=job_id,
            status="processed",
            processedAt=changed_at,
            algorithmVersion=current.algorithm_version or ALGORITHM_VERSION,
            selectedBearTagId=current.selected_bear_tag_id,
            selectedUserId=user_id,
            assignmentId=f"manual-{uuid4().hex}",
            candidates=current.candidates,
            reason=f"Manually assigned: {reason.strip()}",
            assignmentSource="manual",
            manuallyAssignedAt=changed_at,
            replacedUserId=current.selected_user_id,
            manualReason=reason.strip(),
        )
        await self.queue.admin_replace_result(job_id, result, user_id)
        return {
            "jobId": job_id,
            "status": result.status,
            "currentUserId": str(current.selected_user_id) if current.selected_user_id else None,
            "newUserId": str(user_id),
            "newDisplayName": target.display_name,
            "assignmentSource": "manual",
            "manuallyAssignedAt": changed_at.isoformat(),
            "replacedUserId": str(current.selected_user_id) if current.selected_user_id else None,
            "reason": reason.strip(),
        }

    async def preview_history_change(
        self,
        assignment_id: str,
        replacements: tuple[BearTagAssignment, ...],
        *,
        override_manual_assignments: bool,
    ) -> dict[str, Any]:
        original, proposed = self.registry.preview_assignment_replacement(
            assignment_id, replacements
        )
        plan = await self._plan(original, proposed, override_manual_assignments)
        return self._present(plan)

    async def apply_history_change(
        self,
        assignment_id: str,
        replacements: tuple[BearTagAssignment, ...],
        *,
        reason: str,
        override_manual_assignments: bool,
    ) -> dict[str, Any]:
        original, proposed = self.registry.preview_assignment_replacement(
            assignment_id, replacements
        )
        plan = await self._plan(original, proposed, override_manual_assignments)
        self.registry.replace_assignment(
            assignment_id, replacements, reason=reason, recorded_at=datetime.now(timezone.utc)
        )
        for item in plan:
            if not item["changed"]:
                continue
            result: JobResultManifest = item["expectedResult"]
            await self.queue.admin_replace_result(item["jobId"], result, result.selected_user_id)
        applied = self._present(plan)
        applied["applied"] = True
        return applied

    async def _plan(
        self,
        original: BearTagAssignment,
        proposed: RegistryData,
        override_manual_assignments: bool,
    ) -> list[dict[str, Any]]:
        users_before = {str(item.id): item.display_name for item in self.registry.load().users}
        users_after = {str(item.id): item.display_name for item in proposed.users}
        plan: list[dict[str, Any]] = []
        for entry in self.queue.admin_list_jobs():
            if entry["status"] not in {"processed", "unresolved"}:
                continue
            manifest = EdgeJobManifest.model_validate_json(
                await self.queue.admin_read(entry["jobId"], "manifest.json")
            )
            if not (
                original.valid_from < manifest.capture_ended_at
                and manifest.capture_started_at < original.valid_to
            ):
                continue
            observations = tuple(
                BearTagJobObservation.model_validate_json(line)
                for line in (
                    await self.queue.admin_read(entry["jobId"], manifest.observations_filename)
                )
                .decode("utf-8")
                .splitlines()
                if line.strip()
            )
            current = JobResultManifest.model_validate_json(
                await self.queue.admin_read(entry["jobId"], "result.json")
            )
            uses_tag = current.selected_bear_tag_id == original.bear_tag_id or any(
                item.bear_tag_id == original.bear_tag_id for item in observations
            )
            if not uses_tag:
                continue
            protected = current.assignment_source == "manual" and not override_manual_assignments
            expected = (
                current
                if protected
                else proposed.decide_job(
                    manifest,
                    observations,
                    assignment_policy=self.assignment_policy,
                    processed_at=datetime.now(timezone.utc),
                )
            )
            changed = self._decision_key(current) != self._decision_key(expected)
            plan.append(
                {
                    "jobId": entry["jobId"],
                    "currentResult": current,
                    "expectedResult": expected,
                    "current": self._state(current, users_before),
                    "expected": self._state(expected, users_after),
                    "changed": changed,
                    "manualProtected": protected,
                }
            )
        plan.sort(key=lambda item: item["jobId"])
        return plan

    @staticmethod
    def _decision_key(result: JobResultManifest) -> tuple[Any, ...]:
        return (
            result.status,
            result.selected_user_id,
            result.selected_bear_tag_id,
            result.assignment_id,
            result.assignment_source,
        )

    @staticmethod
    def _state(result: JobResultManifest, users: dict[str, str]) -> dict[str, Any]:
        user_id = str(result.selected_user_id) if result.selected_user_id else None
        return {
            "status": result.status,
            "userId": user_id,
            "displayName": users.get(user_id or ""),
            "assignmentId": result.assignment_id,
            "assignmentSource": result.assignment_source,
            "reason": result.reason,
        }

    @staticmethod
    def _present(plan: list[dict[str, Any]]) -> dict[str, Any]:
        items = [
            {
                "jobId": item["jobId"],
                "current": item["current"],
                "expected": item["expected"],
                "changed": item["changed"],
                "manualProtected": item["manualProtected"],
            }
            for item in plan
        ]
        return {
            "items": items,
            "counts": {
                "changed": sum(item["changed"] for item in plan),
                "unchanged": sum(not item["changed"] for item in plan),
                "unresolved": sum(item["expected"]["status"] == "unresolved" for item in plan),
            },
        }
