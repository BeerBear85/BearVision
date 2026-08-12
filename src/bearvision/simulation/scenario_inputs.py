"""Generate deterministic inputs declared compactly in scenario schema 3.0."""

from __future__ import annotations

from datetime import timedelta

from bearvision.contracts import (
    SyntheticBearTagSeries,
    TagObservation,
    TagRegistryEntry,
)

from .adapters import VirtualClock


def generate_bear_tag_series(
    definitions: tuple[SyntheticBearTagSeries, ...],
    clock: VirtualClock,
) -> tuple[tuple[TagObservation, ...], tuple[TagRegistryEntry, ...]]:
    observations: list[TagObservation] = []
    registry: list[TagRegistryEntry] = []
    for definition in definitions:
        registry.append(
            TagRegistryEntry(tag_id=definition.tag_id, rider_id=definition.rider_id)
        )
        count = int(round((definition.end_s - definition.start_s) * definition.sample_rate_hz))
        for index in range(count + 1):
            at_s = round(
                definition.start_s + (index / definition.sample_rate_hz),
                9,
            )
            acceleration = definition.baseline_acceleration_mps2
            rssi_dbm = definition.rssi_dbm
            for window in definition.motion_windows:
                if window.start_s <= at_s <= window.end_s:
                    acceleration = window.acceleration_mps2
                    rssi_dbm = window.rssi_dbm if window.rssi_dbm is not None else rssi_dbm
            observations.append(
                TagObservation(
                    tag_id=definition.tag_id,
                    observed_at_utc=clock.start_utc + timedelta(seconds=at_s),
                    observed_at_monotonic_s=at_s,
                    rssi_dbm=rssi_dbm,
                    acceleration_mps2=acceleration,
                )
            )
    return (
        tuple(sorted(observations, key=lambda item: item.observed_at_monotonic_s)),
        tuple(registry),
    )
