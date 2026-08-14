# Contract rules

In-process domain contracts use `contract_schema_version: "2.0"`. The Box job
contract independently uses `schemaVersion: 1`. Both reject unknown fields and
declare units in field names. UTC timestamps must include a timezone.

Edge serializes observation times as non-negative `offsetMs` from
`captureStartedAt`; process-local monotonic values never cross machines. The
server reconstructs clip-relative values only for scoring. `validFrom` is
inclusive and `validTo` exclusive.
