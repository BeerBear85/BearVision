# Contract rules

All serialized domain contracts use `contract_schema_version: "1.0"`, reject
unknown fields and declare units in field names. UTC timestamps must include a
timezone. Monotonic timestamps are seconds from the current process or virtual
scenario origin.

Vision can trigger capture and provide the jump timestamp, but rider identity
is assigned exclusively from synchronized acceleration and RSSI evidence from
registered BearTags. A tag must pass both evidence gates. When no tag qualifies,
or competing combined scores are too close, callers retain the uncertainty as
`unassigned` or `ambiguous`.
