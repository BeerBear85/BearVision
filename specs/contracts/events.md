# Contract rules

All serialized domain contracts use `contract_schema_version: "2.0"`, reject
unknown fields and declare units in field names. UTC timestamps must include a
timezone. Monotonic timestamps are seconds from the current process or virtual
scenario origin.

Vision triggers a fixed-duration person clip but does not provide rider identity
or a jump timestamp. Identity is assigned exclusively from acceleration and RSSI
evidence recorded by registered BearTags during the complete clip. A tag must
pass sample-count, mean-motion and RSSI gates. When no tag qualifies, or competing
combined scores are too close, callers retain `unassigned` or `ambiguous`.
