# Contract rules

All serialized domain contracts use `contract_schema_version: "1.0"`, reject
unknown fields and declare units in field names. UTC timestamps must include a
timezone. Monotonic timestamps are seconds from the current process or virtual
scenario origin.

Vision can trigger capture, but rider identity is assigned exclusively from
registered BLE tags. When zero or multiple tags qualify, the result is not an
assignment; callers must retain the ambiguity explicitly.
