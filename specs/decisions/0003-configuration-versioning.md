# ADR 0003: Configuration versioning

Status: accepted

Every formal configuration starts with `config_schema_version` followed by
`config_kind`. Configuration schema versions are independent of the application
version. Minor schema changes are additive; breaking changes increment the
schema major version and require an explicit migration.
