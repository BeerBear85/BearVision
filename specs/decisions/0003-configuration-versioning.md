# ADR 0003: Configuration versioning

Status: accepted

Every formal configuration starts with `config_schema_version` followed by
`config_kind`. Configuration schema versions are independent of the application
version. Minor schema changes are additive; breaking changes increment the
schema major version and require an explicit migration.

The active edge, training, annotation and BLE test configuration schemas are
version 2.0. Version 2 removes runtime options that were declared but not
implemented and validates the version in each active consumer.
