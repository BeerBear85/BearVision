# BearVision 3 foundation

BearVision 3 is being rebuilt around executable specifications and deterministic
behavioural simulation. The simulator models component behaviour, timing and
failures. It does not claim physical accuracy and has no rendering dependency.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/remake
```

The active package lives under `src/bearvision`. Legacy implementations remain
available as migration sources but must not be imported by the new core.
