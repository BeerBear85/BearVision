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
available behind production adapters; domain and simulation code do not import
their SDKs.

Run the first complete behavioural scenario with:

```bash
python tools/run_behavioral_scenario.py specs/scenarios/single-rider-success.yaml
```

## Behavioural composition

```mermaid
flowchart LR
    Scenario[Versioned scenario] --> Engine[Deterministic event engine]
    Engine --> Core[Assignment and capture orchestration]
    Core --> Camera[Camera port]
    Core --> Scanner[BLE scanner port]
    Core --> Detector[Detector port]
    Core --> Storage[Storage port]
    Scanner --> Core
    Detector --> Core
    Sim[Simulated adapters] --> Camera
    Sim --> Scanner
    Sim --> Detector
    Sim --> Storage
    Real[GoPro / KBeacon / YOLO / Box adapters] --> Camera
    Real --> Scanner
    Real --> Detector
    Real --> Storage
```

Vision can trigger capture but never supplies rider identity. Only recent,
registered BLE observations enter the assignment policy. Zero candidates are
`unassigned`; multiple candidates are `ambiguous`.
