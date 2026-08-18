# GoPro controller simulator

Status: first high-fidelity control slice implemented; endurance, storage-full,
camera sleep and UDP fault simulation remain open.

## Purpose

`SimulatedGoProController` implements the controller API consumed by
`GoProCameraAdapter`. This deliberately places the simulator below the
production adapter so capture, download, error translation and request
idempotency use the same code in simulation and on hardware.

The older `SimulatedCamera` remains a fast in-memory test double for synthetic
scenario tests. It is not a GoPro emulator and must not be used to validate
camera timing or media behaviour.

## Modelled behaviour

- explicit connect and disconnect lifecycle;
- preview start/stop state with a recorded MP4 as the preview source;
- persistent HindSight setting with supported 15- and 30-second windows;
- shutter/encoding conflicts;
- rolling-buffer capture window (`trigger - HindSight` through shutter stop);
- GoPro-style `100GOPRO/GX01xxxx.MP4` media names;
- a persistent simulated SD card and media list;
- download, individual delete and delete-all operations;
- state inspection for encoding, preview and HindSight;
- protection against paths escaping the simulated SD card.

Captured media is extracted through the production `VideoClipper`, so normal
simulation runs can create valid MP4 files rather than content-type-labelled
placeholder bytes.

## Disk layout

```text
simulated-gopro/
├── camera-state.json
└── 100GOPRO/
    ├── GX010001.MP4
    └── GX010002.MP4
```

The Edge download directory is separate from this SD-card directory. This
matches the physical flow: record on camera, list media, then download a copy to
the Edge computer.

## Known transport difference

The physical HERO12 preview is MPEG-TS over UDP. The controller must return the
local listener URL `udp://@0.0.0.0:8554`; returning the camera IP prevents the
production OpenCV receiver from opening correctly.

The simulator currently returns its recorded preview file path. The Camera port
treats preview locations as opaque, so the production frame source can consume
both. A future network-fault slice may add an actual local UDP transport when
packet loss, startup delay and stream interruption need to be tested.

## Remaining fidelity gaps

- SD-card capacity and storage-full errors;
- sleep, wake and USB reconnect behaviour;
- delayed media-list visibility after shutter stop;
- chaptered recordings and multi-file captures;
- preview packet loss, jitter and firewall behaviour;
- battery, temperature and firmware-specific status values;
- automatic integration into every behavioural scenario composition.
