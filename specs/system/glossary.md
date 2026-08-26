# BearVision 3 glossary

- **Rider:** A registered person who owns one BearTag.
- **Tag observation:** One decoded BLE advertisement with timestamp, RSSI and acceleration.
- **Rider assignment:** A BearTag decision combining whole-clip mean acceleration
  activity and RSSI evidence to link a capture to one registered rider.
- **Detection:** A vision result which may trigger capture but cannot determine rider identity.
- **Preview:** The live camera stream used for detection; it is not the retained raw clip.
- **HindSight:** GoPro's configured rolling-buffer mechanism.
- **Pre-roll:** The retained part of a clip before the authoritative detection timestamp.
- **Capture request:** A request to retain camera media around a preview detection.
- **Raw clip:** The original, unmodified camera file including available pre-roll and
  the fixed post-detection recording interval.
- **Requested window:** The detection-centred interval requested from the camera,
  clamped to the earliest available media after startup.
- **Actual window:** The exact or estimated interval delivered in the raw clip.
- **Processed clip:** Media produced from the raw clip by Virtual Cameraman or other processing.
- **Media asset:** Provider-neutral metadata for one captured or processed file.
- **Scenario:** Versioned deterministic input and expected outcomes for behavioural simulation.
- **Ground truth:** Expected behavioural facts declared by a scenario.
