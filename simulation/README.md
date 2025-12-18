# Wakeboard Cable-Park Simulation

A minimal, deterministic simulation of a wakeboarder being pulled around a fixed cable loop using Panda3D and Bullet physics.

## Overview

This simulation models a wakeboarder moving along a cable-park system defined by tower coordinates. The physics are simplified but deterministic, making it suitable for:

- Generating synthetic sensor data (position, velocity, acceleration)
- Testing motion tracking algorithms
- Validating computer vision pipelines
- CI/regression testing with reproducible outputs

## Features

- **Deterministic physics**: Fixed timestep with configurable substeps
- **Headless mode**: Runs without graphics for CI/automated testing
- **Visual mode**: Interactive 3D visualization with camera following
- **CSV logging**: Outputs position, velocity, IMU-like acceleration data
- **Screenshot capture**: Optional frame capture for image generation
- **Simple physics model**: Wakeboarder as rigid body pulled along cable path

## Installation

```bash
cd simulation
pip install -r requirements.txt
```

Requirements:
- Python 3.8+
- Panda3D 1.10.14 (includes Bullet physics engine)
- NumPy
- pytest (for tests)

## Usage

### Basic Visual Simulation

Run a 30-second simulation with visual window:

```bash
python -m simulation.run_sim --duration 30 --speed 10
```

The window will show:
- Water plane (blue surface)
- Cable towers (yellow boxes)
- Wakeboarder (red box)
- Camera following the wakeboarder

### Headless Mode (for CI)

Run simulation without graphics:

```bash
python -m simulation.run_sim --headless --duration 10
```

### With Screenshots

Capture screenshots every 2 seconds:

```bash
python -m simulation.run_sim --duration 10 --screenshot-every 2.0 --screenshot-dir simulation/out/screenshots
```

Screenshots saved to `simulation/out/screenshots/t_SSSSSS.SSs.png` (e.g., `t_002.00s.png`)

**Note**: Screenshot capture works best in windowed mode. Headless mode has known limitations (see Known Issues below).

### Full CLI Options

```bash
python -m simulation.run_sim [OPTIONS]

Options:
  --headless                  Run without graphics window
  --duration FLOAT            Simulation duration in seconds (default: 20.0)
  --dt FLOAT                  Physics timestep in seconds (default: 0.008)
  --speed FLOAT               Wakeboarder speed along cable (default: 30 km/h = 8.33 m/s)
  --outdir PATH               Output directory (default: simulation/out)
  --seed INT                  Random seed for reproducibility (default: 0)
  --screenshot-every SECONDS  Save screenshot every N seconds (0=disabled, default: 0)
  --screenshot-dir PATH       Screenshot output directory (default: outdir/screenshots)
```

## Output Format

### CSV Log

Each simulation run creates a timestamped CSV file:
`simulation/out/run_YYYYMMDD_HHMMSS.csv`

Columns:
- `time`: Simulation time in seconds
- `pos_x`, `pos_y`, `pos_z`: Position in meters (local XY frame)
- `vel_x`, `vel_y`, `vel_z`: Velocity in m/s
- `accel_x`, `accel_y`, `accel_z`: Acceleration in m/s² (from finite difference)
- `path_s`: Distance along cable path in meters
- `segment_index`: Current cable segment (0-5 for 6 towers)

Example:
```csv
time,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,accel_x,accel_y,accel_z,path_s,segment_index
0.000000,0.000000,0.000000,0.100000,7.947368,-6.012987,0.000000,0.000000,0.000000,0.000000,0.000000,0
0.008000,0.063579,-0.048104,0.099999,7.947368,-6.012987,0.000000,0.000000,0.000000,0.000000,0.064000,0
...
```

### Acceleration Calculation

Acceleration is computed using finite difference:

```python
accel = (vel_current - vel_previous) / dt
```

This provides IMU-like acceleration estimates, with spikes at corners where the wakeboarder changes direction.

## Cable Path Geometry

The cable loop is defined by 6 tower points in counter-clockwise order (meters, local XY):

```python
T0: (  0.00,   0.00)    # Reference tower
T1: ( 73.58, -55.61)
T2: (101.30, -50.12)
T3: (167.24,  80.92)
T4: (155.96, 126.04)
T5: (110.79, 114.50)
```

The wakeboarder follows a piecewise-linear path connecting these towers in a closed loop.

## Physics Model

The simulation uses a simplified physics model:

### Wakeboarder
- Represented as a rigid body (box) with 80 kg mass
- Pulled along cable path using PD controller
- Target position advances at constant speed along path
- Vertical position constrained near water surface (z ≈ 0.1m)

### Forces Applied
1. **Horizontal pulling force**: PD control toward target position on cable
   - Proportional term: `F = k_p * position_error`
   - Derivative term: `F += k_d * velocity_error`

2. **Vertical spring**: Keeps wakeboarder near water surface
   - Spring force: `F_z = k_z * (z_target - z_current)`
   - Damping: `F_z += c_z * (-vz)`

### Determinism
- Fixed timestep (default 0.008s)
- Fixed Bullet substeps (4 substeps per frame)
- Fixed solver iterations (10)
- Seeded random number generator

## Running Tests

```bash
cd simulation
pytest tests/ -v
```

Tests include:
- `test_path_length_positive`: Verify cable path has positive length
- `test_wraparound_continuity`: Position at s=0 equals s=total_length
- `test_sim_runs_headless_for_2s`: Full simulation runs and produces valid CSV
- `test_10s_simulation_with_screenshots`: Visual regression test (captures screenshots at t=0,2,4,6,8,10s)

### Visual Regression Testing

Run the visual regression test to verify rendering quality:

```bash
pytest simulation/tests/test_visual_regression.py -v -s
```

This test:
1. Runs a 10-second simulation with screenshots every 2 seconds
2. Captures 6 screenshots at t=0, 2, 4, 6, 8, 10 seconds
3. Verifies screenshots are non-empty and contain visual content
4. Saves screenshots to `simulation/out/visual_regression/latest/`

**Manual Inspection**: After running the test, manually inspect screenshots to verify:
- ✅ Wakeboarder (red box) is clearly visible
- ✅ Towers (yellow/gold boxes) are positioned on the loop
- ✅ Smooth motion along the path (no teleporting/jitter)
- ✅ Camera follows the action reliably
- ✅ Scene scale looks consistent

```bash
# View screenshots
ls -lh simulation/out/visual_regression/latest/*.png

# Open first screenshot
python -c "from PIL import Image; Image.open('simulation/out/visual_regression/latest/t_000.00s.png').show()"
```

**Note**: This test runs in windowed mode (not headless) due to Panda3D offscreen rendering limitations.

## Project Structure

```
simulation/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── run_sim.py               # Entry point
├── sim/                     # Main package
│   ├── __init__.py
│   ├── config.py            # Configuration and CLI parsing
│   ├── cable_path.py        # Cable path geometry
│   ├── wakeboarder.py       # Wakeboarder physics
│   ├── logger.py            # CSV logging
│   └── app.py               # Panda3D application
├── tests/                   # Pytest tests
│   ├── __init__.py
│   └── test_simulation.py
└── out/                     # Output directory (created on first run)
    ├── run_*.csv            # Logged data
    └── screenshots/         # Optional screenshots
```

## Known Issues

### Screenshot Capture in Headless Mode

**Issue**: When using Panda3D's offscreen rendering (headless mode), screenshot capture produces black/empty screenshots after the first 1-2 frames.

**Root Cause**: Panda3D's offscreen buffer management appears to have limitations on certain platforms/graphics drivers. The display buffer becomes invalid or inaccessible after initial frames when using `getScreenshot()`.

**Workaround**: Run simulations in windowed mode (without `--headless`) for reliable screenshot capture. For CI environments:
- Use Xvfb or similar virtual display
- Skip screenshot validation in tests
- Accept limited screenshot coverage (first 1-2 frames only) in headless mode

**Attempts to Resolve** (all unsuccessful):
- Using `saveScreenshot()` vs `getScreenshot()`
- Forcing render with `graphicsEngine.renderFrame()`
- Separate screenshot task with different priorities
- Delayed screenshot capture via `doMethodLater()`
- Software framebuffer mode
- Direct texture extraction

## Known Limitations

This is a minimal simulation with several simplifications:

1. **No cable physics**: The cable is not simulated; wakeboarder follows a kinematic target
2. **No water physics**: Water is visual only, no buoyancy or drag forces
3. **Simple motion model**: No rope tension, swing dynamics, or tricks
4. **Fixed path**: Wakeboarder follows the polyline exactly, no lateral deviation
5. **No collisions**: Wakeboarder can't collide with towers or ground
6. **Simplified visualization**: Basic geometry (boxes) for all objects

These limitations are intentional to keep the simulation:
- **Fast**: Runs in real-time or faster
- **Deterministic**: Same inputs always produce same outputs
- **Simple**: Easy to understand and modify
- **Stable**: No complex physics that could diverge

## Use Cases

### Synthetic Data Generation
Generate labeled training data for computer vision models:
```bash
python -m simulation.run_sim --headless --duration 60 --screenshot-every 30
```

### Motion Tracking Validation
Test tracking algorithms with known ground truth:
```bash
python -m simulation.run_sim --headless --duration 30 --speed 8
# Use output CSV as ground truth for position/velocity
```

### CI Regression Testing
Verify simulation stability across code changes:
```bash
pytest simulation/tests/ --tb=short
```

## Troubleshooting

### "No module named panda3d"
Install dependencies: `pip install -r simulation/requirements.txt`

### "Failed to open window"
Either:
- Run with `--headless` flag
- Ensure you have OpenGL support (for visual mode)

### Simulation runs too slow
- Increase `--dt` (e.g., `--dt 0.016` for 60fps equivalent)
- Reduce `--duration`
- Use `--headless` mode

### Non-deterministic results
Check:
- Same `--seed` value
- Same `--dt` value
- Same Panda3D/Bullet versions
- No parallel physics (Bullet uses single-threaded deterministic solver)

## Development

To modify the simulation:

1. **Change cable path**: Edit `CABLE_TOWER_LOCATIONS` in `sim/config.py`
2. **Adjust physics**: Modify gains in `SimConfig` dataclass
3. **Add sensors**: Extend `SimulationLogger` to log additional data
4. **Change visuals**: Modify `_setup_scene()` in `app.py`

## License

Part of the BearVision project. See repository root for license information.
