"""Main Panda3D simulation application."""

import sys
from pathlib import Path

import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.bullet import BulletDebugNode, BulletWorld
from panda3d.core import (
    AmbientLight,
    DirectionalLight,
    GeomNode,
    loadPrcFileData,
    Vec3,
    Vec4,
)

from .cable_path import CablePath
from .config import CABLE_TOWER_LOCATIONS, SimConfig
from .logger import SimulationLogger
from .wakeboarder import Wakeboarder


class WakeboardSimulation(ShowBase):
    """Main simulation application using Panda3D."""

    def __init__(self, sim_config: SimConfig):
        """Initialize simulation.

        Args:
            sim_config: Simulation configuration
        """
        # Configure Panda3D for headless mode if requested
        if sim_config.headless:
            loadPrcFileData('', 'window-type offscreen')
            loadPrcFileData('', 'audio-library-name null')

        # Initialize ShowBase (must be before setting self.sim_config)
        super().__init__()

        # Store our config (after ShowBase init to avoid shadowing self.config)
        self.sim_config = sim_config

        # Disable default camera control
        self.disableMouse()

        # Setup scene
        self._setup_scene()

        # Setup physics
        self._setup_physics()

        # Setup logger
        self.logger = SimulationLogger(sim_config.outdir)
        self.logger.start(sim_config.dt)

        # Simulation state
        self.sim_time = 0.0
        self.frame_count = 0
        self.running = True
        self.next_screenshot_index = 0  # for time-based screenshots

        # Start simulation loop
        self.taskMgr.add(self._update_task, "update_task")

    def _setup_scene(self) -> None:
        """Create the visual scene."""
        # Lighting (much brighter for better visibility)
        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.8, 0.8, 0.8, 1))  # Much brighter ambient
        ambient_np = self.render.attachNewNode(ambient)
        self.render.setLight(ambient_np)

        sun = DirectionalLight("sun")
        sun.setColor(Vec4(1.5, 1.5, 1.4, 1))  # Brighter sun (can go >1.0)
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(45, -45, 0)  # Less steep angle for better illumination
        self.render.setLight(sun_np)

        # Water plane (simple visual-only plane)
        from panda3d.core import CardMaker
        cm = CardMaker("water")
        cm.setFrame(-500, 500, -500, 500)
        water = self.render.attachNewNode(cm.generate())
        water.setPos(0, 0, 0)
        water.setP(-90)  # rotate to XY plane
        water.setColor(0.1, 0.3, 0.5, 1)

        # Create cable path and visualize towers
        self.cable_path = CablePath(CABLE_TOWER_LOCATIONS)
        self._create_towers()

        # Sky color
        self.setBackgroundColor(0.53, 0.8, 0.92)

    def _create_towers(self) -> None:
        """Create visual representations of cable towers."""
        from panda3d.core import GeomNode

        for i, (x, y) in enumerate(CABLE_TOWER_LOCATIONS):
            # Simple box tower (taller and more visible)
            tower = self.loader.loadModel("models/box")
            tower.setScale(4, 4, 15)  # 4m x 4m x 15m tall tower
            tower.setPos(x, y, 7.5)  # base at z=0, extends to z=15
            tower.setColor(1, 0.8, 0, 1)  # bright yellow/gold
            tower.setLightOff()  # Self-illuminated, always visible
            tower.reparentTo(self.render)

    def _setup_physics(self) -> None:
        """Setup Bullet physics world."""
        # Create physics world
        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -9.81))

        # Create wakeboarder
        self.wakeboarder = Wakeboarder(
            self.sim_config,
            self.cable_path,
            start_s=0.0
        )

        # Add to physics world
        self.world.attachRigidBody(self.wakeboarder.node)

        # Create visual node for wakeboarder
        self.wakeboarder_np = self.render.attachNewNode(
            self.wakeboarder.node
        )

        # Create simple box visual (VERY large and bright so it's visible)
        box_visual = self.loader.loadModel("models/box")
        box_visual.setScale(5.0, 5.0, 3.0)  # HUGE: 5m x 5m x 3m
        box_visual.setColor(1, 0, 0, 1)  # pure bright red
        # Make it self-illuminated (emissive) so it's always visible
        box_visual.setLightOff()  # Don't let lighting affect it
        box_visual.reparentTo(self.wakeboarder_np)

        # Setup camera to follow wakeboarder
        self._setup_camera()

    def _setup_camera(self) -> None:
        """Setup camera to follow the wakeboarder."""
        # Third-person view offset (zoomed out further to see more of the scene)
        self.camera_offset = Vec3(-40, 0, 20)  # 40m back, 20m up for wider view
        self.camera.setPos(self.camera_offset)
        self.camera.lookAt(0, 0, 0)

        # Set wider field of view to see more of the scene
        lens = self.cam.node().getLens()
        lens.setFov(90)  # 90 degree horizontal FOV (default is 60)

    def _update_camera(self) -> None:
        """Update camera to follow wakeboarder."""
        wb_pos = self.wakeboarder_np.getPos()
        cam_pos = wb_pos + self.camera_offset
        self.camera.setPos(cam_pos)
        self.camera.lookAt(wb_pos)

    def _update_task(self, task) -> int:
        """Main update task called each frame.

        Args:
            task: Panda3D task object

        Returns:
            Task.cont to continue or Task.done to stop
        """
        # Check if simulation should stop
        if self.sim_time >= self.sim_config.duration:
            self._shutdown()
            return task.done

        # Update wakeboarder forces
        self.wakeboarder.update(self.sim_config.dt)

        # Step physics
        self.world.doPhysics(
            self.sim_config.dt,
            self.sim_config.bullet_substeps,
            1.0 / 240.0  # fixed internal substep
        )

        # Update camera
        self._update_camera()

        # Log data
        self.logger.log_frame(
            time=self.sim_time,
            position=self.wakeboarder.get_position(),
            velocity=self.wakeboarder.get_velocity(),
            path_s=self.wakeboarder.get_path_distance(),
            segment_index=self.wakeboarder.get_segment_index(),
        )

        # Advance simulation time
        self.sim_time += self.sim_config.dt
        self.frame_count += 1

        # Schedule screenshot if requested (time-based)
        # Use doMethodLater to capture AFTER rendering completes
        if self.sim_config.screenshot_every > 0:
            # Calculate the target time for the next screenshot
            next_target_time = self.next_screenshot_index * self.sim_config.screenshot_every
            # Check if we've reached or passed the next screenshot time
            if self.sim_time >= next_target_time and next_target_time <= self.sim_config.duration:
                # Schedule screenshot to capture after next render
                # Use task manager to schedule with small delay
                filepath = self._get_screenshot_filepath(next_target_time)

                def take_screenshot_deferred(task, fp=filepath):
                    capture_screenshot(self, str(fp))
                    return task.done

                # Schedule for next frame (use sim dt as the delay)
                self.taskMgr.doMethodLater(
                    self.sim_config.dt * 1.5,  # 1.5x dt to ensure render completes
                    take_screenshot_deferred,
                    f"screenshot_{next_target_time}"
                )
                self.next_screenshot_index += 1

        return task.cont

    def _get_screenshot_filepath(self, target_time: float) -> Path:
        """Get the filepath for a screenshot.

        Args:
            target_time: The target simulation time

        Returns:
            Path to save the screenshot
        """
        # Use custom screenshot_dir if provided, otherwise use outdir/screenshots
        if self.sim_config.screenshot_dir is not None:
            screenshot_dir = self.sim_config.screenshot_dir
        else:
            screenshot_dir = self.sim_config.outdir / "screenshots"

        screenshot_dir.mkdir(parents=True, exist_ok=True)

        # Use target time for filename
        return screenshot_dir / f"t_{target_time:06.2f}s.png"

    def _save_screenshot(self, target_time: float = None) -> None:
        """Save a screenshot of the current frame.

        Args:
            target_time: The target simulation time for this screenshot.
                        If None, uses self.sim_time.
        """
        # Use custom screenshot_dir if provided, otherwise use outdir/screenshots
        if self.sim_config.screenshot_dir is not None:
            screenshot_dir = self.sim_config.screenshot_dir
        else:
            screenshot_dir = self.sim_config.outdir / "screenshots"

        screenshot_dir.mkdir(parents=True, exist_ok=True)

        # Use target time for filename if provided, otherwise use actual sim_time
        time_for_filename = target_time if target_time is not None else self.sim_time

        # Use time-based filename for consistency
        filename = screenshot_dir / f"t_{time_for_filename:06.2f}s.png"
        capture_screenshot(self, str(filename))

    def _shutdown(self) -> None:
        """Shutdown simulation and save results."""
        print(f"\nSimulation complete: {self.sim_time:.2f}s")

        # Wait for pending screenshot tasks to complete
        max_wait_frames = 30  # Increased wait time
        for i in range(max_wait_frames):
            # Check if there are any pending screenshot tasks
            screenshot_tasks = [t for t in self.taskMgr.getAllTasks() if t.getName().startswith("screenshot_")]
            if not screenshot_tasks:
                break
            # Process one more frame to let tasks complete
            self.taskMgr.step()
            # Also force a render
            self.graphicsEngine.renderFrame()

        # Close logger
        output_file = self.logger.stop()
        print(f"Results saved to: {output_file}")

        # Exit application
        sys.exit(0)


def capture_screenshot(base: ShowBase, filepath: str) -> None:
    """Capture a screenshot from the Panda3D window.

    Works in both windowed and offscreen (headless) mode.

    Args:
        base: Panda3D ShowBase instance
        filepath: Full path where to save the PNG file (including extension)
    """
    from panda3d.core import Filename, PNMImage

    # Create a PNMImage to store the screenshot
    img = PNMImage()

    # Get screenshot from window
    if base.win.getScreenshot(img):
        # Save to file
        filename = Filename.fromOsSpecific(filepath)
        img.write(filename)


def run_simulation(config: SimConfig) -> None:
    """Run the simulation with the given configuration.

    Args:
        config: Simulation configuration
    """
    # Set random seed for reproducibility
    np.random.seed(config.seed)

    # Create and run simulation
    app = WakeboardSimulation(config)
    app.run()
