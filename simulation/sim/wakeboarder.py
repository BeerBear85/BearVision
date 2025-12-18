"""Wakeboarder rigid body physics."""

import numpy as np
from panda3d.bullet import BulletBoxShape, BulletRigidBodyNode
from panda3d.core import Vec3

from .cable_path import CablePath
from .config import SimConfig


class Wakeboarder:
    """Represents the wakeboarder as a Bullet rigid body.

    The wakeboarder is pulled along the cable path using PD control
    to follow a target position. Vertical position is constrained
    to stay near the water surface.
    """

    def __init__(
        self,
        config: SimConfig,
        cable_path: CablePath,
        start_s: float = 0.0,
    ):
        """Initialize wakeboarder.

        Args:
            config: Simulation configuration
            cable_path: Cable path to follow
            start_s: Starting distance along cable path
        """
        self.config = config
        self.cable_path = cable_path
        self.path_s = start_s  # current distance along cable

        # Create Bullet rigid body
        shape = BulletBoxShape(Vec3(0.5, 0.3, 0.2))  # simple box shape
        self.node = BulletRigidBodyNode("wakeboarder")
        self.node.setMass(config.wakeboarder_mass)
        self.node.addShape(shape)

        # Set initial position
        start_pos_2d, _ = cable_path.get_position_at_distance(start_s)
        from panda3d.core import TransformState
        transform = TransformState.makePos(Vec3(
            start_pos_2d[0],
            start_pos_2d[1],
            config.z_target
        ))
        self.node.setTransform(transform)

        # Disable rotation for simplicity (keep upright)
        self.node.setAngularFactor(Vec3(0, 0, 0))

        # Set initial velocity in the direction of the cable
        tangent = cable_path.get_tangent_at_distance(start_s)
        initial_vel = Vec3(
            tangent[0] * config.speed,
            tangent[1] * config.speed,
            0
        )
        self.node.setLinearVelocity(initial_vel)

    def update(self, dt: float) -> None:
        """Update wakeboarder physics.

        Applies forces to pull wakeboarder along cable path and
        maintain vertical position.

        Args:
            dt: Time step in seconds
        """
        # Advance target position along cable
        self.path_s += self.config.speed * dt
        self.path_s = self.path_s % self.cable_path.total_length

        # Get target position on cable
        target_pos_2d, _ = self.cable_path.get_position_at_distance(
            self.path_s
        )

        # Current position and velocity
        current_pos = self.node.getTransform().getPos()
        current_vel = self.node.getLinearVelocity()

        # Horizontal pulling force (PD controller)
        # Error in XY plane
        error_x = target_pos_2d[0] - current_pos.x
        error_y = target_pos_2d[1] - current_pos.y

        # Get tangent direction for velocity error
        tangent = self.cable_path.get_tangent_at_distance(self.path_s)
        target_vel_x = tangent[0] * self.config.speed
        target_vel_y = tangent[1] * self.config.speed

        vel_error_x = target_vel_x - current_vel.x
        vel_error_y = target_vel_y - current_vel.y

        # PD control force
        force_x = (
            self.config.pull_force_gain * error_x +
            self.config.pull_damping * vel_error_x
        )
        force_y = (
            self.config.pull_force_gain * error_y +
            self.config.pull_damping * vel_error_y
        )

        # Vertical spring force to keep near water surface
        z_error = self.config.z_target - current_pos.z
        z_vel_error = -current_vel.z  # target z velocity is 0

        force_z = (
            self.config.z_spring * z_error +
            self.config.z_damping * z_vel_error
        )

        # Apply total force
        total_force = Vec3(force_x, force_y, force_z)
        self.node.applyCentralForce(total_force)

    def get_position(self) -> np.ndarray:
        """Get current position.

        Returns:
            Position as [x, y, z] in meters
        """
        pos = self.node.getTransform().getPos()
        return np.array([pos.x, pos.y, pos.z], dtype=np.float64)

    def get_velocity(self) -> np.ndarray:
        """Get current velocity.

        Returns:
            Velocity as [vx, vy, vz] in m/s
        """
        vel = self.node.getLinearVelocity()
        return np.array([vel.x, vel.y, vel.z], dtype=np.float64)

    def get_path_distance(self) -> float:
        """Get current distance along cable path.

        Returns:
            Distance in meters
        """
        return self.path_s

    def get_segment_index(self) -> int:
        """Get current cable segment index.

        Returns:
            Segment index (0 to n_segments-1)
        """
        return self.cable_path.get_segment_index_at_distance(self.path_s)
