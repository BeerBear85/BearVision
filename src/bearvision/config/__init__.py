"""Strict, independently versioned configuration support."""

from .models import AssignmentConfig, EdgeConfig, load_edge_config

__all__ = ["AssignmentConfig", "EdgeConfig", "load_edge_config"]
