"""Core primitives (plugin registry, shared base classes)."""

from .plugin_api import BaseDynamics, registry

__all__ = ["BaseDynamics", "registry"]
