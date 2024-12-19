"""Helper package to visualise and analyse DeepLabCut data."""

from importlib import metadata

__version__ = metadata.version("physbeh")

from . import arena, io, plotting, tracking, utils
from .io import load_tracking

__all__ = ["load_tracking", "plotting", "tracking", "utils", "io", "arena"]
