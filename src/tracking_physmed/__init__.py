"""Helper package to visualise and analyse DeepLabCut data."""

from importlib import metadata

__version__ = metadata.version("tracking-physmed")

from . import io, plotting, tracking, utils
from .io import load_tracking

__all__ = ["load_tracking", "plotting", "tracking", "utils", "io"]
