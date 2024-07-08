"""Helper package to visualise and analyse DeepLabCut data"""

from importlib import metadata

__version__ = metadata.version("tracking-physmed")

from .io import load_tracking
from .plotting import plotting
from .tracking import tracking

__all__ = ["load_tracking", "plotting", "tracking"]
