"""Base implementation to be subclassed by different types of arenas."""

import numpy as np


class BaseArena:
    """Base implementation to be subclassed by different types of arenas.

    The base implementation only provides the top left corner of the arena, the space
    units per pixel ratio and an utility function to compute the extent of arena given
    the total width and height of the video frame.
    """

    @property
    def origin(self):
        """The origin of the arena in pixel coordinates.

        Returns
        -------
        numpy.ndarray
            The origin coordinate of the arena.
        """
        return np.array([0, 0])

    @property
    def space_units_per_pixel(self) -> float:
        """Per pixel ratio of the space units in the dataframe.

        Returns
        -------
        float
            The space units per pixel ratio.
        """
        return 1

    def get_extent(self, px_total_width: float, px_total_height: float):
        """Get the extent of the arena given the total pixel width and height.

        Parameters
        ----------
        px_total_width : float
            The total width of the video frame in pixel.
        px_total_height : float
            The total height of the video frame in pixel.

        Returns
        -------
        numpy.ndarray
            The extent of the arena, typically useful to set the extent of when plotting
            the video frame. The extent is in the form of (left, right, bottom, top).
        """
        original_extent = np.array(
            (-0.5, px_total_width - 0.5, px_total_height - 0.5, -0.5)
        )
        original_extent[:2] -= self.origin[0]
        original_extent[2:] -= self.origin[1]

        return original_extent * self.space_units_per_pixel
