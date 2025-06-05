"""Rectangular arena implementation."""

import numpy as np

from physbeh.arena.base import BaseArena


class RectangularArena(BaseArena):
    """Rectangular arena implementation.

    Parameters
    ----------
    width : float
        The arena width in `space_units`.
    height : float
        The arena height in `space_units`.
    spatial_units : float
        The spatial units used in the arena.
    px_top_left : numpy.ndarray, optional
        The top left corner of the arena in pixel. Default is ``np.array([0, 0])``.
    px_top_right : numpy.ndarray, optional
        The top right corner of the arena in pixel. Default is ``np.array([0, 1])``.
    px_bottom_left : numpy.ndarray, optional
        The bottom left corner of the arena in pixel. Default is ``np.array([1, 0])``.
    px_bottom_right : numpy.ndarray, optional
        The bottom right corner of the arena in pixel. Default is ``np.array([1, 1])``.
    """

    @property
    def width(self):
        """The arena width in `space_units`.

        Returns
        -------
        float
            The arena width in `space_units`.
        """
        return self._width

    @property
    def height(self):
        """The arena height in `space_units`.

        Returns
        -------
        float
            The arena height in `space_units`.
        """
        return self._height

    @property
    def bottom_left(self):
        """The bottom left corner of the arena in pixel.

        Returns
        -------
        numpy.ndarray
            The bottom left corner of the arena in pixel.
        """
        return self._px_bottom_left

    @property
    def top_left(self):
        """The top left corner of the arena in pixel.

        Returns
        -------
        numpy.ndarray
            The top left corner of the arena in pixel.
        """
        return self._px_top_left

    @property
    def bottom_right(self):
        """The bottom right corner of the arena in pixel.

        Returns
        -------
        numpy.ndarray
            The bottom right corner of the arena in pixel.
        """
        return self._px_bottom_right

    @property
    def top_right(self):
        """The top right corner of the arena in pixel.

        Returns
        -------
        numpy.ndarray
            The top right corner of the arena in pixel.
        """
        return self._px_top_right

    @property
    def spatial_units(self) -> str:
        """The spatial units used in the arena.

        Returns
        -------
        str
            The spatial units used in the arena.
        """
        return self._spatial_units

    @property
    def space_units_per_pixel(self) -> float:
        """Per pixel ratio of the space units in the dataframe.

        Returns
        -------
        float
            The space units per pixel ratio.
        """
        return self.calculate_cm_per_pixel()

    def __init__(
        self,
        width,
        height,
        spatial_units,
        px_top_left=np.array([0, 0]),
        px_top_right=np.array([0, 1]),
        px_bottom_left=np.array([1, 0]),
        px_bottom_right=np.array([1, 1]),
    ) -> None:
        super().__init__()

        self._width = width
        self._height = height
        self._spatial_units = spatial_units
        self._px_bottom_left = px_bottom_left
        self._px_bottom_right = px_bottom_right
        self._px_top_left = px_top_left
        self._px_top_right = px_top_right

        self.origin = self.top_left

    def __repr__(self) -> str:
        """Representation of the arena."""
        norm = max(self.width, self.height)
        width = int(5 * self.width / norm)
        height = int(5 * self.height / norm)

        left_len = max(len(f"{self.top_left} "), len(f"{self.bottom_left} "))

        ascii_representation = f"{self.top_left}"
        ascii_representation += " " * (left_len - len(f"{self.top_left}"))
        ascii_representation += "+" + "-" * width
        ascii_representation += f"+ {self.top_right}\n"

        for _ in range(height):
            ascii_representation += " " * left_len + "|" + " " * width + "|\n"
        ascii_representation += (
            f"{self.bottom_left}"
            + " " * (left_len - len(f"{self.bottom_left}"))
            + "+"
            + "-" * width
            + f"+ {self.bottom_right}\n"
        )
        return (
            f"{self.__class__.__name__} "
            f"({self.space_units_per_pixel:.3f} {self.spatial_units}/px)"
            f"\n{ascii_representation}"
        )

    def calculate_cm_per_pixel(self) -> float:
        """Helper function to calculate the cm per pixel ratio in a rectangle.

        Returns
        -------
        float
            The cm per pixel ratio.
        """

        def _distance(p1, p2):
            return np.sqrt(np.sum((p1 - p2) ** 2))

        width_estimates = [
            _distance(self.bottom_left, self.bottom_right),
            _distance(self.top_left, self.top_right),
        ]
        height_estimates = [
            _distance(self.bottom_left, self.top_left),
            _distance(self.bottom_right, self.top_right),
        ]
        diag_estimates = [
            _distance(self.bottom_left, self.top_right),
            _distance(self.bottom_right, self.top_left),
        ]

        real_diag_cm = np.sqrt(self.width**2 + self.height**2)
        px_ratio_width = self.width / np.mean(width_estimates)
        px_ratio_height = self.height / np.mean(height_estimates)
        px_ratio_diag = real_diag_cm / np.mean(diag_estimates)

        return np.mean((px_ratio_diag, px_ratio_height, px_ratio_width))
