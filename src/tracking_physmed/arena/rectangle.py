import numpy as np

from tracking_physmed.arena.base import BaseArena


class RectangularArena(BaseArena):
    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def bottom_left(self):
        return self._px_bottom_left

    @property
    def top_left(self):
        return self._px_top_left

    @property
    def bottom_right(self):
        return self._px_bottom_right

    @property
    def top_right(self):
        return self._px_top_right

    @property
    def spatial_units(self) -> float:
        return self._spatial_units

    @property
    def space_units_per_pixel(self) -> float:
        """Per pixel ratio of the space units in the dataframe.

        Returns
        -------
        float
            The space units per pixel ratio.
        """
        return self.calculate_rectangle_per_pixel()

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

    def calculate_rectangle_per_pixel(self) -> float:
        """Helper function to calculate the cm per pixel ratio in a rectangle."""

        def distance(p1, p2):
            return np.sqrt(np.sum((p1 - p2) ** 2))

        width_estimates = [
            distance(self.bottom_left, self.bottom_right),
            distance(self.top_left, self.top_right),
        ]
        height_estimates = [
            distance(self.bottom_left, self.top_left),
            distance(self.bottom_right, self.top_right),
        ]
        diag_estimates = [
            distance(self.bottom_left, self.top_right),
            distance(self.bottom_right, self.top_left),
        ]

        real_diag_cm = np.sqrt(self.width**2 + self.height**2)
        px_ratio_width = self.width / np.mean(width_estimates)
        px_ratio_height = self.height / np.mean(height_estimates)
        px_ratio_diag = real_diag_cm / np.mean(diag_estimates)

        return np.mean((px_ratio_diag, px_ratio_height, px_ratio_width))
