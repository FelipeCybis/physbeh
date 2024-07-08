import numpy as np


class BaseArena:
    @property
    def top_left(self):
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

    def get_extent(self, px_total_width, px_total_height):
        original_extent = np.array(
            (-0.5, px_total_width - 0.5, px_total_height - 0.5, -0.5)
        )
        original_extent[:2] -= self.top_left[0]
        original_extent[2:] -= self.top_left[1]

        return original_extent * self.space_units_per_pixel
