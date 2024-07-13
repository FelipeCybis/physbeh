"""Miscellaneous helper functions for tracking_physmed."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from matplotlib import colors


def get_line_collection(
    x_array: npt.NDArray | list[float],
    y_array: npt.NDArray | list[float],
    index: list[bool] | np.ndarray[Any, np.dtype[bool]],
) -> Sequence[npt.NDArray]:
    """Get collection of arrays for each segment of `x_array` and `y_array`.

    Returns this collection where index == True.

    Parameters
    ----------
    x_array : numpy.ndarray or list of float
        The ``x`` coordinates for the line collection.
    y_array : numpy.ndarray or list of float
        The ``y`` coordinates for the line collection.
    index : list or numpy.ndarray of bool
        The indices to keep in the line collection.

    Returns
    -------
    numpy.ndarray
        The array representing the line collection.
    """
    listified_x_array = [x_array] if not isinstance(x_array, list) else x_array
    listified_y_array = [y_array] if not isinstance(y_array, list) else y_array
    listified_index = [index] if not isinstance(index, list) else index

    segments = []
    segment_index = []
    for x, y, idx in zip(listified_x_array, listified_y_array, listified_index):  # type: ignore[call-overload]
        x, y = np.squeeze(x), np.squeeze(y)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments.append(np.concatenate([points[:-1], points[1:]], axis=1))
        segment_index.append(np.logical_and(idx[1:], idx[:-1]))

    segments_array = np.concatenate(segments, axis=0)
    segment_index = np.concatenate(segment_index, axis=0)
    return segments_array[segment_index]


def get_gaussian_value(x, u=0, sigma=10) -> npt.NDArray:
    r"""Return the value of `x` in a gaussian curve with mean ``u`` and std ``sigma``.

    The function is written as:

    .. math::

       \frac{1}{\sigma \sqrt{2 \pi}} e^{\frac{-(x - u)^2}{2 \sigma^2}}

    Parameters
    ----------
    x : float or numpy.ndarray
        The ``x`` coordinate.
    u : int, optional
        The mean value of the gaussian function. Default is ``0``.
    sigma : int, optional
        The sigma value of the gaussion function. Default ``10``.

    Returns
    -------
    numpy.ndarray
        The gaussian function value for `x`.
    """
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.e ** (-((x - u) ** 2) / (2 * sigma**2))


def custom_sigmoid(
    x: float | npt.NDArray, a: float = 0.2, b: float = 70.0
) -> npt.NDArray:
    """Get ``x`` coordinate value in a sigmoid function.

    The sigmoid is written as:

    sig(x) = 1 / (1 + np.e ** (-a * (x - b)))

    Parameters
    ----------
    x : float or numpy.ndarray
        The ``x`` coordinate.
    a : float, optional
        The ``a`` argument for the sigmoid using the ``x`` coordinate. Default is
        ``0.2``.
    b : float, optional
        The ``b`` argument for the sigmoid using the ``x`` coordinate. Default is
        ``70``.

    Returns
    -------
    numpy.ndarray
        The sigmoid activation value for the given coordinates.
    """
    return 1 / (1 + np.e ** (-a * (x - b)))


def custom_2d_sigmoid(
    x: float | npt.NDArray,
    y: float | npt.NDArray,
    ax: float = 0.2,
    bx: float = 70.0,
    ay: float = 0.2,
    by: float = 70.0,
) -> npt.NDArray:
    """Get ``x, y`` coordinate value in a 2D sigmoid function.

    The sigmoid is written as:

    sig(x) = 1 / (1 + np.e ** (-a * (x - b)))

    The 2D sigmoid is then calculated using ``sig(x) * sig(y)``.

    Parameters
    ----------
    x : float or numpy.ndarray
        The ``x`` coordinate.
    y : float or numpy.ndarray
        The ``y`` coordinate.
    ax : float, optional
        The ``a`` argument for the sigmoid using the ``x`` coordinate. Default is
        ``0.2``.
    bx : float, optional
        The ``b`` argument for the sigmoid using the ``x`` coordinate. Default is
        ``70``.
    ay : float, optional
        The ``a`` argument for the sigmoid using the ``y`` coordinate. Default is
        ``0.2``.
    by : float, optional
        The ``a`` argument for the sigmoid using the ``y`` coordinate. Default is
        ``70``.

    Returns
    -------
    numpy.ndarray
        The sigmoid activation value for the given coordinates.
    """
    return custom_sigmoid(x, ax, bx) * custom_sigmoid(y, ay, by)


def _plot_color_wheel(ax, cmap):
    # Define colormap normalization for 0 to 2*pi
    norm_2 = colors.Normalize(0, 2 * np.pi)

    # Plot a color mesh on the polar plot
    # with the color set by the angle
    n = 360  # the number of secants for the mesh
    t = np.linspace(0, 2 * np.pi, n)  # theta values
    r = np.linspace(0.6, 1, 2)  # radius values change 0.6 to 0 for full circle
    rg, tg = np.meshgrid(r, t)  # create a r,theta meshgrid
    c = tg  # define color values as theta value
    ax.pcolormesh(
        t, r, c.T, norm=norm_2, cmap=cmap, shading="auto"
    )  # plot the colormesh on axis with colormap
    ax.set(
        yticklabels=[],
        xticks=[0, np.pi / 2, np.pi, np.pi * 3 / 2],
        xticklabels=["E", "N", "W", "S"],
    )
    ax.tick_params(pad=4, labelsize=14)  # cosmetic changes to tick labels
    ax.spines["polar"].set_visible(False)
    ax.set_title("Head direction color wheel", y=1.0, pad=30)


class BlitManager:
    """Manager to use matplotlib blitting only in explicitly added artists.

    Parameters
    ----------
    canvas : FigureCanvasAgg
        The canvas to work with, this only works for sub-classes of the Agg
        canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
        `~FigureCanvasAgg.restore_region` methods.

    animated_artists : Iterable[Artist]
        List of the artists to manage.
    """

    def __init__(self, canvas, animated_artists=()):
        self.canvas = canvas
        self._bg = None
        self._artists = []

        for a in animated_artists:
            self.add_artist(a)
        # grab the background on every draw
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, event):  # numpydoc ignore=PR01
        """Callback to register with 'draw_event'."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def add_artist(self, art):
        """Add an artist to be managed.

        Parameters
        ----------
        art : Artist
            The artist to be added.  Will be set to 'animated' (just
            to be safe).  *art* must be in the figure associated with
            the canvas this class is managing.
        """
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)

    def _draw_animated(self):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self):
        """Update the screen with animated artists."""
        cv = self.canvas
        fig = cv.figure
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            cv.restore_region(self._bg)
            # draw all of the animated artists
            self._draw_animated()
            # update the GUI state
            cv.blit(fig.bbox)
        # let the GUI event loop process anything it has to do
        cv.flush_events()
