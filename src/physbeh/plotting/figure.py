"""Figure object for behavior plotting."""

from matplotlib.collections import LineCollection
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure, SubFigure


class BehFigure:
    """Figure object for behavior plotting.

    This is a wrapper around the matplotlib figure object that allows more flexibility
    in animations and easily keeping track of some properties.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The matplotlib figure object.
    """

    @property
    def cbar(self) -> Colorbar | None:  # numpydoc ignore=PR02
        """Get the colorbar of the figure.

        Parameters
        ----------
        value : matplotlib.colorbar.Colorbar
            The colorbar object.

        Returns
        -------
        matplotlib.colorbar.Colorbar or None
            The colorbar object (or ``None`` if not set).
        """
        return self._cbar

    @cbar.setter
    def cbar(self, value: Colorbar):  # numpydoc ignore=GL08
        self._cbar = value

    @property
    def lc(self) -> LineCollection:  # numpydoc ignore=PR02
        """Get the line collection of the figure.

        Parameters
        ----------
        value : matplotlib.collections.LineCollection
            The line collection object.

        Returns
        -------
        matplotlib.collections.LineCollection
            The line collection object.
        """
        return self._lc

    @lc.setter
    def lc(self, value: LineCollection):  # numpydoc ignore=GL08
        self._lc = value

    @property
    def figure(self) -> Figure:
        """Get the underlying matplotlib figure.

        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib figure object.
        """
        return self._figure.figure

    def __init__(self, figure: Figure | SubFigure):
        self._figure = figure

    def show(self):
        """Show the figure."""
        self._figure.get_figure(root=True).show()
