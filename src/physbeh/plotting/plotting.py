"""Useful plotting functions for Tracking objects."""

import warnings
from collections.abc import Sequence
from typing import Any, Literal, cast, overload

import matplotlib
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import colormaps as mpl_cm
from matplotlib import colors
from matplotlib.collections import LineCollection
from matplotlib.typing import ColorType
from mpl_toolkits.axes_grid1 import make_axes_locatable

from physbeh.plotting.animate2d_decorator import Animate_plot2D, anim2d_decorator
from physbeh.plotting.animate_decorator import Animate_plot, anim_decorator
from physbeh.plotting.figure import BehFigure
from physbeh.tracking import Tracking
from physbeh.utils import _plot_color_wheel, get_line_collection


def _listify_bodyparts(trk, bodyparts):
    if isinstance(bodyparts, str):
        if bodyparts == "all":
            bodyparts = trk.labels
        else:
            bodyparts = [bodyparts]
    return bodyparts


def get_label_color(
    trk: Tracking, bodypart: str, cmap_name: str = "plasma"
) -> tuple[float, float, float, float]:
    """Helper function to get the color of a bodypart label.

    Parameters
    ----------
    trk : Tracking
        The tracking object.
    bodypart : str
        The desired bodypart.
    cmap_name : str, optional
        The matplotlib colormap name. Default is ``"plasma"``.

    Returns
    -------
    tuple of RGBA values
        Matplotlib color tuple corresponding to the given bodypart.
    """
    cmap = mpl_cm.get_cmap(cmap_name).resampled(len(trk.labels))
    return cmap(trk.labels.index(bodypart))


def _check_ax_and_fig(
    ax: matplotlib.axes.Axes | None,
    fig: matplotlib.figure.Figure | matplotlib.figure.SubFigure | BehFigure | None,
    **fig_kwargs,
) -> tuple[matplotlib.axes.Axes, BehFigure]:
    # Wrap figure in BehFigure if matplotlib figure
    if isinstance(fig, matplotlib.figure.Figure | matplotlib.figure.SubFigure):
        fig = BehFigure(fig)
    # Add axes to figure if not provided
    if ax is None:
        if fig is None:
            fig = BehFigure(plt.figure(**fig_kwargs))
        ax = fig.figure.add_subplot(111)
    # If figure still None, axes was provided, take figure from axes
    if fig is None:
        fig = BehFigure(ax.figure)
    assert fig.figure == ax.figure.figure, (
        f"Axes and figure must be from the same object, but {fig.figure} != {ax.figure}"
    )
    return ax, fig


def plot_array(
    array: npt.NDArray | list[npt.NDArray],
    time_array: npt.NDArray | list[npt.NDArray] | None = None,
    index: list[bool] | npt.NDArray | list[npt.NDArray] | None = None,
    slice_lines: slice = slice(None, None, None),
    trk: Tracking | None = None,
    only_running_bouts: bool = False,
    linewidths: float = 2.0,
    label: str = "",
    color: ColorType = (0.5, 0.5, 0.5, 1.0),
    cmap: str | colors.Colormap | None = None,
    norm: colors.Normalize | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    cbar_label: str = "",
    line_collection_array: npt.NDArray | None = None,
    alpha: float = 1.0,
    axes: matplotlib.axes.Axes | None = None,
    figure: matplotlib.figure.Figure | None = None,
    figsize: tuple[float, float] = (12, 6),
    dpi: float = 100,
    set_axes: bool = True,
    **ax_kwargs,
) -> tuple[BehFigure, matplotlib.axes.Axes, npt.NDArray]:
    """Plot an array by transforming it into line collections.

    Parameters
    ----------
    array : numpy.ndarray
        The array to plot.
    time_array : numpy.ndarray, optional
        The time array. Default is ``None``.
    index : numpy.ndarray, optional
        The index array to mask what is going to be plotted or not in `array`. Default
        is ``None`` to plot everything.
    slice_lines : slice, optional
        A slice object that will define how the line collection will be plotted. Default
        is ``slice(None, None, None)`` to plot everything. i.e., ``slice(0, -1, 10)``
        will plot every 10th line.
    trk : Tracking, optional
        A tracking object. This argument is here only for compatibility with the
        animation decorator in case an animation using the tracking video wants to be
        produced using a custom array. Default is ``None``.
    only_running_bouts : bool, optional
        Whether to plot only the running bouts. Default is ``False``.
    linewidths : float, optional
        The linewidth of the line collection. Default is ``2.0``.
    label : str, optional
        The label of the line collection. This is what is going to appear in the figure
        legend. Default is ``""``.
    color : color, optional
        Tuple of RGB(A) values for color of the line collection, if not using
        `head_direction` or any `color_collection_array`. Default is ``(0.5, 0.5, 0.5,
        1.0)``.
    cmap : str, optional
        The colormap to use for the line collection. Mutually exclusive with `color`.
        Takes precedence over `color` if set to anything different than ``None``.
        Default is ``None``.
    norm : matplotlib.colors.Normalize, optional
        The normalization of the colormap. Default is ``None`` to be set to
        `matplotlib.colors.Normalize`.
    vmin, vmax : float or None, optional
        When using no explicit `norm`, `vmin` and `vmax` define the data range that the
        colormap covers. If ``None``, the colormap covers the complete value range of
        the displayed slices. It is an error to use `vmin`/`vmax` when a `norm` instance
        is given, but using a ``str`` `norm` name together with `vmin`/`vmax` is
        accepted. Data range that the colormap covers. Default is ``None``.
    colorbar : bool, optional
        Whether to plot the colorbar. Only possible if `cmap` is not ``None``. Default
        is ``True``.
    cbar_label : str, optional
        Label of the colorbar. Default is ``""``.
    line_collection_array : numpy.ndarray, optional
        The array of values to be used for color mapping the line collection. Default is
        ``None``.
    alpha : float, optional
        The alpha value of the line collection. Default is ``1.0``.
    axes : matplotlib.axes.Axes, optional
        If ``None``, new axes is created in `figure`. Default is ``None``.
    figure : matplotlib.figure.Figure, optional
        If ``None``, new figure is created. Default is ``None``.
    figsize : tuple, optional
        Figure size, if ``figure=None``. Default is ``(12,6)``.
    dpi : float, optional
        Dots per inch of the figure, if ``figure=None``. Default is ``100``.
    set_axes : bool, optional
        Whether to set the axes properties. Default is ``True``.
    **ax_kwargs
        Keywords to pass to ``ax.set(**ax_kwargs)``.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The matplotlib figure object.
    axes : matplotlib.axes.Axes
        The matplotlib axes object.
    """
    axes, behfigure = _check_ax_and_fig(axes, figure, figsize=figsize, dpi=dpi)

    if time_array is None:
        time_array = np.arange(len(array))
    if index is None:
        index = np.ones(len(array), dtype=bool)

    lines = get_line_collection(time_array, array, index)
    lc_kwargs: dict[str, Any] = {}
    if cmap is not None:
        lc_kwargs["cmap"] = cmap
        lc_kwargs["norm"] = colors.Normalize() if norm is None else norm
    else:
        lc_kwargs["color"] = color

    lc_kwargs["array"] = lines[:, 0, 1]
    if line_collection_array is not None:
        if line_collection_array.shape[0] != lines.shape[0]:
            warnings.warn(
                "\n`line_collection_array` have different length than the line\n"
                "collections of arrays. This will very likely give a wrong colormap\n"
                "mapping.\nA possible solution could be using\n"
                "`line_collection_array[index]` with the same index array used for the"
                "\nline collection being plotted or, if `only_running_bouts` is set\n"
                "to ``True``, using the utility function `get_line_collection` to get\n"
                "the correct array with:\n\n"
                ">> lines = get_line_collection(time, array, index)\n"
                ">> plot_array(..., line_collection_array=lines[:, 0, 1])."
            )
        lc_kwargs["array"] = line_collection_array

    if vmin is None:
        vmin = np.nanmin(lc_kwargs["array"])
    if vmax is None:
        vmax = np.nanmax(lc_kwargs["array"])

    lc = LineCollection(
        lines[slice_lines],  # type: ignore
        label=label,
        linewidths=linewidths,
        alpha=alpha,
        **lc_kwargs,
    )
    behfigure.lc = lc
    axes.add_collection(lc)
    if only_running_bouts:
        time_array = np.concatenate(time_array)
        array = np.concatenate(array)
        index = np.concatenate(index)

    axes.autoscale()
    lc.set_clim(vmin, vmax)

    if colorbar and cmap is not None:
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="4%", pad=0.1)
        cbar = behfigure.figure.colorbar(lc, cax=cax)
        cbar.set_label(cbar_label)
        behfigure.cbar = cbar

    if set_axes:
        keys = list(ax_kwargs.keys())
        legend_kwargs = {
            key.replace("legend__", ""): ax_kwargs.pop(key)
            for key in keys
            if key.startswith("legend__")
        }
        grid_kwargs = {
            key.replace("grid__", ""): ax_kwargs.pop(key)
            for key in keys
            if key.startswith("grid__")
        }
        axes.set(**ax_kwargs)
        if label:
            axes.legend(**legend_kwargs)
        axes.grid(**grid_kwargs)

    return behfigure, axes, lines


@overload
def plot_speed(  # numpydoc ignore=GL08
    trk: Tracking,
    bodypart: str = "body",
    *,
    speed_axis: Literal["x", "y", "xy"] = "xy",
    euclidean: bool = False,
    smooth: bool = True,
    speed_cutout: int | float = 0,
    only_running_bouts: bool = False,
    plot_only_running_bouts: bool = True,
    color: tuple[float, float, float, float] | None = None,
    alpha: float = 1.0,
    axes: matplotlib.axes.Axes | None = None,
    figure: matplotlib.figure.Figure | None = None,
    figsize: tuple[float, float] = (8, 4),
    animate: Literal[False] = False,
    **ax_kwargs,
) -> tuple[BehFigure, matplotlib.axes.Axes]: ...


@overload
def plot_speed(  # numpydoc ignore=GL08
    trk: Tracking,
    bodypart: str = "body",
    *,
    speed_axis: Literal["x", "y", "xy"] = "xy",
    euclidean: bool = False,
    smooth: bool = True,
    speed_cutout: int | float = 0,
    only_running_bouts: bool = False,
    plot_only_running_bouts: bool = True,
    color: tuple[float, float, float, float] | None = None,
    alpha: float = 1.0,
    axes: matplotlib.axes.Axes | None = None,
    figure: matplotlib.figure.Figure | None = None,
    figsize: tuple[float, float] = (8, 4),
    animate: Literal[True],
    **ax_kwargs,
) -> tuple[BehFigure, matplotlib.axes.Axes, Animate_plot]: ...


@anim_decorator
def plot_speed(
    trk: Tracking,
    bodypart: str = "body",
    speed_axis: Literal["x", "y", "xy"] = "xy",
    euclidean: bool = False,
    smooth: bool = True,
    speed_cutout: int | float = 0,
    only_running_bouts: bool = False,
    plot_only_running_bouts: bool = True,
    color: ColorType | None = None,
    alpha: float = 1.0,
    axes: matplotlib.axes.Axes | None = None,
    figure: matplotlib.figure.Figure | None = None,
    figsize: tuple[float, float] = (8, 4),
    animate: bool = False,
    **ax_kwargs,
) -> (
    tuple[BehFigure, matplotlib.axes.Axes]
    | tuple[BehFigure, matplotlib.axes.Axes, Animate_plot]
):
    """Plot speed of given label.

    Parameters
    ----------
    trk : Tracking
        The tracking object.
    bodypart : str, optional
        Bodypart label. Default is ``"body"``.
    speed_axis : {"x", "y", "xy"}, optional
        To compute Vx, Vy or V, axis is 'x', 'y' or 'xy', respectively. Default is
        ``"xy"``.
    euclidean : bool, optional
        If `speed_axis` is only one dimension, the distance can be the euclidean
        (absolute) or real. Default is ``False``.
    smooth : bool, optional
        If speed array is to be smoothed using a gaussian kernel. Default is ``True``.
    speed_cutout : int, optional
        If speed is to be thresholded by some value. Default is ``0``.
    only_running_bouts : bool, optional
        If should plot only the running periods using
        :class:`physbeh.tracking.Tracking.get_running_bouts` function. Default
        is ``False``.
    plot_only_running_bouts : bool, optional
        Whether or not to plot a background color on periods of running bouts (and not
        only not plot non running bouts). This only takes effect if `only_running_bouts`
        is set to ``True``. Default is ``True``.
    color : color, optional
        Tuple of RGB(A) values for color of the line collection, if ``None``, uses the
        color defined by the label. Default is ``None``.
    alpha : float, optional
        The alpha value of the line collection. Default is ``1.0``.
    axes : matplotlib.axes.Axes, optional
        If ``None``, new axes is created in `figure`. Default is ``None``.
    figure : matplotlib.figure.Figure, optional
        If ``None``, new figure is created. Default is ``None``.
    figsize : tuple, optional
        Figure size, if ``figure=None``. Default is ``(12,6)``.
    animate : bool, optional
        If set to ``True``, plots an animation with the video of the Tracking class.
        Default is ``False``.
    **ax_kwargs
        Keywords to pass to ``ax.set(**ax_kwargs)``.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The matplotlib figure object.
    axes : matplotlib.axes.Axes
        The matplotlib axes object.
    """
    speed_array, time_array, index = trk.get_speed(
        bodypart=bodypart,
        axis=speed_axis,
        euclidean_distance=euclidean,
        smooth=smooth,
        speed_cutout=speed_cutout,
        only_running_bouts=only_running_bouts,
    )

    speed_units = str(trk.space_units[bodypart + "_x"].units) + "/" + trk.time_units

    ax_kwargs.setdefault("ylabel", f"animal speed ({speed_units})")
    ax_kwargs.setdefault("xlabel", "time (s)")
    ax_kwargs.setdefault("legend__loc", "upper right")
    ax_kwargs.setdefault("grid__linestyle", "--")
    behfigure, axes, _ = plot_array(
        speed_array,
        time_array=time_array,
        index=index,
        only_running_bouts=only_running_bouts,
        label=bodypart,
        color=get_label_color(trk, bodypart) if color is None else color,
        axes=axes,
        figure=figure,
        figsize=figsize,
        alpha=alpha,
        **ax_kwargs,
    )

    if only_running_bouts and plot_only_running_bouts:
        plot_running_bouts(trk, axes=axes, set_axes=False)

    return behfigure, axes


@overload
def plot_acceleration(  # numpydoc ignore=GL08
    trk: Tracking,
    bodypart: str = "body",
    *,
    smooth: bool = True,
    speed_cutout: int | float = 0,
    only_running_bouts: bool = False,
    plot_only_running_bouts: bool = True,
    color: ColorType | None = None,
    alpha: float = 1.0,
    axes: matplotlib.axes.Axes | None = None,
    figure: matplotlib.figure.Figure | None = None,
    figsize: tuple[float, float] = (12, 6),
    animate: Literal[False],
    **ax_kwargs,
) -> tuple[BehFigure, matplotlib.axes.Axes]: ...


@overload
def plot_acceleration(  # numpydoc ignore=GL08
    trk: Tracking,
    bodypart: str = "body",
    *,
    smooth: bool = True,
    speed_cutout: int | float = 0,
    only_running_bouts: bool = False,
    plot_only_running_bouts: bool = True,
    color: ColorType | None = None,
    alpha: float = 1.0,
    axes: matplotlib.axes.Axes | None = None,
    figure: matplotlib.figure.Figure | None = None,
    figsize: tuple[float, float] = (12, 6),
    animate: Literal[True],
    **ax_kwargs,
) -> tuple[BehFigure, matplotlib.axes.Axes, Animate_plot]: ...


@anim_decorator
def plot_acceleration(
    trk: Tracking,
    bodypart: str = "body",
    smooth: bool = True,
    speed_cutout: int | float = 0,
    only_running_bouts: bool = False,
    plot_only_running_bouts: bool = True,
    color: ColorType | None = None,
    alpha: float = 1.0,
    axes: matplotlib.axes.Axes | None = None,
    figure: matplotlib.figure.Figure | None = None,
    figsize: tuple[float, float] = (12, 6),
    animate: bool = False,
    **ax_kwargs,
) -> (
    tuple[BehFigure, matplotlib.axes.Axes]
    | tuple[BehFigure, matplotlib.axes.Axes, Animate_plot]
):
    """Plot acceleration of given label.

    Parameters
    ----------
    trk : Tracking
        The tracking object.
    bodypart : str, optional
        Bodypart label. Default is ``"body"``.
    smooth : bool, optional
        If speed array is to be smoothed using a gaussian kernel before calculating the
        acceleration. Default is ``True``.
    speed_cutout : int, optional
        If speed is to be thresholded by some value. Default is ``0``.
    only_running_bouts : bool, optional
        If should plot only the running periods using
        :class:`physbeh.tracking.Tracking.get_running_bouts` function. Default
        is ``False``.
    plot_only_running_bouts : bool, optional
        Whether or not to plot a background color on periods of running bouts (and not
        only not plot non running bouts). This only takes effect if `only_running_bouts`
        is set to ``True``. Default is ``True``.
    color : color, optional
        Tuple of RGB(A) values for color of the line collection, if ``None``, uses the
        color defined by the label. Default is ``None``.
    alpha : float, optional
        The alpha value of the line collection. Default is ``1.0``.
    axes : matplotlib.axes.Axes, optional
        If ``None``, new axes is created in `figure`. Default is ``None``.
    figure : matplotlib.figure.Figure, optional
        If ``None``, new figure is created. Default is ``None``.
    figsize : tuple, optional
        Figure size, if ``figure=None``. Default is ``(12,6)``.
    animate : bool, optional
        If set to ``True``, plots an animation with the video of the Tracking class.
        Default is ``False``.
    **ax_kwargs
        Keywords to pass to ``ax.set(**ax_kwargs)``.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The matplotlib figure object.
    axes : matplotlib.axes.Axes
        The matplotlib axes object.
    """
    acceleration_array, time_array, index = trk.get_acceleration(
        bodypart=bodypart,
        smooth=smooth,
        speed_cutout=speed_cutout,
        only_running_bouts=only_running_bouts,
    )

    acc_units = (
        str(trk.space_units[bodypart + "_x"].units) + "/" + trk.time_units + "**2"
    )

    ax_kwargs.setdefault("ylabel", f"animal acceleration ({acc_units})")
    ax_kwargs.setdefault("xlabel", "time (s)")
    ax_kwargs.setdefault("legend__loc", "upper right")
    ax_kwargs.setdefault("grid__linestyle", "--")
    behfigure, axes, _ = plot_array(
        acceleration_array,
        time_array=time_array,
        index=index,
        only_running_bouts=only_running_bouts,
        label=bodypart,
        color=get_label_color(trk, bodypart) if color is None else color,
        axes=axes,
        figure=figure,
        figsize=figsize,
        alpha=alpha,
        **ax_kwargs,
    )

    if only_running_bouts and plot_only_running_bouts:
        plot_running_bouts(trk, axes=axes, set_axes=False)

    return behfigure, axes


@overload
def plot_wall_proximity(  # numpydoc ignore=GL08
    animate: Literal[False],
) -> tuple[BehFigure, matplotlib.axes.Axes]: ...


@overload
def plot_wall_proximity(  # numpydoc ignore=GL08
    animate: Literal[True],
) -> tuple[BehFigure, matplotlib.axes.Axes, Animate_plot]: ...


@anim_decorator
def plot_wall_proximity(
    trk: Tracking,
    wall: Literal["left", "right", "top", "bottom", "all"]
    | list[Literal["left", "right", "top", "bottom"]] = "left",
    bodypart="neck",
    only_running_bouts=False,
    plot_only_running_bouts: bool = True,
    color: ColorType | None = None,
    alpha: float = 1.0,
    axes: matplotlib.axes.Axes | None = None,
    figure: matplotlib.figure.Figure | None = None,
    figsize: tuple[float, float] = (14, 7),
    animate: bool = False,
    animate_video: bool = False,
    animate_fus: bool = False,
    **ax_kwargs,
):
    """Plot proximity to specified wall.

    See :class:`physbeh.tracking.Tracking.get_proximity_from_wall`. for more
    information.

    Parameters
    ----------
    trk : physbeh.tracking.Tracking
        The tracking object.
    wall : str or list of str or tuple of str, optional
        Wall to use for computations. Can be one of ("left", "right", "top",
        "bottom"). Default is ``"left"``.
    bodypart : str, optional
        Bodypart to use for computations. Default "probe".
    only_running_bouts : bool, optional
        If should plot only the running periods using
        :class:`physbeh.tracking.Tracking.get_running_bouts` function. Default
        is ``False``.
    plot_only_running_bouts : bool, optional
        Whether or not to plot a background color on periods of running bouts (and not
        only not plot non running bouts). This only takes effect if `only_running_bouts`
        is set to ``True``. Default is ``True``.
    color : color, optional
        Tuple of RGB(A) values for color of the line collection, if ``None``, uses the
        color defined by the label. Default is ``None``.
    alpha : float, optional
        The alpha value of the line collection. Default is ``1.0``.
    axes : matplotlib.axes.Axes, optional
        If ``None``, new axes is created in `figure`. Default is ``None``.
    figure : matplotlib.figure.Figure, optional
        If ``None``, new figure is created. Default is ``None``.
    figsize : tuple, optional
        Figure size, if ``figure=None``. Default is ``(12,6)``.
    animate : bool, optional
        If set to ``True``, plots an animation with the video of the Tracking class.
        Default is ``False``.
    animate_video : bool, optional
        Whether to animate the plot with the video recording. Default is ``False``.
    animate_fus : bool, optional
        Whether to animate the plot with the functional Ultrasound video. Default is
        ``False``.
    **ax_kwargs
        Keywords to pass to ``ax.set(**ax_kwargs)``.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The matplotlib figure object.
    axes : matplotlib.axes.Axes
        The matplotlib axes object.
    """
    cast(
        Literal[False],
        only_running_bouts,
    )
    if only_running_bouts:
        cast(Literal[True], only_running_bouts)

    wall_proximity, time_array, index = trk.get_proximity_from_wall(
        wall=wall, bodypart=bodypart, only_running_bouts=only_running_bouts
    )

    ax_kwargs.setdefault("ylabel", f"Proximity from {wall} wall (a.u)")
    ax_kwargs.setdefault("xlabel", "time (s)")
    ax_kwargs.setdefault("legend__loc", "upper right")
    ax_kwargs.setdefault("grid__linestyle", "--")
    behfigure, axes, _ = plot_array(
        wall_proximity,
        time_array=time_array,
        index=index,
        only_running_bouts=only_running_bouts,
        label=bodypart,
        color=get_label_color(trk, bodypart) if color is None else color,
        axes=axes,
        figure=figure,
        figsize=figsize,
        alpha=alpha,
        **ax_kwargs,
    )

    if only_running_bouts and plot_only_running_bouts:
        plot_running_bouts(trk, axes=axes, set_axes=False)

    return behfigure, axes


@anim_decorator
def plot_center_proximity(
    trk: Tracking,
    bodypart="probe",
    only_running_bouts=False,
    plot_only_running_bouts: bool = True,
    color: ColorType | None = None,
    alpha: float = 1.0,
    axes: matplotlib.axes.Axes | None = None,
    figure: matplotlib.figure.Figure | None = None,
    figsize: tuple[float, float] = (14, 7),
    animate: bool = False,
    animate_video: bool = False,
    animate_fus: bool = False,
    **ax_kwargs,
):
    """Plot proximity to the center of the environment.

    See :class:`physbeh.tracking.Tracking.get_proximity_from_center`. for more
    information.

    Parameters
    ----------
    trk : physbeh.tracking.Tracking
        The tracking object.
    bodypart : str, optional
        Bodypart to use for computations. Default "probe".
    only_running_bouts : bool, optional
        If should plot only the running periods using
        :class:`physbeh.tracking.Tracking.get_running_bouts` function. Default
        is ``False``.
    plot_only_running_bouts : bool, optional
        Whether or not to plot a background color on periods of running bouts (and not
        only not plot non running bouts). This only takes effect if `only_running_bouts`
        is set to ``True``. Default is ``True``.
    color : color, optional
        Tuple of RGB(A) values for color of the line collection, if ``None``, uses the
        color defined by the label. Default is ``None``.
    alpha : float, optional
        The alpha value of the line collection. Default is ``1.0``.
    axes : matplotlib.axes.Axes, optional
        If ``None``, new axes is created in `figure`. Default is ``None``.
    figure : matplotlib.figure.Figure, optional
        If ``None``, new figure is created. Default is ``None``.
    figsize : tuple, optional
        Figure size, if ``figure=None``. Default is ``(12,6)``.
    animate : bool, optional
        If set to ``True``, plots an animation with the video of the Tracking class.
        Default is ``False``.
    animate_video : bool, optional
        Whether to animate the plot with the video recording. Default is ``False``.
    animate_fus : bool, optional
        Whether to animate the plot with the functional Ultrasound video. Default is
        ``False``.
    **ax_kwargs
        Keywords to pass to ``ax.set(**ax_kwargs)``.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The matplotlib figure object.
    axes : matplotlib.axes.Axes
        The matplotlib axes object.
    """

    center_proximity, time_array, index = trk.get_proximity_from_center(
        bodypart=bodypart, only_running_bouts=only_running_bouts
    )

    ax_kwargs.setdefault("ylabel", "Proximity from center of stage (a.u)")
    ax_kwargs.setdefault("xlabel", "time (s)")
    ax_kwargs.setdefault("legend__loc", "upper right")
    ax_kwargs.setdefault("grid__linestyle", "--")
    behfigure, axes, _ = plot_array(
        center_proximity,
        time_array=time_array,
        index=index,
        only_running_bouts=only_running_bouts,
        label=bodypart,
        color=get_label_color(trk, bodypart) if color is None else color,
        axes=axes,
        figure=figure,
        figsize=figsize,
        alpha=alpha,
        **ax_kwargs,
    )

    if only_running_bouts and plot_only_running_bouts:
        plot_running_bouts(trk, axes=axes, set_axes=False)

    return behfigure, axes


@anim_decorator
def plot_corner_proximity(
    trk: Tracking,
    corner: str = "top right",
    bodypart="probe",
    only_running_bouts=False,
    plot_only_running_bouts: bool = True,
    color: ColorType | None = None,
    alpha=1.0,
    axes: matplotlib.axes.Axes | None = None,
    figure: matplotlib.figure.Figure | None = None,
    figsize: tuple[float, float] = (14, 7),
    animate: bool = False,
    animate_video: bool = False,
    animate_fus: bool = False,
    **ax_kwargs,
):
    """Plot proximity to specified corner.

    See :class:`physbeh.tracking.Tracking.get_proximity_from_corner`. for more
    information.

    Parameters
    ----------
    trk : Tracking
        The tracking object.
    corner : str, optional
        Must be one of the four corners of a rectangle ("top right", "top left", "bottom
        right", "bottom left"). Default is ``"top right"``.
    bodypart : str, optional
        Bodypart to use for computations. Default "probe".
    only_running_bouts : bool, optional
        If should plot only the running periods using
        :class:`physbeh.tracking.Tracking.get_running_bouts` function. Default
        is ``False``.
    plot_only_running_bouts : bool, optional
        Whether or not to plot a background color on periods of running bouts (and not
        only not plot non running bouts). This only takes effect if `only_running_bouts`
        is set to ``True``. Default is ``True``.
    color : color, optional
        Tuple of RGB(A) values for color of the line collection, if ``None``, uses the
        color defined by the label. Default is ``None``.
    alpha : float, optional
        The alpha value of the line collection. Default is ``1.0``.
    axes : matplotlib.axes.Axes, optional
        If ``None``, new axes is created in `figure`. Default is ``None``.
    figure : matplotlib.figure.Figure, optional
        If ``None``, new figure is created. Default is ``None``.
    figsize : tuple, optional
        Figure size, if ``figure=None``. Default is ``(12,6)``.
    animate : bool, optional
        If set to ``True``, plots an animation with the video of the Tracking class.
        Default is ``False``.
    animate_video : bool, optional
        Whether to animate the plot with the video recording. Default is ``False``.
    animate_fus : bool, optional
        Whether to animate the plot with the functional Ultrasound video. Default is
        ``False``.
    **ax_kwargs
        Keywords to pass to ``ax.set(**ax_kwargs)``.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The matplotlib figure object.
    axes : matplotlib.axes.Axes
        The matplotlib axes object.
    """

    corner_proximity, time_array, index = trk.get_proximity_from_corner(
        corner=corner, bodypart=bodypart, only_running_bouts=only_running_bouts
    )

    ax_kwargs.setdefault("ylabel", f"Proximity from {corner} corner (a.u)")
    ax_kwargs.setdefault("xlabel", "time (s)")
    ax_kwargs.setdefault("legend__loc", "upper right")
    ax_kwargs.setdefault("grid__linestyle", "--")
    behfigure, axes, _ = plot_array(
        corner_proximity,
        time_array=time_array,
        index=index,
        only_running_bouts=only_running_bouts,
        label=bodypart,
        color=get_label_color(trk, bodypart) if color is None else color,
        axes=axes,
        figure=figure,
        figsize=figsize,
        alpha=alpha,
        **ax_kwargs,
    )

    if only_running_bouts and plot_only_running_bouts:
        plot_running_bouts(trk, axes=axes, set_axes=False)

    return behfigure, axes


@overload
def plot_angular_velocity(  # numpydoc ignore=GL08
    trk: Tracking,
    label0: str = "neck",
    label1: str = "probe",
    *,
    smooth: bool = True,
    only_running_bouts=False,
    plot_only_running_bouts: bool = True,
    color: ColorType | None = None,
    alpha: float = 1.0,
    axes: matplotlib.axes.Axes | None = None,
    figure: matplotlib.figure.Figure | None = None,
    figsize: tuple[float, float] = (14, 7),
    animate: Literal[True],
    **ax_kwargs,
) -> tuple[BehFigure, matplotlib.axes.Axes, Animate_plot]: ...


@overload
def plot_angular_velocity(  # numpydoc ignore=GL08
    trk: Tracking,
    label0: str = "neck",
    label1: str = "probe",
    *,
    smooth: bool = True,
    only_running_bouts=False,
    plot_only_running_bouts: bool = True,
    color: ColorType | None = None,
    alpha: float = 1.0,
    axes: matplotlib.axes.Axes | None = None,
    figure: matplotlib.figure.Figure | None = None,
    figsize: tuple[float, float] = (14, 7),
    animate: Literal[False] = False,
) -> tuple[BehFigure, matplotlib.axes.Axes]: ...


@overload
def plot_angular_velocity(  # numpydoc ignore=GL08
    trk: Tracking,
    label0: str = "neck",
    label1: str = "probe",
    *,
    smooth: bool = True,
    only_running_bouts=False,
    plot_only_running_bouts: bool = True,
    color: ColorType | None = None,
    alpha: float = 1.0,
    axes: matplotlib.axes.Axes | None = None,
    figure: matplotlib.figure.Figure | None = None,
    figsize: tuple[float, float] = (14, 7),
    animate: bool,
    **ax_kwargs,
) -> (
    tuple[BehFigure, matplotlib.axes.Axes]
    | tuple[BehFigure, matplotlib.axes.Axes, Animate_plot]
): ...


@anim_decorator
def plot_angular_velocity(
    trk: Tracking,
    label0: str = "neck",
    label1: str = "probe",
    *,
    smooth: bool = True,
    only_running_bouts=False,
    plot_only_running_bouts: bool = True,
    color: ColorType | None = None,
    alpha: float = 1.0,
    axes: matplotlib.axes.Axes | None = None,
    figure: matplotlib.figure.Figure | None = None,
    figsize: tuple[float, float] = (14, 7),
    animate: bool = False,
    **ax_kwargs,
) -> (
    tuple[BehFigure, matplotlib.axes.Axes]
    | tuple[BehFigure, matplotlib.axes.Axes, Animate_plot]
):
    """Plot angular velocity calculated from the vector 'label0' -> 'label1'.

    See :class:`physbeh.tracking.Tracking.get_angular_velocity`, for
    more information.

    Parameters
    ----------
    trk : physbeh.tracking.Tracking
        The tracking object.
    label0 : str, optional
        Label where the vector will start. Default is ``'neck'``.
    label1 : str, optional
        Label where the vector will finish. Default is ``'probe'``.
    smooth : bool, optional
        Whether or not to smooth the direction data. Default is ``False``.
    only_running_bouts : bool, optional
        If should plot only the running periods using
        :class:`physbeh.tracking.Tracking.get_running_bouts` function. Default
        is ``False``.
    plot_only_running_bouts : bool, optional
        Whether or not to plot a background color on periods of running bouts (and not
        only not plot non running bouts). This only takes effect if `only_running_bouts`
        is set to ``True``. Default is ``True``.
    color : color, optional
        Tuple of RGB(A) values for color of the line collection, if ``None``, uses the
        color defined by the label. Default is ``None``.
    alpha : float, optional
        The alpha value of the line collection. Default is ``1.0``.
    axes : matplotlib.axes.Axes, optional
        If ``None``, new axes is created in `figure`. Default is ``None``.
    figure : matplotlib.figure.Figure, optional
        If ``None``, new figure is created. Default is ``None``.
    figsize : tuple, optional
        Figure size, if ``figure=None``. Default is ``(12,6)``.
    animate : bool, optional
        If set to ``True``, plots an animation with the video of the Tracking class.
        Default is ``False``.
    **ax_kwargs
        Keywords to pass to ``ax.set(**ax_kwargs)``.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The matplotlib figure object.
    axes : matplotlib.axes.Axes
        The matplotlib axes object.
    """

    ang_velocity, time_array, index = trk.get_angular_velocity(
        label0=label0,
        label1=label1,
        smooth=smooth,
        only_running_bouts=only_running_bouts,
    )

    ax_kwargs.setdefault("ylabel", "Angular velocity (rad/s)")
    ax_kwargs.setdefault("xlabel", "time (s)")
    ax_kwargs.setdefault("legend__loc", "upper right")
    ax_kwargs.setdefault("grid__linestyle", "--")
    behfigure, axes, _ = plot_array(
        ang_velocity,
        time_array=time_array,
        index=index,
        only_running_bouts=only_running_bouts,
        label=label0 + " -> " + label1,
        color=get_label_color(trk, label1) if color is None else color,
        axes=axes,
        figure=figure,
        figsize=figsize,
        alpha=alpha,
        **ax_kwargs,
    )

    if only_running_bouts and plot_only_running_bouts:
        plot_running_bouts(trk, axes=axes, set_axes=False)

    return behfigure, axes


@overload
def plot_angular_acceleration(  # numpydoc ignore=GL08
    animate: Literal[True],
) -> tuple[BehFigure, matplotlib.axes.Axes, Animate_plot]: ...


@overload
def plot_angular_acceleration(  # numpydoc ignore=GL08
    animate: Literal[False],
) -> tuple[BehFigure, matplotlib.axes.Axes]: ...


@anim_decorator
def plot_angular_acceleration(
    trk: Tracking,
    label0: str = "neck",
    label1: str = "probe",
    smooth: bool = True,
    *,
    only_running_bouts=False,
    plot_only_running_bouts: bool = True,
    color: ColorType | None = None,
    alpha: float = 1.0,
    axes: matplotlib.axes.Axes | None = None,
    figure: matplotlib.figure.Figure | None = None,
    figsize: tuple[float, float] = (14, 7),
    animate: bool = False,
    animate_video: bool = False,
    animate_fus: bool = False,
    **ax_kwargs,
):
    """Plot angular acceleration calculated from the vector 'label0' -> 'label1'.

    See :class:`physbeh.tracking.Tracking.get_angular_acceleration`, for
    more information.

    Parameters
    ----------
    trk : physbeh.tracking.Tracking
        The tracking object.
    label0 : str, optional
        Label where the vector will start. Default is ``'neck'``.
    label1 : str, optional
        Label where the vector will finish. Default is ``'probe'``.
    smooth : bool, optional
        Whether or not to smooth the direction data. Default is ``False``.
    only_running_bouts : bool, optional
        If should plot only the running periods using
        :class:`physbeh.tracking.Tracking.get_running_bouts` function. Default
        is ``False``.
    plot_only_running_bouts : bool, optional
        Whether or not to plot a background color on periods of running bouts (and not
        only not plot non running bouts). This only takes effect if `only_running_bouts`
        is set to ``True``. Default is ``True``.
    color : color, optional
        Tuple of RGB(A) values for color of the line collection, if ``None``, uses the
        color defined by the label. Default is ``None``.
    alpha : float, optional
        The alpha value of the line collection. Default is ``1.0``.
    axes : matplotlib.axes.Axes, optional
        If ``None``, new axes is created in `figure`. Default is ``None``.
    figure : matplotlib.figure.Figure, optional
        If ``None``, new figure is created. Default is ``None``.
    figsize : tuple, optional
        Figure size, if ``figure=None``. Default is ``(12,6)``.
    animate : bool, optional
        If set to ``True``, plots an animation with the video of the Tracking class.
        Default is ``False``.
    animate_video : bool, optional
        Whether to animate the plot with the video recording. Default is ``False``.
    animate_fus : bool, optional
        Whether to animate the plot with the functional Ultrasound video. Default is
        ``False``.
    **ax_kwargs
        Keywords to pass to ``ax.set(**ax_kwargs)``.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The matplotlib figure object.
    axes : matplotlib.axes.Axes
        The matplotlib axes object.
    """

    ang_acceleration, time_array, index = trk.get_angular_acceleration(
        label0=label0,
        label1=label1,
        smooth=smooth,
        only_running_bouts=only_running_bouts,
    )

    ax_kwargs.setdefault("ylabel", "Angular acceleration (rad/s**2)")
    ax_kwargs.setdefault("xlabel", "time (s)")
    ax_kwargs.setdefault("legend__loc", "upper right")
    ax_kwargs.setdefault("grid__linestyle", "--")
    behfigure, axes, _ = plot_array(
        ang_acceleration,
        time_array=time_array,
        index=index,
        only_running_bouts=only_running_bouts,
        label=label0 + " -> " + label1,
        color=get_label_color(trk, label1) if color is None else color,
        axes=axes,
        figure=figure,
        figsize=figsize,
        alpha=alpha,
        **ax_kwargs,
    )

    if only_running_bouts and plot_only_running_bouts:
        plot_running_bouts(trk, axes=axes, set_axes=False)

    return behfigure, axes


@overload
def plot_running_bouts(  # numpydoc ignore=GL08
    trk: Tracking,
    *,
    axes: matplotlib.axes.Axes | None = None,
    figure: matplotlib.figure.Figure | None = None,
    figsize: tuple[float, float] = (12, 6),
    set_axes: bool = False,
    animate: Literal[False] = False,
    **ax_kwargs,
) -> tuple[BehFigure, matplotlib.axes.Axes]: ...


@overload
def plot_running_bouts(  # numpydoc ignore=GL08
    trk: Tracking,
    *,
    axes: matplotlib.axes.Axes | None = None,
    figure: matplotlib.figure.Figure | None = None,
    figsize: tuple[float, float] = (12, 6),
    set_axes: bool = False,
    animate: Literal[True],
    **ax_kwargs,
) -> tuple[BehFigure, matplotlib.axes.Axes, Animate_plot]: ...


@anim_decorator
def plot_running_bouts(
    trk: Tracking,
    axes: matplotlib.axes.Axes | None = None,
    figure: matplotlib.figure.Figure | None = None,
    figsize: tuple[float, float] = (12, 6),
    set_axes: bool = False,
    animate: bool = False,
    **ax_kwargs,
) -> (
    tuple[BehFigure, matplotlib.axes.Axes]
    | tuple[BehFigure, matplotlib.axes.Axes, Animate_plot]
):
    """Plot the running periods of the animal.

    See :class:`physbeh.tracking.Tracking.get_running_bouts`.

    Parameters
    ----------
    trk : Tracking
        The tracking object.
    axes : matplotlib.axes.Axes, optional
        If ``None``, new axes is created in `figure`. Default is ``None``.
    figure : matplotlib.figure.Figure, optional
        If ``None``, new figure is created. Default is ``None``.
    figsize : tuple, optional
        Figure size, if ``figure=None``. Default is ``(12,6)``.
    set_axes : bool, optional
        Whether to set the axes properties. Default is ``True``.
    animate : bool, optional
        If set to ``True``, plots an animation with the video of the Tracking class.
        Default is ``False``.
    **ax_kwargs
        Keywords to pass to ``ax.set(**ax_kwargs)``.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The matplotlib figure object.
    axes : matplotlib.axes.Axes
        The matplotlib axes object.
    """
    if not hasattr(trk, "running_bouts"):
        trk.get_running_bouts()

    axes, behfigure = _check_ax_and_fig(axes, figure, figsize=figsize)

    axes.fill_between(
        trk.time,
        0,
        1,
        where=trk.running_bouts,
        transform=axes.get_xaxis_transform(),
        color="orange",
        alpha=0.5,
    )

    ax_kwargs.setdefault("ylabel", "running bouts")
    ax_kwargs.setdefault("xlabel", "time (s)")
    ax_kwargs.setdefault("grid__linestyle", "--")
    if set_axes:
        keys = list(ax_kwargs.keys())
        grid_kwargs = {
            key.replace("grid__", ""): ax_kwargs.pop(key)
            for key in keys
            if key.startswith("grid__")
        }
        axes.set(**ax_kwargs)
        axes.grid(**grid_kwargs)

    return behfigure, axes


@overload
def plot_position_2d(  # numpydoc ignore=GL08
    trk: Tracking,
    bodypart: str = "body",
    *,
    color_collection_array: npt.NDArray | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    head_direction: bool = True,
    head_direction_vector_labels: tuple[str, str] | list[str] = ["neck", "probe"],
    only_running_bouts: bool = False,
    cmap: str | colors.Colormap | None = None,
    colorwheel=True,
    colorbar: bool = True,
    cbar_label: str = "",
    color="gray",
    axes=None,
    figure=None,
    figsize: tuple[float, float] = (8, 6),
    animate: Literal[False] = False,
    **ax_kwargs,
) -> tuple[BehFigure, matplotlib.axes.Axes, dict[str, npt.NDArray]]: ...


@overload
def plot_position_2d(  # numpydoc ignore=GL08
    trk: Tracking,
    bodypart: str = "body",
    *,
    color_collection_array: npt.NDArray | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    head_direction: bool = True,
    head_direction_vector_labels: tuple[str, str] | list[str] = ["neck", "probe"],
    only_running_bouts: bool = False,
    cmap: str | colors.Colormap | None = None,
    colorwheel=True,
    colorbar: bool = True,
    cbar_label: str = "",
    color="gray",
    axes=None,
    figure=None,
    figsize: tuple[float, float] = (8, 6),
    animate: Literal[True],
    **ax_kwargs,
) -> tuple[BehFigure, matplotlib.axes.Axes, Animate_plot2D]: ...


@anim2d_decorator
def plot_position_2d(
    trk: Tracking,
    bodypart: str = "body",
    *,
    color_collection_array: npt.NDArray | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    head_direction: bool = True,
    head_direction_vector_labels: tuple[str, str] | list[str] = ["neck", "probe"],
    only_running_bouts: bool = False,
    cmap: str | colors.Colormap | None = None,
    colorwheel=True,
    colorbar: bool = True,
    cbar_label: str = "",
    color="gray",
    axes=None,
    figure=None,
    figsize: tuple[float, float] = (8, 6),
    animate: bool = False,
    **ax_kwargs,
) -> tuple[BehFigure, matplotlib.axes.Axes, dict[str, npt.NDArray] | Animate_plot2D]:
    """Plot position of the animal in 2D coordinates.

    Parameters
    ----------
    trk : Tracking
        The tracking object.
    bodypart : str, optional
        Bodypart label. Default is ``"body"``.
    color_collection_array : numpy.ndarray, optional
        The array of values to be used for color mapping the line collection. Default is
        ``None``.
    vmin, vmax : float or None, optional
        When using no explicit `norm`, `vmin` and `vmax` define the data range that the
        colormap covers. If ``None``, the colormap covers the complete value range of
        the displayed slices. It is an error to use `vmin`/`vmax` when a `norm` instance
        is given, but using a ``str`` `norm` name together with `vmin`/`vmax` is
        accepted. Data range that the colormap covers. Default is ``None``.
    head_direction : bool, optional
        Whether or not to color the lines based on the head direction. Default is
        ``True``.
    head_direction_vector_labels : tuple or list of str, optional
        The labels of the vectors to be used for the head direction. Default is
        ``["neck", "probe"]``.
    only_running_bouts : bool, optional
        If should plot only the running periods using
        :class:`physbeh.tracking.Tracking.get_running_bouts` function. Default
        is ``False``.
    cmap : str, optional
        The colormap to use for the plot. Default is ``"hsv"``.
    colorwheel : bool, optional
        Whether to plot the color wheel. Takes precedence over `colorbar`. Default is
        ``True``.
    colorbar : bool, optional
        Whether to plot the colorbar. Default is ``True``.
    cbar_label : str, optional
        The label of the colorbar. Default is ``None``.
    color : color, optional
        Tuple of RGB(A) values for color of the line collection, if not using
        `head_direction` or any `color_collection_array`. Default is ``(0.5, 0.5, 0.5,
        1.0)``.
    axes : matplotlib.axes.Axes, optional
        If ``None``, new axes is created in `figure`. Default is ``None``.
    figure : matplotlib.figure.Figure, optional
        If ``None``, new figure is created. Default is ``None``.
    figsize : tuple, optional
        Figure size, if ``figure=None``. Default is ``(8,6)``.
    animate : bool, optional
        If set to ``True``, plots an animation with the video of the Tracking class.
        Default is ``False``.
    **ax_kwargs
        Additional keyword arguments to pass to the axes.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The matplotlib figure object.
    axes : matplotlib.axes.Axes
        The matplotlib axes object.
    """

    x_bp, _, index = trk.get_position_x(bodypart=bodypart)
    y_bp = trk.get_position_y(bodypart=bodypart)[0]

    if only_running_bouts:
        trk.get_running_bouts()
        index = trk.running_bouts

    use_head_direction = False
    if color_collection_array is None and head_direction:
        use_head_direction = True
        hd_array, _, index = trk.get_direction_array(
            label0=head_direction_vector_labels[0],
            label1=head_direction_vector_labels[1],
            mode="deg",
            smooth=True,
            only_running_bouts=only_running_bouts,
        )
        lines = get_line_collection(x_bp, hd_array, index)
        color_collection_array = lines[:, 0, 1]
        cmap = "hsv"
        colorbar = False if colorwheel else colorbar
        vmin, vmax = 0, 360

    # lines = get_line_collection(x_array=x_bp, y_array=y_bp, index=index)
    spatial_units = trk.space_units[bodypart + "_x"].units
    ax_kwargs.setdefault("ylabel", f"Y ({spatial_units})")
    ax_kwargs.setdefault("xlabel", f"X ({spatial_units})")
    ax_kwargs.setdefault(
        "title", "Animal position in the arena [bodypart: " + bodypart + "]"
    )
    figure, axes, lines = plot_array(
        y_bp,
        x_bp,
        index=index,
        only_running_bouts=only_running_bouts,
        line_collection_array=color_collection_array,
        color=color,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        colorbar=colorbar,
        cbar_label=cbar_label,
        axes=axes,
        figure=figure,
        figsize=figsize,
        **ax_kwargs,
    )
    axes.set_aspect("equal", "box")
    if not axes.yaxis.get_inverted():
        axes.invert_yaxis()

    if use_head_direction:
        if colorwheel:
            figure.figure.set_size_inches(14, 7.5)
            axes.set_position([0.12, 0.12, 0.5, 0.75])
            ax_cw = figure.figure.add_axes(
                rect=[0.65, 0.26, 0.3, 0.48], projection="polar"
            )
            _plot_color_wheel(ax=ax_cw, cmap=cmap)

    lines_index = np.where(index)[0]
    dict_lines = {"lines": lines, "index": lines_index}
    return figure, axes, dict_lines


@anim_decorator
def plot_likelihood(
    trk: Tracking,
    bodyparts="all",
    alpha: float = 0.7,
    axes: matplotlib.axes.Axes | None = None,
    figure: matplotlib.figure.Figure | None = None,
    figsize: tuple[float, float] = (14, 7),
    animate: bool = False,
    animate_video: bool = False,
    animate_fus: bool = False,
    **ax_kwargs,
):
    """Plot likelihood for `bodyparts` in each frame.

    Parameters
    ----------
    trk : Tracking
        The tracking object.
    bodyparts : list or str, optional
        Labels to be plotted, it can be a string, a list of strings or `"all"` for all
        labels. Default is ``"all"``.
    alpha : float, optional
        The alpha value of the line collection. Default is ``0.7``.
    axes : matplotlib.axes.Axes, optional
        If ``None``, new axes is created in `figure`. Default is ``None``.
    figure : matplotlib.figure.Figure, optional
        If ``None``, new figure is created. Default is ``None``.
    figsize : tuple, optional
        Figure size, if ``figure=None``. Default is ``(12,6)``.
    animate : bool, optional
        If set to ``True``, plots an animation with the video of the Tracking class.
        Default is ``False``.
    animate_video : bool, optional
        Whether to animate the plot with the video recording. Default is ``False``.
    animate_fus : bool, optional
        Whether to animate the plot with the functional Ultrasound video. Default is
        ``False``.
    **ax_kwargs
        Additional keyword arguments to pass to the axes.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The matplotlib figure object.
    axes : matplotlib.axes.Axes
        The matplotlib axes object.
    """

    bodyparts = _listify_bodyparts(trk, bodyparts)

    axes, behfigure = _check_ax_and_fig(axes, figure, figsize=figsize)

    for bp in bodyparts:
        lk = trk.get_likelihood(bodypart=bp)

        axes.plot(
            trk.time,
            lk,
            ".",
            markersize=4,
            color=get_label_color(trk, bp),
            label=bp,
            alpha=alpha,
        )

    axes.set(ylabel="likelihood", xlabel="frames", ylim=(-0.05, 1.05))
    axes.set(**ax_kwargs)
    axes.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    axes.grid(linestyle="--")

    return behfigure, axes


@anim_decorator
def plot_position_x(
    trk: Tracking,
    bodyparts: str = "all",
    alpha: float = 1.0,
    axes: matplotlib.axes.Axes | None = None,
    figure: matplotlib.figure.Figure | None = None,
    figsize: tuple[float, float] = (14, 7),
    animate: bool = False,
    animate_video: bool = False,
    animate_fus: bool = False,
    **ax_kwargs,
):
    """Plot the ``x`` coordinates of requested `bodyparts`.

    Parameters
    ----------
    trk : Tracking
        The tracking object.
    bodyparts : str or list of str, optional
        Bodypart labels, accepts string or list of strings or ``"all"`` for all labels.
        Default is ``"all"``.
    alpha : float, optional
        The alpha value of the line collection. Default is ``1.0``.
    axes : matplotlib.axes.Axes, optional
        If ``None``, new axes is created in `figure`. Default is ``None``.
    figure : matplotlib.figure.Figure, optional
        If ``None``, new figure is created. Default is ``None``.
    figsize : tuple, optional
        Figure size, if ``figure=None``. Default is ``(12,6)``.
    animate : bool, optional
        If set to ``True``, plots an animation with the video of the Tracking class.
        Default is ``False``.
    animate_video : bool, optional
        Whether to animate the plot with the video recording. Default is ``False``.
    animate_fus : bool, optional
        Whether to animate the plot with the functional Ultrasound video. Default is
        ``False``.
    **ax_kwargs
        Additional keyword arguments to pass to the axes.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The matplotlib figure object.
    axes : matplotlib.axes.Axes
        The matplotlib axes object.
    """

    bodyparts = _listify_bodyparts(trk, bodyparts)

    spatial_units = trk.space_units[bodyparts[0] + "_x"].units
    ax_kwargs.setdefault("ylabel", f"X position ({spatial_units})")
    ax_kwargs.setdefault("xlabel", "time (s)")
    ax_kwargs.setdefault("legend__loc", "upper right")
    ax_kwargs.setdefault("grid__linestyle", "--")
    n_bodyparts = len(bodyparts)
    for i, bp in enumerate(bodyparts):
        x_bp, time_array, index = trk.get_position_x(bodypart=bp)

        figure, axes, _ = plot_array(
            x_bp,
            time_array,
            index,
            label=bp,
            alpha=alpha,
            axes=axes,
            figure=figure,
            figsize=figsize,
            color=get_label_color(trk, bp),
            set_axes=False if (i == 0 and n_bodyparts != 1) else True,
            **ax_kwargs,
        )

    return figure, axes


@anim_decorator
def plot_position_y(
    trk: Tracking,
    bodyparts: str | list[str] = "all",
    alpha: float = 1.0,
    axes: matplotlib.axes.Axes | None = None,
    figure: matplotlib.figure.Figure | None = None,
    figsize: tuple[float, float] = (14, 7),
    animate: bool = False,
    animate_video: bool = False,
    animate_fus: bool = False,
    **ax_kwargs,
):
    """Plot the ``y`` coordinates of `bodyparts`.

    Parameters
    ----------
    trk : Tracking
        The tracking object.
    bodyparts : str or list of str, optional
        Bodypart labels, accepts string or list of strings or ``"all"`` for all labels.
        Default is ``"all"``.
    alpha : float, optional
        The alpha value of the line collection. Default is ``1.0``.
    axes : matplotlib.axes.Axes, optional
        If ``None``, new axes is created in `figure`. Default is ``None``.
    figure : matplotlib.figure.Figure, optional
        If ``None``, new figure is created. Default is ``None``.
    figsize : tuple, optional
        Figure size, if ``figure=None``. Default is ``(12,6)``.
    animate : bool, optional
        If set to ``True``, plots an animation with the video of the Tracking class.
        Default is ``False``.
    animate_video : bool, optional
        Whether to animate the plot with the video recording. Default is ``False``.
    animate_fus : bool, optional
        Whether to animate the plot with the functional Ultrasound video. Default is
        ``False``.
    **ax_kwargs
        Additional keyword arguments to pass to the axes.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The matplotlib figure object.
    axes : matplotlib.axes.Axes
        The matplotlib axes object.
    """

    bodyparts = _listify_bodyparts(trk, bodyparts)

    spatial_units = trk.space_units[bodyparts[0] + "_y"].units
    ax_kwargs.setdefault("ylabel", f"Y position ({spatial_units})")
    ax_kwargs.setdefault("xlabel", "time (s)")
    ax_kwargs.setdefault("legend__loc", "upper right")
    ax_kwargs.setdefault("grid__linestyle", "--")
    n_bodyparts = len(bodyparts)
    for i, bp in enumerate(bodyparts):
        y_bp, time_array, index = trk.get_position_y(bodypart=bp)

        figure, axes, _ = plot_array(
            y_bp,
            time_array,
            index,
            label=bp,
            alpha=alpha,
            axes=axes,
            figure=figure,
            figsize=figsize,
            color=get_label_color(trk, bp),
            set_axes=False if (i == 0 and n_bodyparts != 1) else True,
            **ax_kwargs,
        )

    return figure, axes


def plot_position(
    trk: Tracking,
    bodyparts: str = "all",
    figure: matplotlib.figure.Figure | None = None,
    figsize: tuple[float, float] = (12, 6),
    **ax_kwargs,
):
    """Plot X and Y coordinates of requested `bodyparts` in two subplots.

    The top one is for X coordinates and bottom one is for Y coordinates

    Parameters
    ----------
    trk : Tracking
        The tracking object.
    bodyparts : str or list of str, optional
        Bodypart labels, accepts string or list of strings or ``"all"`` for all labels.
        Default is ``"all"``.
    figure : matplotlib.figure.Figure, optional
        If ``None``, new figure is created. Default is ``None``.
    figsize : tuple, optional
        Figure size, if `figure` is ``None``. Default is ``(12,6)``.
    **ax_kwargs
        Additional keyword arguments to be passed to the axes. These arguments are
        passed to both x and y position plotting functions.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The matplotlib figure object.
    axes : tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]
        The matplotlib axes object.
    """

    if figure is None:
        figure = plt.figure(figsize=figsize)

    ax_x = figure.add_subplot(211)
    ax_y = figure.add_subplot(212)

    behfigure = BehFigure(figure)

    plot_position_x(
        trk,
        bodyparts=bodyparts,
        axes=ax_x,
        figure=behfigure.figure,
        xlabel="",
        **ax_kwargs,
    )
    plot_position_y(
        trk, bodyparts=bodyparts, axes=ax_y, figure=behfigure.figure, **ax_kwargs
    )

    return behfigure, (ax_x, ax_y)


@overload
def plot_head_direction(  # numpydoc ignore=GL08
    trk: Tracking,
    head_direction_vector_labels=["neck", "probe"],
    *,
    ang: Literal["deg", "rad"] = "deg",
    smooth: bool = False,
    only_running_bouts: bool = False,
    plot_only_running_bouts: bool = True,
    color: ColorType | None = None,
    alpha: float = 1.0,
    label: str | None = "head direction",
    cmap: str | colors.Colormap = "hsv",
    axes: matplotlib.axes.Axes | None = None,
    figure: matplotlib.figure.Figure | None = None,
    figsize: tuple[float, float] = (8, 4),
    animate: Literal[True],
    **ax_kwargs,
) -> tuple[BehFigure, matplotlib.axes.Axes, Animate_plot]: ...


@overload
def plot_head_direction(  # numpydoc ignore=GL08
    trk: Tracking,
    head_direction_vector_labels=["neck", "probe"],
    *,
    ang: Literal["deg", "rad"] = "deg",
    smooth: bool = False,
    only_running_bouts: bool = False,
    plot_only_running_bouts: bool = True,
    color: ColorType | None = None,
    alpha: float = 1.0,
    label: str | None = "head direction",
    cmap: str | colors.Colormap = "hsv",
    axes: matplotlib.axes.Axes | None = None,
    figure: matplotlib.figure.Figure | None = None,
    figsize: tuple[float, float] = (8, 4),
    animate: Literal[False] = False,
    **ax_kwargs,
) -> tuple[BehFigure, matplotlib.axes.Axes]: ...


@overload
def plot_head_direction(  # numpydoc ignore=GL08
    trk: Tracking,
    head_direction_vector_labels=["neck", "probe"],
    *,
    ang: Literal["deg", "rad"] = "deg",
    smooth: bool = False,
    only_running_bouts: bool = False,
    plot_only_running_bouts: bool = True,
    color: ColorType | None = None,
    alpha: float = 1.0,
    label: str | None = "head direction",
    cmap: str | colors.Colormap = "hsv",
    axes: matplotlib.axes.Axes | None = None,
    figure: matplotlib.figure.Figure | None = None,
    figsize: tuple[float, float] = (8, 4),
    animate: bool,
    **ax_kwargs,
) -> (
    tuple[BehFigure, matplotlib.axes.Axes]
    | tuple[BehFigure, matplotlib.axes.Axes, Animate_plot]
): ...


@anim_decorator
def plot_head_direction(
    trk: Tracking,
    head_direction_vector_labels=["neck", "probe"],
    ang: Literal["deg", "rad"] = "deg",
    smooth: bool = False,
    only_running_bouts: bool = False,
    plot_only_running_bouts: bool = True,
    color: ColorType | None = None,
    alpha: float = 1.0,
    label: str | None = "head direction",
    cmap: str | colors.Colormap = "hsv",
    axes: matplotlib.axes.Axes | None = None,
    figure: matplotlib.figure.Figure | None = None,
    figsize: tuple[float, float] = (8, 4),
    animate: bool = False,
    **ax_kwargs,
) -> (
    tuple[BehFigure, matplotlib.axes.Axes]
    | tuple[BehFigure, matplotlib.axes.Axes, Animate_plot]
):
    """Plot head direction using `head_direction_vector_labels`.

    Parameters
    ----------
    trk : Tracking
        The tracking object.
    head_direction_vector_labels : list
        Pair of bodyparts from where to get the head direction from. Default is
        ``["neck", "probe"]``.
    ang : str, optional
        Whether to plot in "deg" for degrees or in "rad" for radians. Default is "deg".
    smooth : bool, optional
        Whether or not to smooth the direction data. Default is ``False``.
    only_running_bouts : bool, optional
        If should plot only the running periods using
        :class:`physbeh.tracking.Tracking.get_running_bouts` function. Default
        is ``False``.
    plot_only_running_bouts : bool, optional
        Whether or not to plot a background color on periods of running bouts (and not
        only not plot non running bouts). This only takes effect if `only_running_bouts`
        is set to ``True``. Default is ``True``.
    color : color, optional
        `cmap` must be ``None`` for `color` to be used. Tuple of RGB(A) values for color
        of the line collection, if ``None``, uses the color defined by the label.
        Default is ``None``.
    alpha : float, optional
        The alpha value of the line collection. Default is ``1.0``.
    label : str, optional
        The label of the plot to appear in the figure legend. ``""`` will show no legend
        for this plot. Default is ``"head direction"``.
    cmap : str, optional
        The colormap to use for the plot, if ``None`` a `color` is going to be used as
        color of the plot. Default is ``"hsv"``.
    axes : matplotlib.axes.Axes, optional
        If ``None``, new axes is created in `figure`. Default is ``None``.
    figure : matplotlib.figure.Figure, optional
        If ``None``, new figure is created. Default is ``None``.
    figsize : tuple, optional
        Figure size, if ``figure=None``. Default is ``(12,6)``.
    animate : bool, optional
        If set to `True`, plots an animation with the video of the Tracking class.
        Default is ``False``.
    **ax_kwargs
        Additional keyword arguments to pass to the axes.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The matplotlib figure object.
    axes : tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]
        The matplotlib axes object.
    """
    head_direction_array, time_array, index = trk.get_direction_array(
        label0=head_direction_vector_labels[0],
        label1=head_direction_vector_labels[1],
        mode=ang,
        smooth=smooth,
        only_running_bouts=only_running_bouts,
    )

    if label is None:
        label = f"{head_direction_vector_labels[0]} $\\rightarrow$ "
        label += f"{head_direction_vector_labels[1]}"

    index_wrapped_dict = {"deg": 270, "rad": 270 / 360 * 2 * np.pi}
    index_wrapped_angles = np.where(
        np.append(np.abs(np.diff(head_direction_array)), 0) >= index_wrapped_dict[ang]
    )[0]
    index[index_wrapped_angles] = False

    ylabel_dict = {"deg": "Degree", "rad": "Radians"}
    ax_kwargs.setdefault("ylabel", ylabel_dict[ang])
    ax_kwargs.setdefault("cbar_label", ylabel_dict[ang])
    ax_kwargs.setdefault("xlabel", "time (s)")
    vrange = {"deg": (0, 360), "rad": (0, 2 * np.pi)}
    ax_kwargs.setdefault("vmin", vrange[ang][0])
    ax_kwargs.setdefault("vmax", vrange[ang][1])
    ax_kwargs.setdefault("legend__loc", "upper right")
    ax_kwargs.setdefault("grid__linestyle", "--")
    behfigure, axes, _ = plot_array(
        head_direction_array,
        time_array=time_array,
        index=index,
        label=label,
        cmap=cmap,
        only_running_bouts=only_running_bouts,
        color=get_label_color(trk, head_direction_vector_labels[0])
        if color is None
        else color,
        alpha=alpha,
        axes=axes,
        figure=figure,
        figsize=figsize,
        **ax_kwargs,
    )

    if cmap is not None:
        # take the legend object and remove the symbol only, not the entry
        legend = axes.get_legend()
        if legend is not None:
            legend.legend_handles[0].set_visible(False)
            legend.handlelength = 0

    if only_running_bouts and plot_only_running_bouts:
        plot_running_bouts(trk, axes=axes, set_axes=False)
    return behfigure, axes


@anim_decorator
def plot_head_direction_interval(
    trk: Tracking,
    deg=180,
    sigma=10.0,
    head_direction_vector_labels=["neck", "probe"],
    only_running_bouts=False,
    plot_only_running_bouts: bool = True,
    color: ColorType | None = None,
    alpha: float = 1.0,
    axes: matplotlib.axes.Axes | None = None,
    figure: matplotlib.figure.Figure | None = None,
    figsize: tuple[float, float] = (14, 7),
    animate_video=False,
    animate_fus=False,
    **ax_kwargs,
):
    """Plot the head direction interval for a given degree.

    The head direction interval can be seen as an activation signal for a given head
    direction.

    Parameters
    ----------
    trk : Tracking
        The tracking object containing the data.
    deg : int, optional
        The degree to plot the head direction interval for. Default is ``180``.
    sigma : float, optional
        The sigma value of the gaussian function. Default is ``10.0``.
    head_direction_vector_labels : list
        Pair of bodyparts from where to get the head direction from. Default is
        ``["neck", "probe"]``.
    only_running_bouts : bool, optional
        Whether to plot only the head direction intervals during running bouts. Default
        is ``False``.
    plot_only_running_bouts : bool, optional
        Whether or not to plot a background color on periods of running bouts (and not
        only not plot non running bouts). This only takes effect if `only_running_bouts`
        is set to ``True``. Default is ``True``.
    color : color, optional
        Tuple of RGB(A) values for color of the line collection, if ``None``, uses the
        color defined by the label. Default is ``None``.
    alpha : float, optional
        The alpha value of the line collection. Default is ``1.0``.
    axes : matplotlib.axes.Axes, optional
        If ``None``, new axes is created in `figure`. Default is ``None``.
    figure : matplotlib.figure.Figure, optional
        If ``None``, new figure is created. Default is ``None``.
    figsize : tuple, optional
        Figure size, if ``figure=none``. Default is ``(12,6)``.
    animate_video : bool, optional
        Whether to animate the plot with the video recording. Default is ``False``.
    animate_fus : bool, optional
        Whether to animate the plot with the functional Ultrasound video. Default is
        ``False``.
    **ax_kwargs
        Additional keyword arguments to pass to the axes.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The matplotlib figure object.
    axes : tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]
        The matplotlib axes object.
    """

    hd_interval_array, time_array, index = trk.get_degree_interval_hd(
        deg,
        label0=head_direction_vector_labels[0],
        label1=head_direction_vector_labels[1],
        only_running_bouts=only_running_bouts,
        sigma=sigma,
    )

    ax_kwargs.setdefault("ylabel", "a.u.")
    ax_kwargs.setdefault("xlabel", "time (s)")
    ax_kwargs.setdefault("legend__loc", "upper right")
    ax_kwargs.setdefault("grid__linestyle", "--")
    behfigure, axes, _ = plot_array(
        hd_interval_array,
        time_array=time_array,
        index=index,
        only_running_bouts=only_running_bouts,
        label=f"{deg} degrees",
        color=get_label_color(trk, head_direction_vector_labels[0])
        if color is None
        else color,
        axes=axes,
        figure=figure,
        figsize=figsize,
        alpha=alpha,
        **ax_kwargs,
    )

    if only_running_bouts and plot_only_running_bouts:
        plot_running_bouts(trk, axes=axes, set_axes=False)

    return behfigure, axes


def plot_occupancy(
    trk: Tracking,
    bins: int | Sequence[int] = 10,
    only_running_bouts: bool = True,
    axes: matplotlib.axes.Axes | None = None,
    figure: matplotlib.figure.Figure | None = None,
    figsize: tuple[float, float] = (8, 6),
) -> tuple[BehFigure, matplotlib.axes.Axes]:
    """Plot the occupancy of the animal in the arena.

    Parameters
    ----------
    trk : Tracking
        The tracking object.
    bins : int or [int, int], optional
        The bin specification:

        * If ``int``, the number of bins for the two dimensions (``nx=ny=bins``).
        * If ``[int, int]``, the number of bins in each dimension
            (``nx, ny = bins``).

        Default is ``10``.
    only_running_bouts : bool, optional
        Whether to plot only the occupancy during running bouts. Default is ``True``.
    axes : matplotlib.axes.Axes, optional
        If ``None``, new axes is created in `figure`. Default is ``None``.
    figure : matplotlib.figure.Figure, optional
        If ``None``, new figure is created. Default is ``None``.
    figsize : tuple, optional
        Figure size, if ``figure=none``. Default is ``(12,6)``.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The matplotlib figure object.
    axes : tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]
        The matplotlib axes object.
    """
    axes, behfigure = _check_ax_and_fig(axes, figure, figsize=figsize)

    H = trk.get_binned_position(bins=bins, only_running_bouts=only_running_bouts)
    H[0][H[0] == 0] = np.nan

    i = axes.pcolormesh(H[1], H[2], H[0].T)
    axes.invert_yaxis()
    axes.set_aspect("equal", "box")
    axes.set(xlabel="cm", ylabel="cm")
    behfigure.cbar = behfigure.figure.colorbar(i, ax=axes, label="count")

    return behfigure, axes
