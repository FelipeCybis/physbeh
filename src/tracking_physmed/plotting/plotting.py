"""Useful plotting functions for Tracking objects."""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import colormaps as mpl_cm
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection

from tracking_physmed.plotting.animate2d_decorator import anim2d_decorator
from tracking_physmed.plotting.animate_decorator import anim_decorator
from tracking_physmed.plotting.animate_plot_fUS import Animate_video_fUS
from tracking_physmed.tracking import Tracking
from tracking_physmed.utils import _plot_color_wheel, get_line_collection


def _check_ax_and_fig(ax, fig, figsize):
    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        if fig is None:
            fig = ax.figure
        assert fig == ax.figure, "Axes and figure must be from the same object."
    return ax, fig


def _listify_bodyparts(trk, bodyparts):
    if isinstance(bodyparts, str):
        if bodyparts == "all":
            bodyparts = trk.labels
        else:
            bodyparts = [bodyparts]
    return bodyparts


def get_label_color(
    Trk: Tracking, bodypart: str, cmap_name: str = "plasma"
) -> tuple[float, float, float, float]:
    """Helper function to get the color of a bodypart label.

    Parameters
    ----------
    Trk : Tracking
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
    cmap = mpl_cm.get_cmap(cmap_name).resampled(len(Trk.labels))
    return cmap(Trk.labels.index(bodypart))


@anim_decorator
def plot_array(
    array: npt.NDArray,
    time_array: npt.NDArray | None = None,
    index: list[bool] | npt.NDArray | None = None,
    trk: Tracking | None = None,
    only_running_bouts: bool = False,
    label: str = "",
    alpha: float = 1.0,
    color: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
    ax=None,
    fig=None,
    figsize=(12, 6),
    plot_invisible_array: bool = True,
    **ax_kwargs,
):
    """Plot an array.

    Parameters
    ----------
    array : npt.ArrayLike
        The array to plot.
    ax : matplotlib Axes, optional
        If None, new axes is created in `fig`. Default is ``None``.
    figsize : tuple, optional
        Figure size, if `fig=None`. Default is ``(12,6)``.
    fig : matplotlib Figure, optional
        If ``None``, new figure is created. Default is ``None``.
    **ax_kwargs
        Keywords to pass to ``ax.set(**ax_kwargs)``.

    Returns
    -------
    figure : matplotlib.figure.Figure
    axes : matplotlib.axes.Axes
    """
    ax, fig = _check_ax_and_fig(ax, fig, figsize)

    if time_array is None:
        time_array = np.arange(len(array))
    if index is None:
        index = np.ones(len(array), dtype=bool)

    lines = get_line_collection(time_array, array, index)

    lc = LineCollection(
        lines,
        label=label,
        linewidths=2,
        alpha=alpha,
        colors=color,
    )
    ax.add_collection(lc)
    if only_running_bouts:
        time_array = np.concatenate(time_array)
        array = np.concatenate(array)
        index = np.concatenate(index)

    if plot_invisible_array:
        ax.plot(time_array[index], array[index], ".", markersize=0)
    ax.set(**ax_kwargs)

    return fig, ax


@anim_decorator
def plot_speed(
    trk: Tracking,
    bodypart="body",
    speed_axis="xy",
    euclidean=False,
    smooth=True,
    speed_cutout=0,
    only_running_bouts=False,
    plot_only_running_bouts: bool = True,
    alpha=1.0,
    ax=None,
    fig=None,
    figsize=(12, 6),
    animate_video=False,
    animate_fus=False,
    **ax_kwargs,
):
    """Plot speed of given label.

    Parameters
    ----------
    Trk_cls : :class:`tracking_physmed.tracking.Tracking` instance
    bodypart : str, optional
        Bodypart label. Default is "body"
    smooth : bool, optional
        If speed array is to be smoothed using a gaussian kernel. Default is ``True``.
    speed_cutout : int, optional
        If speed is to be thresholded by some value. Default is 0
    only_running_bouts : bool, optional
        If should plot only the running periods using
        :class:`tracking_physmed.tracking.Tracking.get_running_bouts` function. Default
        is ``False``.
    ax : matplotlib Axes, optional
        If None, new axes is created in `fig`. Default is ``None``.
    figsize : tuple, optional
        Figure size, if `fig=None`. Default is ``(12,6)``.
    fig : matplotlib Figure, optional
        If ``None``, new figure is created. Default is ``None``.
    animate_video : bool, optional
        If set to ``True``, plots an animation synched with the video of the Tracking
        class.
    animate_fus : bool, optional
        If set to ``True``, plots an animation synched with the associated scan (should
        be attached to Tracking).

    Returns
    -------
    tuple (matplotlib.Figure, matplotlib.Axes)
    """

    speed_array, time_array, index, speed_units = trk.get_speed(
        bodypart=bodypart,
        axis=speed_axis,
        euclidean_distance=euclidean,
        smooth=smooth,
        speed_cutout=speed_cutout,
        only_running_bouts=only_running_bouts,
    )

    ax_kwargs.setdefault("ylabel", f"animal speed ({speed_units})")
    ax_kwargs.setdefault("xlabel", "time (s)")
    fig, ax = plot_array(
        speed_array,
        time_array=time_array,
        index=index,
        only_running_bouts=only_running_bouts,
        label=bodypart,
        color=get_label_color(trk, bodypart),
        ax=ax,
        fig=fig,
        figsize=figsize,
        alpha=alpha,
        **ax_kwargs,
    )

    if only_running_bouts and plot_only_running_bouts:
        plot_running_bouts(trk, ax=ax)

    ax.legend(loc="upper right")
    ax.grid(linestyle="--")
    return fig, ax


@anim_decorator
def plot_wall_proximity(
    trk: Tracking,
    wall: str = "all",
    bodypart="neck",
    only_running_bouts=False,
    plot_only_running_bouts: bool = True,
    alpha=1.0,
    ax=None,
    fig=None,
    figsize=(14, 7),
    **ax_kwargs,
):
    """Plot proximity to specified wall.

    See :class:`tracking_physmed.tracking.Tracking.get_proximity_from_wall`. for more
    information.

    Parameters
    ----------
    trk : tracking_physmed.tracking.Tracking
        The tracking object.
    wall : str or list of str or tuple of str
        Wall to use for computations. Can be one of ("left", "right", "top", "bottom").
        Default is "left".
    bodypart : str, optional
        Bodypart to use for computations. Default "probe".
    only_running_bouts : bool, optional
        If should plot only the running periods using
        :class:`tracking_physmed.tracking.Tracking.get_running_bouts` function. Default
        is ``False``.
    ax : matplotlib Axes, optional
        If None, new axes is created in `fig`. Default is ``None``.
    figsize : tuple, optional
        Figure size, if `fig=None`. Default is ``(12,6)``.
    fig : matplotlib Figure, optional
        If ``None``, new figure is created. Default is ``None``.
    **ax_kwargs
        Keywords to pass to ``ax.set(**ax_kwargs)``.

    Returns
    -------
    tuple
        matplotlib Figure, matplotlib Axis
    """

    wall_proximity, time_array, index = trk.get_proximity_from_wall(
        wall=wall, bodypart=bodypart, only_running_bouts=only_running_bouts
    )

    ax_kwargs.setdefault("ylabel", f"Proximity from {wall} wall (a.u)")
    ax_kwargs.setdefault("xlabel", "time (s)")
    fig, ax = plot_array(
        wall_proximity,
        time_array=time_array,
        index=index,
        only_running_bouts=only_running_bouts,
        label=bodypart,
        color=get_label_color(trk, bodypart),
        ax=ax,
        fig=fig,
        figsize=figsize,
        alpha=alpha,
        **ax_kwargs,
    )

    if only_running_bouts and plot_only_running_bouts:
        plot_running_bouts(trk, ax=ax)

    ax.legend(loc="upper right")
    ax.grid(linestyle="--")
    return fig, ax


@anim_decorator
def plot_center_proximity(
    trk: Tracking,
    bodypart="probe",
    only_running_bouts=False,
    plot_only_running_bouts: bool = True,
    alpha=1.0,
    ax=None,
    fig=None,
    figsize=(14, 7),
    **ax_kwargs,
):
    """Plot proximity to the center of the environment.

    See :class:`tracking_physmed.tracking.Tracking.get_proximity_from_center`. for more
    information.

    Parameters
    ----------
    trk : tracking_physmed.tracking.Tracking
        The tracking object.
    bodypart : str, optional
        Bodypart to use for computations. Default "probe".
    only_running_bouts : bool, optional
        If should plot only the running periods using
        :class:`tracking_physmed.tracking.Tracking.get_running_bouts` function. Default
        is ``False``.
    ax : matplotlib Axes, optional
        If None, new axes is created in `fig`. Default is ``None``.
    figsize : tuple, optional
        Figure size, if `fig=None`. Default is ``(12,6)``.
    fig : matplotlib Figure, optional
        If ``None``, new figure is created. Default is ``None``.
    **ax_kwargs
        Keywords to pass to ``ax.set(**ax_kwargs)``.

    Returns
    -------
    tuple
        matplotlib Figure, matplotlib Axis
    """

    center_proximity, time_array, index = trk.get_proximity_from_center(
        bodypart=bodypart, only_running_bouts=only_running_bouts
    )

    ax_kwargs.setdefault("ylabel", "Proximity from center of stage (a.u)")
    ax_kwargs.setdefault("xlabel", "time (s)")
    fig, ax = plot_array(
        center_proximity,
        time_array=time_array,
        index=index,
        only_running_bouts=only_running_bouts,
        label=bodypart,
        color=get_label_color(trk, bodypart),
        ax=ax,
        fig=fig,
        figsize=figsize,
        alpha=alpha,
        **ax_kwargs,
    )

    if only_running_bouts and plot_only_running_bouts:
        plot_running_bouts(trk, ax=ax)

    ax.legend(loc="upper right")
    ax.grid(linestyle="--")
    return fig, ax


@anim_decorator
def plot_corner_proximity(
    trk: Tracking,
    corner: str = "top right",
    bodypart="probe",
    only_running_bouts=False,
    plot_only_running_bouts: bool = True,
    alpha=1.0,
    ax=None,
    fig=None,
    figsize=(14, 7),
    **ax_kwargs,
):
    """Plot proximity to specified corner.

    See :class:`tracking_physmed.tracking.Tracking.get_proximity_from_corner`. for more
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
        :class:`tracking_physmed.tracking.Tracking.get_running_bouts` function. Default
        is ``False``.
    ax : matplotlib Axes, optional
        If None, new axes is created in `fig`. Default is ``None``.
    figsize : tuple, optional
        Figure size, if `fig=None`. Default is ``(12,6)``.
    fig : matplotlib Figure, optional
        If ``None``, new figure is created. Default is ``None``.
    **ax_kwargs
        Keywords to pass to ``ax.set(**ax_kwargs)``.

    Returns
    -------
    tuple
        matplotlib Figure, matplotlib Axis
    """

    corner_proximity, time_array, index = trk.get_proximity_from_corner(
        corner=corner, bodypart=bodypart, only_running_bouts=only_running_bouts
    )

    ax_kwargs.setdefault("ylabel", f"Proximity from {corner} corner (a.u)")
    ax_kwargs.setdefault("xlabel", "time (s)")
    fig, ax = plot_array(
        corner_proximity,
        time_array=time_array,
        index=index,
        only_running_bouts=only_running_bouts,
        label=bodypart,
        color=get_label_color(trk, bodypart),
        ax=ax,
        fig=fig,
        figsize=figsize,
        alpha=alpha,
        **ax_kwargs,
    )

    if only_running_bouts and plot_only_running_bouts:
        plot_running_bouts(trk, ax=ax)

    ax.legend(loc="upper right")
    ax.grid(linestyle="--")
    return fig, ax


@anim_decorator
def plot_running_bouts(
    trk: Tracking,
    ax=None,
    figsize=(12, 6),
    fig=None,
    animate_video=False,
    animate_fus=False,
):
    """Plot the running periods of the animal.

    See :class:`tracking_physmed.tracking.Tracking.get_running_bouts`.

    Parameters
    ----------
    Trk : :class:`tracking_physmed.tracking.Tracking` instance
    ax : matplotlib Axes, optional
        If None, new axes is created in `fig`. By default ``None``
    figsize : tuple, optional
        Figure size, if `fig` is ``None``. By default `(12,6)`
    fig : matplotlib Figure, optional
        If ``None``, new figure is created. By default ``None``
    animate : bool, optional
        If set to ``True``, plots an animation alongside the video of the Tracking
        class.

    Returns
    -------
    fig : matplotlib.Figure
    """
    if not hasattr(trk, "running_bouts"):
        trk.get_running_bouts()

    ax, fig = _check_ax_and_fig(ax, fig, figsize)

    ax.fill_between(
        trk.time,
        0,
        1,
        where=trk.running_bouts,
        transform=ax.get_xaxis_transform(),
        color="orange",
        alpha=0.5,
    )

    return fig, ax


@anim2d_decorator
def plot_position_2d(
    Trk_cls: Tracking,
    bodypart: str = "body",
    color_collection_array: npt.ArrayLike | None = None,
    clim: tuple[float, float] | None = None,
    head_direction: bool = True,
    head_direction_vector_labels: tuple[str, str] | list[str] = ["neck", "probe"],
    only_running_bouts: bool = False,
    figsize: tuple[int | float, int | float] = (8, 6),
    colormap="hsv",
    colorbar: bool = True,
    colorbar_label: str | None = None,
    colorwheel=True,
    color="gray",
    ax=None,
    ax_kwargs=None,
    fig=None,
    animate=False,
):
    """Plots position of the animal in 2D coordinates.

    Parameters
    ----------
    Trk_cls : :class:`tracking_physmed.tracking.Tracking` instance
    bodypart : str, optional
        Bodypart label, by default "body"
    color_collection_array : [type], optional
        The array of values to be used for color mapping the line collection. Default is
        ``None``.
    clim : (float, float), optional
        The color limits for the color mapping. Default is ``None``.
    head_direction : bool, optional
        Whether or not to color the lines based on the head direction. Default is
        ``True``.
    head_direction_vector_labels : tuple or list of str, optional
        The labels of the vectors to be used for the head direction. Default is
        ``["neck", "probe"]``.
    only_running_bouts : bool, optional
        If should plot only the running periods using
        :class:`tracking_physmed.tracking.Tracking.get_running_bouts` function. Default
        is ``False``.
    figsize : tuple, optional
        The matplotlib figure size. Default is ``(8,6)``.
    colormap : str, optional
        [description], by default "hsv"
    ax : [type], optional
        [description], by default None
    ax_kwargs : [type], optional
        [description], by default None
    fig : matplotlib Figure, optional
        If ``None``, new figure is created. By default ``None``
    animate : bool, optional
        If set to `True`, plots an animation of the animal position in 2D.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The matplotlib figure object.
    axes : matplotlib.axes.Axes
        The matplotlib axes object.
    lines : matplotlib.collections.LineCollection
        The matplotlib collection of lines representing the segments of animal position
        from on epoch to another.
    """

    x_bp, _, index = Trk_cls.get_position_x(bodypart=bodypart)
    y_bp = Trk_cls.get_position_y(bodypart=bodypart)[0]

    if only_running_bouts:
        Trk_cls.get_running_bouts()
        index = Trk_cls.running_bouts

    lines = get_line_collection(x_array=x_bp, y_array=y_bp, index=index)

    ax_1 = ax
    if ax_1 is None:
        if fig is None:
            fig = plt.figure(figsize=figsize)

        ax_1 = fig.add_axes(rect=[0.125, 0.125, 0.775, 0.775])
        ax_1.set(
            xlabel="X pixel",
            ylabel="Y pixel",
            title="Animal position in the arena [bodypart: " + bodypart + "]",
        )
        ax_1.set_aspect("equal", "box")
        ax_1.invert_yaxis()
    else:
        if fig is None:
            fig = ax_1.figure
        assert fig == ax_1.figure, "Axes and figure must be from the same object."

    if color_collection_array is not None:
        if clim is None:
            clim = (color_collection_array.min(), color_collection_array.max())

        cmap = mpl_cm.get_cmap(colormap).resampled(200)
        norm = colors.BoundaryNorm(np.linspace(clim[0], clim[1], cmap.N), cmap.N)

        lc = LineCollection(lines, linewidths=3, cmap=cmap, norm=norm)
        lc.set_alpha(0.7)
        lc.set_array(color_collection_array[index])
        # ax_1.set_position([0.12, 0.12, 0.7, 0.8])
        # ax_2 = fig.add_axes(rect=[0.85, 0.12, 0.03, 0.8])
        if colorbar:
            cbar = fig.colorbar(
                ScalarMappable(norm=norm, cmap=cmap),
                ax=ax_1,
                label=colorbar_label,
            )
            # cbar.ax.locator_params(nbins=4)
            cbar.set_ticks(np.linspace(clim[0], clim[1], 4))
            cbar.minorticks_off()

    elif head_direction:
        index = Trk_cls.get_index(head_direction_vector_labels[0], Trk_cls.pcutout)
        if only_running_bouts:
            index = Trk_cls.running_bouts

        cmap = mpl_cm.get_cmap(colormap).resampled(360)

        head_direction_array, _, _ = Trk_cls.get_direction_array(
            label0=head_direction_vector_labels[0],
            label1=head_direction_vector_labels[1],
            mode="deg",
        )

        norm = colors.BoundaryNorm(np.arange(0, 360), cmap.N)

        lc = LineCollection(lines, linewidths=3, cmap=cmap, norm=norm)
        lc.set_array(head_direction_array[index])

        if colorwheel:
            fig.set_size_inches(14, 7.5)
            ax_1.set_position([0.12, 0.12, 0.5, 0.75])
            ax_2 = fig.add_axes(rect=[0.65, 0.26, 0.3, 0.48], projection="polar")
            _plot_color_wheel(ax=ax_2, cmap=cmap)
        elif colorbar:
            fig.colorbar(
                ScalarMappable(norm=norm, cmap=cmap),
                ax=ax_1,
                label="Head direction (deg)",
            )

    else:
        lc = LineCollection(lines, linewidths=3, color=color)
        lc.set_alpha(0.3)

    ax_1.add_collection(lc)
    ax_1.scatter(x_bp[index], y_bp[index], s=0)

    if ax_kwargs is not None:
        ax_1.set(**ax_kwargs)

    return fig, ax_1, lines


@anim_decorator
def plot_likelihood(
    trk: Tracking,
    bodyparts="all",
    ax=None,
    figsize=(12, 6),
    fig=None,
    animate_video=False,
    animate_fus=False,
    **ax_kwargs,
):
    """Plot likelihood for labels in each frame

    Parameters
    ----------
    trk : :class:`tracking_physmed.tracking.Tracking` instance
    bodyparts : list or str, optional
        Labels to be plotted, it can be a string, a list of strings or `"all"` for all
        labels. Default is ``"all"``.
    ax : matplotlib Axes, optional
        If None, new axes is created in `fig`. By default ``None``
    figsize : tuple, optional
        Figure size, if `fig` is ``None``. By default (12,6)
    fig : matplotlib Figure, optional
        If ``None``, new figure is created. By default ``None``
    animate : bool, optional
        If set to `True`, plots an animation synched with the video of the Tracking
        class.

    Returns
    -------
    fig     : matplotlib.Figure
    """

    bodyparts = _listify_bodyparts(trk, bodyparts)

    ax, fig = _check_ax_and_fig(ax, fig, figsize)

    for bp in bodyparts:
        lk = trk.get_likelihood(bodypart=bp)

        ax.plot(
            trk.time,
            lk,
            ".",
            markersize=4,
            color=get_label_color(trk, bp),
            label=bp,
            alpha=0.6,
        )

    ax.set(ylabel="likelihood", xlabel="frames", ylim=(-0.05, 1.05))
    ax.set(**ax_kwargs)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(linestyle="--")

    return fig, ax


@anim_decorator
def plot_position_x(
    trk: Tracking,
    bodyparts="all",
    ax=None,
    fig=None,
    figsize=(12, 6),
    animate_video=False,
    animate_fus=False,
    **ax_kwargs,
):
    """Plots X coordinates of requested `bodyparts`

    Parameters
    ----------
    trk : `Tracking` instance
    bodyparts : str or list of str, optional
        Bodypart labels, accepts string or list of strings or ``"all"`` for all labels.
        Default is ``"all"``.
    ax : matplotlib Axes, optional
        If None, new axes is created in `fig`. By default ``None``
    figsize : tuple, optional
        Figure size, if `fig` is ``None``. Default is ``(12,6)``.
    fig : matplotlib Figure, optional
        If ``None``, new figure is created. By default ``None``
    animate : bool, optional
        If set to ``True``, plots an animation synched with the video of the Tracking
        class.

    Returns
    -------
    fig     : matplotlib.Figure
    """

    bodyparts = _listify_bodyparts(trk, bodyparts)

    spatial_units = trk.space_units[bodyparts[0] + "_x"].units
    ax_kwargs.setdefault("ylabel", f"X position ({spatial_units})")
    ax_kwargs.setdefault("xlabel", "time (s)")
    for i, bp in enumerate(bodyparts):
        x_bp, time_array, index = trk.get_position_x(bodypart=bp)

        fig, ax = plot_array(
            x_bp,
            time_array,
            index,
            label=bp,
            ax=ax,
            fig=fig,
            figsize=figsize,
            color=get_label_color(trk, bp),
            plot_invisible_array=False if i == 0 else True,
            **ax_kwargs,
        )

    ax.legend(loc="upper right")
    ax.grid(linestyle="--")

    return fig, ax


@anim_decorator
def plot_position_y(
    trk: Tracking,
    bodyparts="all",
    ax=None,
    fig=None,
    figsize=(12, 6),
    animate_video=False,
    animate_fus=False,
    **ax_kwargs,
):
    """Plot Y coordinates of `bodyparts`.

    Parameters
    ----------
    trk : Tracking
        The tracking object.
    bodyparts : str or list of str, optional
        Bodypart labels, accepts string or list of strings or ``"all"`` for all labels.
        Default is ``"all"``.
    ax : matplotlib Axes, optional
        If None, new axes is created in `fig`. By default ``None``
    figsize : tuple, optional
        Figure size, if `fig` is ``None``. By default (12,6)
    fig : matplotlib Figure, optional
        If ``None``, new figure is created. By default ``None``
    animate : bool, optional
        If set to `True`, plots an animation synched with the video of the Tracking
        class.

    Returns
    -------
    fig     : matplotlib.Figure
    """

    bodyparts = _listify_bodyparts(trk, bodyparts)

    spatial_units = trk.space_units[bodyparts[0] + "_y"].units
    ax_kwargs.setdefault("ylabel", f"Y position ({spatial_units})")
    ax_kwargs.setdefault("xlabel", "time (s)")
    for i, bp in enumerate(bodyparts):
        y_bp, time_array, index = trk.get_position_y(bodypart=bp)

        fig, ax = plot_array(
            y_bp,
            time_array,
            index,
            label=bp,
            ax=ax,
            fig=fig,
            figsize=figsize,
            color=get_label_color(trk, bp),
            plot_invisible_array=False if i == 0 else True,
            **ax_kwargs,
        )

    ax.legend(loc="upper right")
    ax.grid(linestyle="--")

    return fig, ax


def plot_position(Trk, bodyparts="all", figsize=(12, 6), fig=None, **ax_kwargs):
    """Plot X and Y coordinates of requested `bodyparts` in two subplots.

    The top one is for X coordinates and bottom one is for Y coordinates

    Parameters
    ----------
    Trk : `Tracking` instance
    bodyparts : str or list of str, optional
        Bodypart labels, accepts string or list of strings or ``"all"`` for all labels.
        Default is ``"all"``.
    figsize : tuple, optional
        Figure size, if `fig` is ``None``. Default is ``(12,6)``.
    fig : matplotlib Figure, optional
        If ``None``, new figure is created. Default is ``None``.

    Returns
    -------
    matplotlib.Figure
    """

    if fig is None:
        fig = plt.figure(figsize=figsize)

    ax_x = fig.add_subplot(211)
    ax_y = fig.add_subplot(212)

    plot_position_x(Trk, bodyparts=bodyparts, ax=ax_x, xlabel="", **ax_kwargs)
    plot_position_y(Trk, bodyparts=bodyparts, ax=ax_y, **ax_kwargs)

    return fig, (ax_x, ax_y)


@anim_decorator
def plot_head_direction(
    Trk: Tracking,
    head_direction_vector_labels=["neck", "probe"],
    ang="deg",
    smooth=False,
    only_running_bouts=False,
    color_collection_array=None,
    clim=None,
    colormap="hsv",
    colorbar=True,
    colorbar_label=None,
    figsize=(12, 6),
    ax=None,
    fig=None,
    animate_video=False,
    animate_fus=False,
    **ax_kwargs,
):
    """Plot head direction using `head_direction_vector_labels`.

    Parameters
    ----------
    Trk : `Tracking` instance
    head_direction_vector_labels : list
        Pair of bodyparts from where to get the head direction from. By default
        ``["neck", "probe"]``.
    ang : str, optional
        If plotting in "deg" for degrees or in "rad" for radians. By default "deg"
    figsize : tuple, optional
        Figure size, if `fig` is ``None``. By default (12,6)
    ax : matplotlib Axes, optional
        If None, new axes is created in `fig`. By default ``None``
    fig : matplotlib Figure, optional
        If ``None``, new figure is created. By default ``None``
    animate : bool, optional
        If set to `True`, plots an animation synched with the video of the Tracking
        class. Default is ``False``.

    Returns
    -------
    fig     : matplotlib.Figure
    """

    index = Trk.get_index(head_direction_vector_labels[1], Trk.pcutout)
    if only_running_bouts:
        if not hasattr(Trk, "running_bouts"):
            Trk.get_running_bouts()
        index = Trk.running_bouts

    head_direction_array, _, _ = Trk.get_direction_array(
        label0=head_direction_vector_labels[0],
        label1=head_direction_vector_labels[1],
        mode=ang,
        smooth=smooth,
    )

    index_wrapped_dict = {"deg": 320, "rad": 320 / 360 * 2 * np.pi}
    index_wrapped_angles = np.where(
        np.insert(np.abs(np.diff(head_direction_array)), 0, 0)
        >= index_wrapped_dict[ang]
    )[0]
    index[index_wrapped_angles] = False

    lines = get_line_collection(
        x_array=Trk.time, y_array=head_direction_array, index=index
    )

    if color_collection_array is not None:
        if clim is None:
            clim = (color_collection_array.min(), color_collection_array.max())

        cmap = mpl_cm.get_cmap(colormap).resampled(200)
        norm = colors.BoundaryNorm(
            np.arange(clim[0], clim[1], (clim[1] - clim[0]) / 100), cmap.N
        )

        line_collection_array = color_collection_array[index]
    else:
        cmap = mpl_cm.get_cmap(colormap).resampled(360)
        norm_dict = {
            "deg": np.arange(0, 361),
            "rad": np.arange(0, 2 * np.pi * (1 + 1 / 360), 2 * np.pi / 360),
        }
        norm = colors.BoundaryNorm(norm_dict[ang], cmap.N)

        line_collection_array = head_direction_array[index]

    lc = LineCollection(lines, linewidths=2, cmap=cmap, norm=norm)
    if colormap is None:
        lc = LineCollection(lines, linewidths=2, colors="#1f77b4")
    lc.set_array(line_collection_array)

    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(rect=[0.1, 0.12, 0.8, 0.8])
    else:
        if fig is None:
            fig = ax.figure
        assert fig == ax.figure, "Axes and figure must be from the same instance"
    ax.add_collection(lc)

    if colorbar:
        cax = fig.add_axes(rect=[0.92, 0.12, 0.025, 0.8])
        fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap), cax=cax, label=colorbar_label
        )

    ax.plot(Trk.time, head_direction_array, ".", markersize=0)
    ax.grid(linestyle="--")
    ylabel_dict = {"deg": "Degree", "rad": "Radians"}
    ax.set(
        xlabel="time (s)",
        ylabel=ylabel_dict[ang],
        title="Head direction",
    )
    ax.set(**ax_kwargs)

    return fig, ax


@anim_decorator
def plot_head_direction_interval(
    Trk: Tracking,
    deg=180,
    sigma=10.0,
    only_running_bouts=False,
    figsize=(12, 6),
    fig=None,
    ax=None,
    animate_video=False,
    animate_fus=False,
    **ax_kwargs,
):
    """Plot the head direction interval for a given degree.

    The head direction interval can be seen as an activation signal for a given head
    direction.

    Parameters
    ----------
    Trk : Tracking
        The tracking object containing the data.
    deg : int, optional
        The degree to plot the head direction interval for. Default is ``180``.
    sigma : float, optional
        The sigma value of the gaussian function. Default is ``10.0``.
    only_running_bouts : bool, optional
        Whether to plot only the head direction intervals during running bouts. Default
        is ``False``.
    figsize : tuple, optional
        The size of the figure. Default is ``(12, 6)``.
    fig : matplotlig.figure.Figure, optional
        The figure to plot on. If ``None``, a new figure will be created. Default is
        ``None``.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If ``None``, a new axes will be created. Default is
        ``None``.
    animate_video : bool, optional
        Whether to animate the plot with the video recording. Default is ``False``.
    animate_fus : bool, optional
        Whether to animate the plot with the functional Ultrasound video. Default is
        ``False``.
    **ax_kwargs
        Additional keyword arguments to pass to the axes.

    Returns
    -------
    Figure, Axes
        The figure and axes objects.
    """

    hd_interval_array, time_array, index = Trk.get_degree_interval_hd(
        deg, only_running_bouts=only_running_bouts, sigma=sigma
    )

    lines = get_line_collection(time_array, hd_interval_array, index)

    lc = LineCollection(
        lines,
        linewidths=2,
    )

    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(rect=[0.1, 0.12, 0.8, 0.8])
    else:
        if fig is None:
            fig = ax.figure
        assert fig == ax.figure, "Axes and figure must be from the same instance"

    ax.add_collection(lc)

    if only_running_bouts:
        time_array = np.concatenate(time_array)
        hd_interval_array = np.concatenate(hd_interval_array)
        index = np.concatenate(index)
        plot_running_bouts(Trk, ax=ax)

    ax.plot(
        time_array[index],
        hd_interval_array[index],
        ".",
        markersize=0,
        label=f"{deg} degrees",
    )
    ax.set(ylabel="a.u", xlabel="time (s)")
    ax.legend(loc="upper right")
    ax.grid(linestyle="--")
    ax.set(**ax_kwargs)

    return fig, ax


def plot_occupancy(
    Trk: Tracking, bins=40, only_running_bouts=True, figsize=(8, 7), fig=None, ax=None
):
    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    H = Trk.get_binned_position(bins=40, only_running_bouts=True)
    H[0][H[0] == 0] = np.nan

    i = ax.pcolormesh(H[1], H[2], H[0].T)
    ax.invert_yaxis()
    ax.set_aspect("equal", "box")
    ax.set(xlabel="cm", ylabel="cm")
    fig.colorbar(i, ax=ax, label="count")

    return fig, ax


def animation_behavior_fus(Trk, fig=None):
    assert (
        Trk.scan is not None
    ), "Tracking class needs to have an attached scan for this animation"

    return Animate_video_fUS(tracking=Trk, fig=fig)
