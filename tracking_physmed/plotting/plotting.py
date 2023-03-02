import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors
from matplotlib.cm import ScalarMappable

import numpy as np

from ..utils import get_line_collection, _plot_color_wheel, get_cmap
from .animate_decorator import anim_decorator
from .animate2d_decorator import anim2d_decorator
from .animate_plot_fUS import Animate_video_fUS


@anim_decorator
def plot_speed(
    Trk_cls,
    bodypart="body",
    speed_axis="xy",
    euclidean=False,
    smooth=True,
    speed_cutout=0,
    only_running_bouts=False,
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
    Trk_cls : `Tracking` instance
    bodypart : str, optional
        Bodypart label. By default `body`
    smooth : bool, optional
        If speed array is to be smoothed using a gaussian kernel. By default `True`
    speed_cutout : int, optional
        If speed is to be thresholded by some value. By default `0`
    only_running_bouts : bool, optional
        If should plot only the running periods using `Tracking.get_running_bouts` function. By default `False`
    ax : matplotlib Axes, optional
        If None, new axes is created in `fig`. By default ``None``
    figsize : tuple, optional
        Figure size, if `fig` is ``None``. By default `(12,6)`
    fig : matplotlib Figure, optional
        If ``None``, new figure is created. By default ``None``
    animate_video : bool, optional
        If set to `True`, plots an animation synched with the video of the Tracking class.
    animate_fus : bool, optional
        If set to `True`, plots an animation synched with the associated scan (should be attached to Tracking).

    Returns
    -------
    tuple (matplotlib.Figure, matplotlib.Axes)
    """

    speed_array, time_array, index, speed_units = Trk_cls.get_speed(
        bodypart=bodypart,
        axis=speed_axis,
        euclidean_distance=euclidean,
        smooth=smooth,
        speed_cutout=speed_cutout,
        only_running_bouts=only_running_bouts,
    )

    lines = get_line_collection(time_array, speed_array, index)

    lc = LineCollection(
        lines,
        label=bodypart,
        linewidths=2,
        colors=Trk_cls.colors[bodypart],
    )

    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    ax.add_collection(lc)

    if only_running_bouts:
        time_array = np.concatenate(time_array)
        speed_array = np.concatenate(speed_array)
        index = np.concatenate(index)
        plot_running_bouts(Trk_cls, ax=ax)

    ax.plot(time_array[index], speed_array[index], ".", markersize=0)
    ax.set(ylabel=speed_units, xlabel="time (s)")
    ax.legend(loc="upper right")
    ax.grid(linestyle="--")
    ax.set(**ax_kwargs)

    return fig, ax


@anim_decorator
def plot_wall_proximity(
    Trk,
    wall,
    bodypart="probe",
    only_running_bouts=False,
    ax=None,
    fig=None,
    figsize=(14, 7),
    **ax_kwargs,
):

    wall_proximity, time_array, index = Trk.get_proximity_from_wall(
        wall=wall, bodypart=bodypart, only_running_bouts=only_running_bouts
    )

    lines = get_line_collection(time_array, wall_proximity, index)

    lc = LineCollection(
        lines,
        label=bodypart,
        linewidths=2,
        colors=Trk.colors[bodypart],
    )

    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    ax.add_collection(lc)

    if only_running_bouts:
        time_array = np.concatenate(time_array)
        wall_proximity = np.concatenate(wall_proximity)
        index = np.concatenate(index)
        plot_running_bouts(Trk, ax=ax)

    ax.plot(time_array[index], wall_proximity[index], ".", markersize=0)
    ax.set(ylabel=f"Proximity from {wall} wall (a.u)", xlabel="time (s)")
    ax.legend(loc="upper right")
    ax.grid(linestyle="--")
    ax.set(**ax_kwargs)

    return fig, ax


@anim_decorator
def plot_center_proximity(
    Trk,
    bodypart="probe",
    only_running_bouts=False,
    ax=None,
    fig=None,
    figsize=(14, 7),
    **ax_kwargs,
):

    center_proximity, time_array, index = Trk.get_proximity_from_center(
        bodypart=bodypart, only_running_bouts=only_running_bouts
    )

    lines = get_line_collection(time_array, center_proximity, index)

    lc = LineCollection(
        lines,
        label=bodypart,
        linewidths=2,
        colors=Trk.colors[bodypart],
    )

    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    ax.add_collection(lc)

    if only_running_bouts:
        time_array = np.concatenate(time_array)
        center_proximity = np.concatenate(center_proximity)
        index = np.concatenate(index)
        plot_running_bouts(Trk, ax=ax)

    ax.plot(time_array[index], center_proximity[index], ".", markersize=0)
    ax.set(ylabel=f"Proximity from center of stage (a.u)", xlabel="time (s)")
    ax.legend(loc="upper right")
    ax.grid(linestyle="--")
    ax.set(**ax_kwargs)

    return fig, ax


@anim_decorator
def plot_corner_proximity(
    Trk,
    corner,
    bodypart="probe",
    only_running_bouts=False,
    ax=None,
    fig=None,
    figsize=(14, 7),
    **ax_kwargs,
):

    corner_proximity, time_array, index = Trk.get_proximity_from_corner(
        corner=corner, bodypart=bodypart, only_running_bouts=only_running_bouts
    )

    lines = get_line_collection(time_array, corner_proximity, index)

    lc = LineCollection(
        lines,
        label=bodypart,
        linewidths=2,
        colors=Trk.colors[bodypart],
    )

    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    ax.add_collection(lc)

    if only_running_bouts:
        time_array = np.concatenate(time_array)
        corner_proximity = np.concatenate(corner_proximity)
        index = np.concatenate(index)
        plot_running_bouts(Trk, ax=ax)

    ax.plot(time_array[index], corner_proximity[index], ".", markersize=0)
    ax.set(ylabel=f"Proximity from {corner} corner (a.u)", xlabel="time (s)")
    ax.legend(loc="upper right")
    ax.grid(linestyle="--")
    ax.set(**ax_kwargs)

    return fig, ax


@anim_decorator
def plot_running_bouts(
    Trk, ax=None, figsize=(12, 6), fig=None, animate_video=False, animate_fus=False
):
    """Plots the running periods of the animal automatically computed by `Tracking.get_running_bouts`.

    Parameters
    ----------
    Trk : `Tracking` instance
    ax : matplotlib Axes, optional
        If None, new axes is created in `fig`. By default ``None``
    figsize : tuple, optional
        Figure size, if `fig` is ``None``. By default `(12,6)`
    fig : matplotlib Figure, optional
        If ``None``, new figure is created. By default ``None``
    animate : bool, optional
        If set to `True`, plots an animation synched with the video of the Tracking class.

    Returns
    -------
    fig : matplotlib.Figure
    """
    if not hasattr(Trk, "running_bouts"):
        Trk.get_running_bouts()

    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.set(xlabel="time (s)")

    ax.fill_between(Trk.time, 0, 1, where=Trk.running_bouts, transform=ax.get_xaxis_transform(), color="orange", alpha=0.5)

    return fig, ax


@anim2d_decorator
def plot_position_2d(
    Trk_cls,
    bodypart="body",
    absolute=False,
    color_collection_array=None,
    clim=None,
    head_direction=True,
    head_direction_vector_labels=["neck", "probe"],
    only_running_bouts=False,
    figsize=(8, 6),
    colormap="hsv",
    colorbar=True,
    colorbar_label=None,
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
    Trk_cls : `Tracking` instance
    bodypart : str, optional
        Bodypart label, by default "body"
    color_collection_array : [type], optional
        [description], by default None
    clim : [type], optional
        [description], by default None
    head_direction : bool, optional
        [description], by default True
    head_direction_vector_labels : list, optional
        [description], by default ["neck", "probe"]
    only_running_bouts : bool, optional
        [description], by default False
    figsize : tuple, optional
        [description], by default (8, 6)
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
    tuple (matplotlib.Figure, LineCollection)
    """

    x_bp, _, index = Trk_cls.get_position_x(bodypart=bodypart, absolute=absolute)
    y_bp = Trk_cls.get_position_y(bodypart=bodypart, absolute=absolute)[0]

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

    if color_collection_array is not None:
        if clim is None:
            clim = (color_collection_array.min(), color_collection_array.max())

        cmap = get_cmap(name=colormap, n=200)
        norm = colors.BoundaryNorm(
            np.linspace(clim[0], clim[1], cmap.N), cmap.N
        )

        lc = LineCollection(lines, linewidths=3, cmap=cmap, norm=norm)
        lc.set_alpha(0.7)
        lc.set_array(color_collection_array[index])
        # ax_1.set_position([0.12, 0.12, 0.7, 0.8])
        # ax_2 = fig.add_axes(rect=[0.85, 0.12, 0.03, 0.8])
        if colorbar:
            cbar = fig.colorbar(
                ScalarMappable(norm=norm, cmap=cmap), ax=ax_1, label=colorbar_label,
            )
            # cbar.ax.locator_params(nbins=4)
            cbar.set_ticks(np.linspace(clim[0], clim[1], 4))
            cbar.minorticks_off()

    elif head_direction:

        index = Trk_cls.get_index(head_direction_vector_labels[0], Trk_cls.pcutout)
        if only_running_bouts:
            index = Trk_cls.running_bouts

        cmap = get_cmap(name=colormap, n=360)

        head_direction_array, _ = Trk_cls.get_direction_array(
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
    Trk,
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
    Trk : `Tracking` instance
    bodyparts : list or str, optional
        Labels to be plotted, it can be a string, a list of strings or `"all"` for all labels. By default `"all"`
    ax : matplotlib Axes, optional
        If None, new axes is created in `fig`. By default ``None``
    figsize : tuple, optional
        Figure size, if `fig` is ``None``. By default (12,6)
    fig : matplotlib Figure, optional
        If ``None``, new figure is created. By default ``None``
    animate : bool, optional
        If set to `True`, plots an animation synched with the video of the Tracking class.

    Returns
    -------
    fig     : matplotlib.Figure
    """

    if isinstance(bodyparts, str):
        if bodyparts == "all":
            bodyparts = Trk.bodyparts
        else:
            bodyparts = [bodyparts]

    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.1, 0.125, 0.75, 0.775])

    for bp in bodyparts:
        lk = Trk.Dataframe[Trk.scorer][bp]["likelihood"].values

        ax.plot(
            Trk.time, lk, ".", markersize=4, color=Trk.colors[bp], label=bp, alpha=0.6
        )

    ax.set(ylabel="likelihood", xlabel="frames", ylim=(-0.05, 1.05))
    ax.set(**ax_kwargs)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(linestyle="--")

    return fig, ax


@anim_decorator
def plot_position_x(
    Trk,
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
    Trk : `Tracking` instance
    bodyparts : str or list of str, optional
        Bodypart labels, accepts string or list of strings or `"all"` for all labels. By default `"all"`
    ax : matplotlib Axes, optional
        If None, new axes is created in `fig`. By default ``None``
    figsize : tuple, optional
        Figure size, if `fig` is ``None``. By default (12,6)
    fig : matplotlib Figure, optional
        If ``None``, new figure is created. By default ``None``
    animate : bool, optional
        If set to `True`, plots an animation synched with the video of the Tracking class.

    Returns
    -------
    fig     : matplotlib.Figure
    """

    if isinstance(bodyparts, str):
        if bodyparts == "all":
            bodyparts = Trk.bodyparts
        else:
            bodyparts = [bodyparts]

    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    for bp in bodyparts:
        x_bp, time_array, index = Trk.get_position_x(bodypart=bp)

        lines = get_line_collection(x_array=Trk.time, y_array=x_bp, index=index)

        lc = LineCollection(lines, label=bp, linewidths=2, colors=Trk.colors[bp])
        ax.add_collection(lc)

    ax.plot(Trk.time, x_bp, ".", markersize=0)

    ax.set(ylabel="X pixel", xlabel="time (s)")
    ax.set(**ax_kwargs)
    # ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.legend(loc="upper right")
    ax.grid(linestyle="--")

    return fig, ax


@anim_decorator
def plot_position_y(
    Trk,
    bodyparts="all",
    ax=None,
    fig=None,
    figsize=(12, 6),
    animate_video=False,
    animate_fus=False,
    **ax_kwargs,
):
    """Plots Y coordinates of requested `bodyparts`

    Parameters
    ----------
    Trk : `Tracking` instance
    bodyparts : str or list of str, optional
        Bodypart labels, accepts string or list of strings or `"all"` for all labels. By default `"all"`
    ax : matplotlib Axes, optional
        If None, new axes is created in `fig`. By default ``None``
    figsize : tuple, optional
        Figure size, if `fig` is ``None``. By default (12,6)
    fig : matplotlib Figure, optional
        If ``None``, new figure is created. By default ``None``
    animate : bool, optional
        If set to `True`, plots an animation synched with the video of the Tracking class.

    Returns
    -------
    fig     : matplotlib.Figure
    """

    if isinstance(bodyparts, str):
        if bodyparts == "all":
            bodyparts = Trk.bodyparts
        else:
            bodyparts = [bodyparts]

    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    for bp in bodyparts:
        y_bp, time_array, index = Trk.get_position_y(bodypart=bp)

        lines = get_line_collection(x_array=Trk.time, y_array=y_bp, index=index)

        lc = LineCollection(lines, label=bp, linewidths=2, colors=Trk.colors[bp])
        ax.add_collection(lc)

    ax.plot(Trk.time, y_bp, ".", markersize=0)

    ax.set(ylabel="Y pixel", xlabel="time (s)")
    ax.set(**ax_kwargs)
    # ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.legend(loc="upper right")
    ax.grid(linestyle="--")

    return fig, ax


def plot_position(Trk, bodyparts="all", figsize=(12, 6), fig=None, **ax_kwargs):
    """Plots X and Y coordinates of requested `bodyparts` in two subplots, top one is for X coordinates and bottom one is for Y coordinates

    Parameters
    ----------
    Trk : `Tracking` instance
    bodyparts : str or list of str, optional
        Bodypart labels, accepts string or list of strings or `"all"` for all labels. By default `"all"`
    figsize : tuple, optional
        Figure size, if `fig` is `None`. By default (12,6)
    fig : matplotlib Figure, optional
        If `None`, new figure is created. By default `None`

    Returns
    -------
    fig     : matplotlib.Figure
    """

    if bodyparts == "all":
        bodyparts = Trk.bodyparts

    if fig is None:
        fig = plt.figure(figsize=figsize)

    ax_x = fig.add_subplot(211)
    ax_y = fig.add_subplot(212)

    plot_position_x(Trk, bodyparts=bodyparts, ax=ax_x, xlabel="", **ax_kwargs)
    plot_position_y(Trk, bodyparts=bodyparts, ax=ax_y, **ax_kwargs)

    return fig


@anim_decorator
def plot_head_direction(
    Trk,
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
    """Plots head direction using `head_direction_vector_labels` to compute the head direction vector.

    Parameters
    ----------
    Trk : `Tracking` instance
    head_direction_vector_labels : list
        Pair of bodyparts from where to get the head direction from. By default ["neck", "probe"]
    ang : str, optional
        If plotting in "deg" for degrees or in "rad" for radians. By default "deg"
    figsize : tuple, optional
        Figure size, if `fig` is ``None``. By default (12,6)
    ax : matplotlib Axes, optional
        If None, new axes is created in `fig`. By default ``None``
    fig : matplotlib Figure, optional
        If ``None``, new figure is created. By default ``None``
    animate : bool, optional
        If set to `True`, plots an animation synched with the video of the Tracking class.

    Returns
    -------
    fig     : matplotlib.Figure
    """

    index = Trk.get_index(head_direction_vector_labels[1], Trk.pcutout)
    if only_running_bouts:
        if not hasattr(Trk, "running_bouts"):
            Trk.get_running_bouts()
        index = Trk.running_bouts

    head_direction_array, _ = Trk.get_direction_array(
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

        cmap = get_cmap(name=colormap, n=200)
        norm = colors.BoundaryNorm(
            np.arange(clim[0], clim[1], (clim[1] - clim[0]) / 100), cmap.N
        )

        line_collection_array = color_collection_array[index]
    else:

        cmap = get_cmap(name=colormap, n=360)
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
    Trk,
    deg=180,
    only_running_bouts=False,
    figsize=(12, 6),
    fig=None,
    ax=None,
    animate_video=False,
    animate_fus=False,
    **ax_kwargs,
):

    hd_interval_array, time_array, index = Trk.get_degree_interval_hd(
        deg, only_running_bouts=only_running_bouts
    )

    lines = get_line_collection(time_array, hd_interval_array, index)

    lc = LineCollection(
        lines,
        linewidths=2,
    )

    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    ax.add_collection(lc)

    if only_running_bouts == True:
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
    Trk, bins=40, only_running_bouts=True, figsize=(8, 7), fig=None, ax=None
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
