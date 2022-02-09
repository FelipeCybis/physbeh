import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors
from matplotlib.cm import ScalarMappable

import numpy as np

from tracking_physmed.utils import get_line_collection, plot_color_wheel, get_cmap


def plot_speed(Trk_cls, bodypart="body", smooth=True, speed_cutout=0, only_running_bouts=False, ax=None, figsize=(12,5)):
    """Plot speed of given label.

    Parameters
    ----------
    Trk_cls : `Tracking` instance
    kwargs  : Keyword arguments can be passed to the get_speed method (see Tracking.get_speed?). They can be a matplotlib axes `ax` or a matplotlib figure `fig`. They can also `figsize` for matplotlib figure if this is not passed or they can be matplotlib axes parameters, such as `ylabel`, `title`, etc.

    Returns
    -------
    fig     : matplotlib.Figure
    ax      : matplotlib.Axes
    """
    speed_array, time_array, index, speed_units = Trk_cls.get_speed(bodypart=bodypart, smooth=smooth, speed_cutout=speed_cutout, only_running_bouts=only_running_bouts)

    lines = get_line_collection(time_array, speed_array, index)

    lc = LineCollection(
        lines,
        label=bodypart,
        linewidths=2,
        colors=Trk_cls.colors[bodypart],
    )

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    ax.add_collection(lc)

    if only_running_bouts == True:
        time_array = np.concatenate(time_array)
        speed_array = np.concatenate(speed_array)
        index = np.concatenate(index)
        Trk_cls.plot_running_bouts(ax)

    ax.plot(time_array[index], speed_array[index], ".", markersize=0)
    ax.set(ylabel=speed_units, xlabel="time (s)")
    ax.legend(loc="upper right")
    ax.grid(linestyle="--")

    return fig, ax


def plot_position_2d(
    Trk_cls,
    bodypart="body",
    color_collection_array=None,
    clim=None,
    head_direction=True,
    head_direction_vector_labels=["neck", "probe"],
    only_running_bouts=False,
    figsize=(8, 6),
    colormap="hsv",
    ax=None,
    ax_direction=True,
    ax_kwargs=None,
    fig=None,
):

    x_bp, _, index = Trk_cls.get_position_x(bodypart=bodypart)
    y_bp = Trk_cls.get_position_y(bodypart=bodypart)[0]
    if head_direction:
        index = (
            Trk_cls.Dataframe[Trk_cls.scorer][head_direction_vector_labels[0]][
                "likelihood"
            ].values
            > 0.8
        )

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
        ax_1.axis("equal")
        ax_1.invert_yaxis()

    if color_collection_array is not None:
        if clim is None:
            clim = (color_collection_array.min(), color_collection_array.max())
        
        cmap = get_cmap(name=colormap, n=200)
        norm = colors.BoundaryNorm(np.arange(clim[0], clim[1],(clim[1] - clim[0])/100), cmap.N)

        lc = LineCollection(lines, linewidths=3, cmap=cmap, norm=norm)
        lc.set_alpha(0.7)
        lc.set_array(color_collection_array[index])
        ax_1.set_position([0.12, 0.12, 0.7, 0.8])
        ax_2 = fig.add_axes(rect=[0.85, 0.12, 0.03, 0.8])
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=ax_2)

    elif head_direction:

        index = (
            Trk_cls.Dataframe[Trk_cls.scorer][head_direction_vector_labels[0]][
                "likelihood"
            ].values
            > 0.8
        )

        cmap = get_cmap(name=colormap, n=360)

        head_direction_array = Trk_cls.get_direction_array(
            label0=head_direction_vector_labels[0],
            label1=head_direction_vector_labels[1],
            mode="deg",
        )

        norm = colors.BoundaryNorm(np.arange(0, 360), cmap.N)

        lc = LineCollection(lines, linewidths=3, cmap=cmap, norm=norm)
        lc.set_array(head_direction_array[index])

        if ax_direction:
            fig.set_size_inches(14, 7.5)
            ax_1.set_position([0.12, 0.12, 0.5, 0.75])
            ax_2 = fig.add_axes(rect=[0.65, 0.26, 0.3, 0.48], projection="polar")
            plot_color_wheel(ax=ax_2, cmap=cmap)

    else:
        lc = LineCollection(lines, linewidths=3)
        lc.set_alpha(0.8)

    ax_1.add_collection(lc)
    ax_1.scatter(x_bp[index], y_bp[index], s=0)

    if ax_kwargs is not None:
        ax_1.set(**ax_kwargs)

    return fig, lc

def plot_likelihood(Trk, bodyparts='all', ax=None, fig=None, **ax_kwargs):
    """Plot likelihood for labels in each frame

    Parameters
    ----------
    Trk : `Tracking` instance
    bodyparts : list or str, optional
        Labels to be plotted, it can be a string, a list of strings or `all` for all labels. By default `all`
    ax : matplotlib Axes, optional
        If None, new axes is created in `fig`. By default ``None``
    fig : matplotlib Figure, optional
        If ``None``, new figure is created. By default ``None``

    Returns
    -------
    fig     : matplotlib.Figure
    """
        
    if isinstance(bodyparts, str):
        if bodyparts == "all":
            bodyparts = Trk.bodyparts
        else:
            bodyparts = [bodyparts]

    time_array = np.array(Trk.Dataframe.index) / Trk.fps
        
    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=(14,5))
        ax = fig.add_axes([0.1,0.125,0.75,0.775])
    
    for bp in bodyparts:
        lk = Trk.Dataframe[Trk.scorer][bp]['likelihood'].values
        
        ax.plot(time_array,lk,'.', markersize=4, color=Trk.colors[bp], label=bp, alpha=.6)
    
    ax.set(ylabel='likelihood', xlabel='frames', ylim=(-0.05,1.05))
    ax.set(**ax_kwargs)
    ax.legend(bbox_to_anchor=(1.02,1), loc="upper left")
    ax.grid(linestyle='--')

    return fig

def plot_position_x(Trk, bodyparts="all", ax=None, fig=None, figsize=(12,6), **ax_kwargs):
    """[summary]

    Parameters
    ----------
    Trk : [type]
        [description]
    bodyparts : [type]
        [description]
    ax : [type], optional
        [description], by default None
    fig : [type], optional
        [description], by default None
    """

    if isinstance(bodyparts, str):
        if bodyparts == "all":
            bodyparts = Trk.bodyparts
        else:
            bodyparts = [bodyparts]

    if ax == None:
        if fig is None:
            fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    
    for bp in bodyparts:
        x_bp, time_array, index = Trk.get_position_x(bodypart=bp)

        lines = get_line_collection(x_array=time_array, y_array=x_bp, index=index)
            
        lc = LineCollection(lines, label=bp, linewidths=2, colors=Trk.colors[bp])
        ax.add_collection(lc)

    ax.plot(time_array, x_bp,'.', markersize=0)

    ax.set(ylabel='X pixel', xlabel='time (s)')
    ax.set(**ax_kwargs)
    ax.legend(bbox_to_anchor=(1.02,1), loc="upper left")
    ax.grid(linestyle='--')

    return fig

def plot_position_y(Trk, bodyparts="all", ax=None, fig=None, figsize=(12,6), **ax_kwargs):
    """[summary]

    Parameters
    ----------
    Trk : [type]
        [description]
    bodyparts : [type]
        [description]
    ax : [type], optional
        [description], by default None
    fig : [type], optional
        [description], by default None
    """

    if isinstance(bodyparts, str):
        if bodyparts == "all":
            bodyparts = Trk.bodyparts
        else:
            bodyparts = [bodyparts]

    if ax == None:
        if fig is None:
            fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    
    for bp in bodyparts:
        y_bp, time_array, index = Trk.get_position_y(bodypart=bp)

        lines = get_line_collection(x_array=time_array, y_array=y_bp, index=index)
            
        lc = LineCollection(lines, label=bp, linewidths=2, colors=Trk.colors[bp])
        ax.add_collection(lc)

    ax.plot(time_array, y_bp,'.', markersize=0)

    ax.set(ylabel='Y pixel', xlabel='time (s)')
    ax.set(**ax_kwargs)
    ax.legend(bbox_to_anchor=(1.02,1), loc="upper left")
    ax.grid(linestyle='--')

    return fig

def plot_position(Trk,bodyparts="all", figsize=(12,6), fig=None):
        
    if bodyparts == "all":
        bodyparts = Trk.bodyparts
        
    if fig is None:
        fig = plt.figure(figsize=figsize)
    
    ax_x = fig.add_subplot(211)
    ax_y = fig.add_subplot(212)
    
    plot_position_x(Trk,bodyparts=bodyparts,ax=ax_x,xlabel='')
    plot_position_y(Trk,bodyparts=bodyparts,ax=ax_y)

    return fig