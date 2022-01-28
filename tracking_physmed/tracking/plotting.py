import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors

import numpy as np

from tracking_physmed.utils import get_line_collection, plot_color_wheel, get_cmap


def plot_speed(Trk_cls, **kwargs):
    """Plot speed of given label.

    Parameters
    ----------
    Trk_cls : Tracking class
    kwargs  : Keyword arguments can be passed to the get_speed method (see Tracking.get_speed?). They can be a matplotlib axes `ax` or a matplotlib figure `fig`. They can also `figsize` for matplotlib figure if this is not passed or they can be matplotlib axes parameters, such as `ylabel`, `title`, etc.

    Returns
    -------
    fig     : matplotlib.Figure
    ax      : matplotlib.Axes
    """

    speed_kwargs = {
        "bodypart": kwargs.pop("bodypart", "body"),
        "smooth": kwargs.pop("smooth", True),
        "speed_cutout": kwargs.pop("speed_cutout", 0),
        "only_running_bouts": kwargs.pop("only_running_bouts", False),
    }
    speed_array, time_array, index, speed_units = Trk_cls.get_speed(**speed_kwargs)

    lines = get_line_collection(time_array, speed_array, index)

    lc = LineCollection(
        lines,
        label=speed_kwargs["bodypart"],
        linewidths=2,
        colors=Trk_cls.colors[speed_kwargs["bodypart"]],
    )

    ax = kwargs.pop("ax", None)
    if ax is None:
        fig = kwargs.pop("fig", plt.figure(figsize=kwargs.pop("figsize", (12, 5))))
        ax = fig.add_subplot(111)

    ax.add_collection(lc)

    if speed_kwargs["only_running_bouts"] == True:
        time_array = np.concatenate(time_array)
        speed_array = np.concatenate(speed_array)
        index = np.concatenate(index)
        Trk_cls.plot_running_bouts(ax)

    ax.plot(time_array[index], speed_array[index], ".", markersize=0)
    ax.set(ylabel=speed_units, xlabel="time (s)")
    ax.set(**kwargs)
    ax.legend(loc="upper right")
    ax.grid(linestyle="--")

    return fig, ax


def plot_position_2d(
    Trk_cls,
    bodypart="body",
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

    if head_direction == False:
        lc = LineCollection(lines, linewidths=3)
        lc.set_alpha(0.8)

    else:

        index = (
            Trk_cls.Dataframe[Trk_cls.scorer][head_direction_vector_labels[0]][
                "likelihood"
            ].values
            > 0.8
        )
        lines = get_line_collection(x_array=x_bp, y_array=y_bp, index=index)

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
            ax_1.set_position([0.1, 0.12, 0.5, 0.75])
            ax_2 = fig.add_axes(rect=[0.65, 0.26, 0.3, 0.48], projection="polar")
            plot_color_wheel(ax=ax_2, cmap=cmap)

    ax_1.add_collection(lc)
    ax_1.scatter(x_bp[index], y_bp[index], s=0)

    if ax_kwargs is not None:
        ax_1.set(**ax_kwargs)

    return fig, lc
